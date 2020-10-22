'''Train Adversarially Robust Models with Feature Scattering'''
from __future__ import print_function
from argparse import ArgumentParser
import logging
import time
import random
import numpy as np
import copy
import os
import datetime
 
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn

from dev.loaders import LibriSpeech4SpeakerRecognition
from dev.models import RawAudioCNN
from torch.autograd.gradcheck import zero_gradients 
from torch.autograd import Variable 
from hparams import hp
from tqdm import tqdm  
from models.wideresnet1 import WideResNet

import utils
from utils import softCrossEntropy
from utils import one_hot_tensor


from feature_scatter_attack_hybrid import Attack_FeaScatter 
#from feature_scatter_attack_fused_new1 import Attack_FeaScatter

import pdb, sys, os
from art.classifiers import PyTorchClassifier
from dev.factories import AttackerFactory
from dev.defences import EnsembleAdversarialTrainer
from art.attacks import FastGradientMethod

def _is_cuda_available():
    return torch.cuda.is_available()


def _get_device():
    return torch.device("cuda" if _is_cuda_available() else "cpu")

def resolve_attacker_args(args, eps, eps_step):
    targeted = False
    if args.attack == "NoiseAttack":
        kwargs = {"eps": eps}
    elif args.attack in ["FastGradientMethod", "ProjectedGradientDescent"]:
        kwargs = {"eps": eps, "eps_step": eps_step, "targeted": targeted}
    else:
        raise NotImplementedError
    return kwargs

def train_fun(epoch, net, args, optimizer, trainloader, device):
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    # update learning rate
    if epoch < args.decay_epoch1:
        lr = args.lr
    elif epoch < args.decay_epoch2:
        lr = args.lr * args.decay_rate
    else:
        lr = args.lr * args.decay_rate * args.decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    def get_acc(outputs, targets):
        _, predicted = outputs.max(1)
        total = targets.size(0)
        correct = predicted.eq(targets).sum().item()
        acc = 1.0 * correct / total
        return acc

    iterator = tqdm(trainloader, ncols=0, leave=False)
    a2 = []
    b2 = []
    c2 = []
    for batch_idx, (inputs, targets) in enumerate(iterator):
        start_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)

        a = inputs.shape
        b = targets.shape

        adv_acc = 0

        optimizer.zero_grad()
        #pdb.set_trace()

        # forward
        outputs, loss_fs, _ = net(inputs.detach(), targets)

        optimizer.zero_grad()
        loss = loss_fs
        loss.backward()

        optimizer.step()

        train_loss = loss.item()
        #pdb.set_trace()

        duration = time.time() - start_time
        if batch_idx % args.log_step == 0:
            if adv_acc == 0:
                adv_acc = get_acc(outputs, targets)
            iterator.set_description(str(adv_acc))

            nat_outputs, _ = net(inputs, targets, attack=False)
            nat_acc = get_acc(nat_outputs, targets)

            print(
                "epoch %d, step %d, lr %.4f, duration %.2f, training nat acc %.2f, training adv acc %.2f, training adv loss %.4f"
                % (epoch, batch_idx, lr, duration, 100 * nat_acc,
                   100 * adv_acc, train_loss))
        
            a2.append(100 * nat_acc)
            b2.append(100 * adv_acc) 
            c2.append(train_loss)

    if epoch % args.save_epochs == 0 or epoch >= args.max_epoch - 2:
        print('Saving..')
        f_path = os.path.join(args.model_dir, ('checkpoint-%s' % epoch))
        state = {'net': net.state_dict(),
                 # 'optimizer': optimizer.state_dict()
                 }
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        torch.save(state, f_path)

    if epoch >= 0:
        print('Saving latest @ epoch %s..' % (epoch))
        f_path = os.path.join(args.model_dir, 'latest')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        torch.save(state, f_path)
    return a2, b2, c2

def main(args):

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0
    n_epochs = args.max_epoch

    # Step 0: parse args and init logger
    logging.basicConfig(level=logging.INFO)

    generator_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 8,
        'pin_memory': True
    }

    # Step 1: load data set
    train_data = LibriSpeech4SpeakerRecognition(
        root=hp.data_root,
        url=hp.data_subset,
        train_speaker_ratio=hp.train_speaker_ratio,
        train_utterance_ratio=hp.train_utterance_ratio,
        subset="train",
        project_fs=hp.sr,
        wav_length=args.wav_length,
    )
    train_generator = torch.utils.data.DataLoader(
        train_data,
        **generator_params,
    )

    # Step 2: prepare training
    device = _get_device()
    logging.info(device)

    if args.model_type=='cnn':
        basic_net = RawAudioCNN(num_class=251)
    elif args.model_type=='wideresnet':
        basic_net = WideResNet(depth=28, num_classes=251, widen_factor=10)
    else:
        logging.error('Please provide a valid model architecture type!')
        sys.exit(-1)
        
    def print_para(net):
        for name, param in net.named_parameters():
            if param.requires_grad:
               print(name)
               print(param.data)
            break

    basic_net = basic_net.to(device)

    # config for feature scatter
    config_feature_scatter = {
        'train': True,
        #'epsilon': 8.0 / 255 * 2,
        'epsilon': args.epsilon,
        'num_steps': 10,
        #'step_size': 8.0 / 255 * 2,
        'step_size': args.epsilon / 5, 
        'random_start': True,
        'ls_factor': 0.7,
    }
    # pdb.set_trace()
    if args.adv_mode.lower() == 'feature_scatter':
        print('-----Feature Scatter mode -----')
        net = Attack_FeaScatter(basic_net, config_feature_scatter)
        # pdb.set_trace()
        #print(net)
    else:
        print('-----OTHER_ALGO mode -----')
        raise NotImplementedError("Please implement this algorithm first!")

    #if device == 'cuda':
     #   net = torch.nn.DataParallel(net)
      #  cudnn.benchmark = True

   #multi-gpu code, if needed in future
   #if torch.cuda.device_count() > 1:
   #    print("Let's use", torch.cuda.device_count(), "GPUs!")
   #    model = nn.DataParallel(model)
 
    #if _is_cuda_available():
     #   net.to(device)
    #criterion = torch.nn.CrossEntropyLoss()

    if args.optimizer=='sgd':
        #optimizer = torch.optim.SGD(model.parameters(), lr=2e-3, momentum=0.9)
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer=='adam':
        print()
        print('Using Adam optimizer\n')
        optimizer = torch.optim.Adam(net.parameters())

    if args.resume and args.init_model_pass != '-1':
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        f_path_latest = os.path.join(args.model_dir, 'latest')
        f_path = os.path.join(args.model_dir,
                              ('checkpoint-%s' % args.init_model_pass))
        if not os.path.isdir(args.model_dir):
            print('train from scratch: no checkpoint directory or file found')
        elif args.init_model_pass == 'latest' and os.path.isfile(f_path_latest):
            checkpoint = torch.load(f_path_latest)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch'] + 1
            print('resuming from epoch %s in latest' % start_epoch)
        elif os.path.isfile(f_path):
            checkpoint = torch.load(f_path)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch'] + 1
            print('resuming from epoch %s' % (start_epoch - 1))
        elif not os.path.isfile(f_path) or not os.path.isfile(f_path_latest):
            print('train from scratch: no checkpoint directory or file found')
    
    soft_xent_loss = softCrossEntropy()       

    # adversarial training
    a1 = []
    b1 = []
    c1 = []
    for epoch in range(start_epoch, args.max_epoch):
        #pdb.set_tirace()
        #if epoch < 0:
            #net = Attack_FeaScatter1(basic_net, config_feature_scatter)
        #else:
        net = Attack_FeaScatter(basic_net, config_feature_scatter)
        a2, b2, c2 = train_fun(epoch, net, args, optimizer, train_generator, device)
        a1.append(a2)
        b1.append(b2)
        c1.append(c2)
    np.save('training_acc_nat1.npy', a1)
    np.save('training_acc_adv1.npy', b1)
    np.save('adv_loss1.npy', c1)

def parse_args():
    parser = ArgumentParser("Feature Scatterring Training")
    parser.add_argument('--resume', default='True')
    parser.add_argument('--init_model_pass', default='latest', type=str, help='init model pass (-1: from scratch; K: checkpoint-K)')
    parser.add_argument("-m", "--model_ckpt", type=str, default=None, help="model checkpoint")
    parser.add_argument("-mt", "--model_type", type=str, default='cnn', help="model type: cnn, wideresnet, tdnn, resnet, etc.")
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument("-e", "--epsilon", type=float, default=0.002, help="perturbation scale")

    parser.add_argument('--model_dir', default='/data/monisankha/darpa_gard/gard-adversarial-audio/model_moni_hybrid/', type=str, help='model path')
    parser.add_argument('--max_epoch', default=200, type=int, help='max number of epochs')
    parser.add_argument('--save_epochs', default=20, type=int, help='save period')
    parser.add_argument('--decay_epoch1', default=60, type=int, help='learning rate decay epoch one')
    parser.add_argument('--decay_epoch2', default=90, type=int, help='learning rate decay point two')
    parser.add_argument('--decay_rate', default=0.1, type=float, help='learning rate decay rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (1-tf.momentum)')
    parser.add_argument('--weight_decay', default=2e-4, type=float, help='weight decay')
    parser.add_argument('--log_step', default=10, type=int, help='log_step')
    
    parser.add_argument("-l", "--wav_length", type=int, default=80_000, help="max length of waveform in a batch")
    #parser.add_argument("-epochs", "--num_epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("-opt", "--optimizer", type=str, default='sgd', help="optimizer: sgd, adam, etc.")
    parser.add_argument("-aug", "--augment", type=int, default=0, help="Gaussian noise augmentation to normal samples")
    parser.add_argument("--ratio", type=float, default=0.5, help="proportion of adversarial samples, 1=train only on adversarial samples")
    #parser.add_argument('-a', '--attack', type=str, default="feature_scatter")
    parser.add_argument('--adv_mode', default='feature_scatter', type=str, help='adv_mode (feature_scatter)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
