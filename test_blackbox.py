from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import utils
import collections

from dev.loaders import LibriSpeech4SpeakerRecognition
from dev.models import RawAudioCNN

from feature_scatter_attack import Attack_FeaScatter

from utils import softCrossEntropy
from utils import one_hot_tensor, label_smoothing

import os
import argparse
import sys
import datetime

from tqdm import tqdm
#from models import *
from hparams import hp
import utils
import pdb

from attack_methods import Attack_None, Attack_PGD

from utils import softCrossEntropy, CWLoss

parser = argparse.ArgumentParser(description='Feature Scattering Adversarial Training')

parser.register('type', 'bool', utils.str2bool)
parser.add_argument('--attack', default=True, type='bool', help='attack')

parser.add_argument('--attack_method', default='pgd', type=str, help='adv_mode (natural, pdg or cw)')
parser.add_argument('--attack_method_list', default='natural-fgsm-pgd-cw-fea_sca', type=str)
parser.add_argument('--log_step', default='200', type=int, help='log_step')
#parser.add_argument("-m", "--model_ckpt", required=True)
parser.add_argument('--model_dir', default='/scratch/mp_323/gard-adversarial-audio/model_standard_moni/', type=str, help='model path')
parser.add_argument('--model_dir1', default='/scratch/mp_323/gard-adversarial-audio/model_pgd_moni1/', type=str, help='model path')
parser.add_argument("-mt", "--model_type", type=str, default='cnn', help="model type: cnn, tdnn, resnet, etc.")
parser.add_argument('--init_model_pass', default='latest', type=str, help='init model pass')
parser.add_argument('--resume', default='True')
parser.add_argument("-e", "--epsilon", type=float, default=0.002)

args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0
return_file_name=False

# Data
print('==> Preparing data..')

dataset = LibriSpeech4SpeakerRecognition(
        root=hp.data_root,
        url=hp.data_subset,
        subset="test",
        train_speaker_ratio=hp.train_speaker_ratio,
        train_utterance_ratio=hp.train_utterance_ratio,
        project_fs=hp.sr,  # FIXME: unused
        wav_length=None,
    )
testloader = DataLoader(dataset, batch_size=1, shuffle=False)
#pdb.set_trace()
print('==> Building model..')

if args.model_type == 'cnn':
    basic_net = RawAudioCNN(num_class=251)
    attack_net = RawAudioCNN(num_class=251)
else:
    logging.error('Please provide a valid model architecture type!')
    sys.exit(-1)

basic_net = basic_net.to(device)
attack_net = attack_net.to(device)

# configs
config_natural = {'train': False}

config_fgsm = {
    'train': False,
    'targeted': False,
    'epsilon': args.epsilon,
    'num_steps': 1,
    'step_size': args.epsilon,
    'random_start': True
}

config_pgd = {
    'train': False,
    'targeted': False,
    'epsilon': args.epsilon,
    'num_steps': 100,
    'step_size': args.epsilon / 5,
    'random_start': True,
    'loss_func': torch.nn.CrossEntropyLoss(reduction='none'),
    'box_type': 'black'
}

config_cw = {
    'train': False,
    'targeted': False,
    'epsilon': args.epsilon,
    'num_steps': 100,
    'step_size': args.epsilon / 5,
    'random_start': True,
    'loss_func': CWLoss(251),
    'box_type': 'black',
}

# config for feature scatter
config_feature_scatter = {
    'train': False,
   #'epsilon': 8.0 / 255 * 2,
    'epsilon': args.epsilon,
    'num_steps': 10,
   #'step_size': 8.0 / 255 * 2,
    'step_size': args.epsilon / 5,
    'random_start': True,
    'ls_factor': 0.7,
}

def test(epoch, net, criterion, flag):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    #iterator = tqdm(testloader, ncols=0, leave=False)
    #for i, (waveform, label) in enumerate(loader, 1):
    #for batch_idx, (inputs, targets) in enumerate(iterator):
    for i, data in enumerate(testloader, 1):
        if return_file_name:
           inputs, targets, filename = data
           filename = filename[0]
        else:
           inputs, targets = data

        start_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        #pdb.set_trace()

        pert_inputs = inputs.detach()
        #pdb.set_trace()
        outputs, _, _ = net(pert_inputs, targets)
        #pdb.set_trace()
        #num_classes = targets.size(0)
        #print(num_classes)
        
        #loss = criterion(outputs, targets)
        if flag == 1:
           targets1 = one_hot_tensor(targets, 251, device)
           loss = criterion(outputs, targets1.detach())
        else:
           loss = criterion(outputs, targets)
        
        test_loss += loss.item()

        duration = time.time() - start_time

        _, predicted = outputs.max(1)
        #pdb.set_trace()
        batch_size = targets.size(0)
        total += batch_size
        correct_num = predicted.eq(targets).sum().item()
        correct += correct_num
        #iterator.set_description(
            #str(predicted.eq(targets).sum().item() / targets.size(0)))

        if i % args.log_step == 0:
            print(
                "step %d, duration %.2f, test  acc %.2f, avg-acc %.2f, loss %.2f"
                % (i, duration, 100. * correct_num / batch_size,
                   100. * correct / total, test_loss / total))

    acc = 100. * correct / total
    print('Val acc:', acc)
    return acc


attack_list = args.attack_method_list.split('-')
attack_num = len(attack_list)

for attack_idx in range(2, 4):

    args.attack_method = attack_list[attack_idx]

    if args.attack_method == 'natural':
        print('-----natural non-adv mode -----')
        # config is only dummy, not actually used
        net = Attack_None(basic_net, config_natural)
    elif args.attack_method.upper() == 'FGSM':
        print('-----FGSM adv mode -----')
        net = Attack_PGD(basic_net, config_fgsm)
    elif args.attack_method.upper() == 'PGD':
        print('-----PGD adv mode -----')
        net = Attack_PGD(basic_net, config_pgd, attack_net)
    elif args.attack_method.upper() == 'CW':
        print('-----CW adv mode -----')
        net = Attack_PGD(basic_net, config_cw, attack_net)
    elif args.attack_method == 'fea_sca':
        print('-----Feature Scatter adv mode -----')
        net = Attack_FeaScatter(basic_net, config_feature_scatter)
    else:
        raise Exception(
            'Should be a valid attack method. The specified attack method is: {}'
            .format(args.attack_method))

    #if device == 'cuda':
     #   net = torch.nn.DataParallel(net)
      #  cudnn.benchmark = True
    #pdb.set_trace()
    if args.resume and args.init_model_pass != '-1':
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        f_path_latest = os.path.join(args.model_dir1, 'latest')
        f_path = os.path.join(args.model_dir1,
                              ('checkpoint-%s' % args.init_model_pass))
        if not os.path.isdir(args.model_dir1):
            print('train from scratch: no checkpoint directory or file found')
        elif args.init_model_pass == 'latest' and os.path.isfile(
                f_path_latest):
            checkpoint = torch.load(f_path_latest)
            state_dict = checkpoint['net']
            state_dict_new = collections.defaultdict(list)
            keys = list(state_dict.keys())
            for key in keys:
                key_new = ".".join(key.split('.')[1:])
                state_dict_new[key_new] = state_dict[key]
            #pdb.set_trace()
            net.basic_net.load_state_dict(state_dict_new)
            #net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            print('resuming from epoch %s in latest' % start_epoch)
        elif os.path.isfile(f_path):
            checkpoint = torch.load(f_path)
            net.load_state_dict(checkpoint['net'])
            #start_epoch = checkpoint['epoch']
            #print('resuming from epoch %s' % start_epoch)
        elif not os.path.isfile(f_path) or not os.path.isfile(f_path_latest):
            print('train from scratch: no checkpoint directory or file found')

    x = 7
    #pdb.set_trace()
    y = 7
    if args.resume and args.init_model_pass != '-1':
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        f_path_latest = os.path.join(args.model_dir, 'latest')
        f_path = os.path.join(args.model_dir,
                              ('checkpoint-%s' % args.init_model_pass))
        if not os.path.isdir(args.model_dir):
            print('train from scratch: no checkpoint directory or file found')
        elif args.init_model_pass == 'latest' and os.path.isfile(
                f_path_latest):
            checkpoint = torch.load(f_path_latest)
            state_dict = checkpoint['net']
            state_dict_new = collections.defaultdict(list)
            keys = list(state_dict.keys())
            for key in keys:
                key_new = ".".join(key.split('.')[1:])
                state_dict_new[key_new] = state_dict[key]
            #pdb.set_trace()
            net.attack_net.load_state_dict(state_dict_new)
            #net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            print('resuming from epoch %s in latest' % start_epoch)
        elif os.path.isfile(f_path):
            checkpoint = torch.load(f_path)
            net.load_state_dict(checkpoint['net'])
            #start_epoch = checkpoint['epoch']
            #print('resuming from epoch %s' % start_epoch)
        elif not os.path.isfile(f_path) or not os.path.isfile(f_path_latest):
            print('train from scratch: no checkpoint directory or file found')

    #state_dict = checkpoint['net']
    #state_dict_new = defaultdict(list)
    #pdb.set_trace()
    #keys = list(state_dict.keys())
    #for key in keys:
     #   key_new = ".".join(key.split('.')[1:])
     #   state_dict_new[key_new] = state_dict[key]
    #del state_dict_new[key]
    #model.load_state_dict(state_dict_new)
    #model.eval()
    #pdb.set_trace()
    criterion = nn.CrossEntropyLoss()
    flag = 0
    if args.attack_method == 'fea_sca':
       criterion = softCrossEntropy()
       flag = 1

    test(0, net, criterion, flag)
