import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#from models import *
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import utils
import math

from utils import softCrossEntropy, CWLoss
from utils import softCrossEntropy
from utils import one_hot_tensor, label_smoothing
import ot
import pickle
import pdb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Attack_FeaScatter(nn.Module):
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_FeaScatter, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']
        self.ls_factor = 0.1 if 'ls_factor' not in config.keys(
        ) else config['ls_factor']

        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        a = inputs.shape
        b = targets.shape
        if not attack:
            outputs = self.basic_net(inputs)
            return outputs, None
        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))

        aux_net.eval()
        #pdb.set_trace()
        batch_size = inputs.size(0)
        m = batch_size
        n = batch_size

        logits = aux_net(inputs)[0]
        a = logits.shape
        num_classes = logits.size(0)

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=0)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()

        x_org = x.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pred_nat = aux_net(inputs)
        a = logits_pred_nat.shape

        num_classes = logits_pred_nat.size(1)
        y_gt = one_hot_tensor(targets, num_classes, device)

        loss_ce = softCrossEntropy()
        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        loss_func1 = CWLoss(251) 

        iter_num = self.num_steps

        for i in range(iter_num):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)

            logits_pred = aux_net(x)

            ot_loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, logits_pred_nat,
                                                  logits_pred, None, None,
                                                  0.01, m, n)
            
            loss = loss_func(logits_pred, y_tensor_adv)
            loss = loss.mean()
            #pdb.set_trace() 
            loss1 = loss_func1(logits_pred, y_tensor_adv)
            loss1 = loss1.mean()
            #pdb.set_trace() 
            loss2 = ot_loss + loss + loss1
            aux_net.zero_grad()
            adv_loss = loss2
            adv_loss.backward(retain_graph=True)
            x_adv = x.data + self.step_size * torch.sign(x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

        logits_pred = self.basic_net(x)
        self.basic_net.zero_grad()
        #y_sm = utils.label_smoothing(y_gt, y_gt.size(1), self.ls_factor)
        #adv_loss = loss_ce(logits_pred, y_sm.detach())
        #adv_loss = loss_func(logits_pred, y_tensor_adv)
        adv_loss1 = loss_func(logits_pred_nat, y_tensor_adv)
        adv_loss2 = loss_func(logits_pred, y_tensor_adv)
        adv_loss = adv_loss1 + adv_loss2
        adv_loss = adv_loss.mean()
        return logits_pred, adv_loss, x

