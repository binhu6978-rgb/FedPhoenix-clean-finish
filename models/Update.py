#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
# import test
import sys
import io   
import copy
from models.test import test_img
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset,WeightedRandomSampler

import torch.nn.functional as F
import numpy as np
import random
from optimizer.Adabelief import AdaBelief



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.labels = [self.dataset.targets[i] for i in idxs]  # 新增labels属性

    def __len__(self): 
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate_FedAvg(object):
    def __init__(self, args, dataset_test, dataset=None, idxs=None, verbose=False, freeze_conv=False, freeze_fc=False,stopped_count=0):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.verbose = verbose
        self.mean_loss = 0.0
        self.freeze_conv = freeze_conv  # Freeze convolutional layers
        self.freeze_fc = freeze_fc      # Freeze fully connected layers
        self.stopped_count=stopped_count

        loader_args = {
            'batch_size': self.args.local_bs,
            'num_workers': args.num_workers,
            'shuffle': True  
        }
        if args.num_workers>0:   
            loader_args.update({
                'pin_memory':True,
                'persistent_workers': True
            })
        
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs),
            **loader_args
        )    
    def train(self, net):
        net.train()
        # Define optimizer
        if self.args.optimizer == 'sgd':
            # SGD optimizer
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, net.parameters()),    lr=self.args.lr, momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            # Adam optimizer
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, net.parameters()),
                lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            # AdaBelief optimizer
            optimizer = AdaBelief(
                filter(lambda p: p.requires_grad, net.parameters()),
                lr=self.args.lr)
            
        Predict_loss = 0

        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)['output']
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                Predict_loss += loss.item() 
        self.mean_loss=Predict_loss / (self.args.local_ep * len(self.ldr_train))
        if self.verbose:
            
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        return net.state_dict()
    
    def get_mean_loss(self):
        return self.mean_loss


def test(model, dataset_test, args):
    
    # testing
    acc_test, loss_test = test_img(model, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()

class LocalUpdate_ClientSampling(object):
    def __init__(self, args, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(
        DatasetSplit(dataset, idxs), 
        batch_size=self.args.local_bs, 
        shuffle=True,
        num_workers=args.num_workers,  
        # pin_memory=True,
      
        )
        self.verbose = verbose

    def train(self, net):

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)['output']
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                Predict_loss += loss.item()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        return net

class LocalUpdate_FedProx(object):
    def __init__(self, args, glob_model, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")
        self.selected_clients = []
        self.ldr_train = DataLoader(
        DatasetSplit(dataset, idxs), 
        batch_size=self.args.local_bs, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True
      
        )
        self.glob_model = glob_model
        self.prox_alpha = args.prox_alpha
        self.verbose = verbose

    def train(self, net):

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        Penalize_loss = 0

        global_weight_collector = list(self.glob_model.parameters())

        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)['output']
                predictive_loss = self.loss_func(log_probs, labels)

                # for fedprox
                fed_prox_reg = 0.0
                # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += ((self.prox_alpha / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)

                loss = predictive_loss + fed_prox_reg
                Predict_loss += predictive_loss.item()
                Penalize_loss += fed_prox_reg.item()

                loss.backward()
                optimizer.step()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            info += ', Penalize loss={:.4f}'.format(Penalize_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        return net.state_dict()

class LocalUpdate_FedGen(object):
    def __init__(self, args, generative_model, dataset=None, idxs=None, verbose=False, regularization=True):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.KLDivLoss(reduction='batchmean')
        self.crossentropy_loss = nn.CrossEntropyLoss(reduce=False)
        self.selected_clients = []
        self.ldr_train = DataLoader(
        DatasetSplit(dataset, idxs), 
        batch_size=self.args.local_bs, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True,
      
        )
        self.verbose = verbose
        self.generative_model = generative_model
        self.regularization = regularization
        self.generative_alpha = args.generative_alpha
        self.generative_beta = args.generative_beta
        self.latent_layer_idx = -1

    def train(self, net):

        net.train()
        self.generative_model.eval()

        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        Teacher_loss = 0
        Latent_loss = 0
        for iter in range(self.args.local_ep):

            for batch_idx, (images, y) in enumerate(self.ldr_train):
                images, y = images.to(self.args.device), y.to(self.args.device)
                net.zero_grad()
                user_output_logp = net(images)['output']
                predictive_loss = self.loss_func(user_output_logp, y)

                #### sample y and generate z
                if self.regularization:
                    ### get generator output(latent representation) of the same label
                    gen_output = self.generative_model(y, latent_layer_idx=self.latent_layer_idx)['output'].to(
                        self.args.device)
                    logit_given_gen = net(gen_output, start_layer_idx=self.latent_layer_idx)['output']
                    target_p = F.softmax(logit_given_gen, dim=1).clone().detach()
                    user_latent_loss = self.generative_beta * self.ensemble_loss(F.log_softmax(user_output_logp, dim=1),
                                                                            target_p)

                    sampled_y = np.random.choice(self.args.num_classes, self.args.bs)
                    sampled_y = torch.LongTensor(sampled_y).to(self.args.device)
                    gen_result = self.generative_model(sampled_y, latent_layer_idx=self.latent_layer_idx)
                    gen_output = gen_result['output'].to(
                        self.args.device)  # latent representation when latent = True, x otherwise
                    user_output_logp = net(gen_output, start_layer_idx=self.latent_layer_idx)['output']
                    teacher_loss = self.generative_alpha * torch.mean(
                        self.crossentropy_loss(user_output_logp, sampled_y)
                    )
                    # this is to further balance oversampled down-sampled synthetic data
                    gen_ratio = self.args.bs / self.args.bs
                    loss = predictive_loss + gen_ratio * teacher_loss + user_latent_loss
                    Teacher_loss += teacher_loss.item()
                    Latent_loss += user_latent_loss.item()
                else:
                    #### get loss and perform optimization
                    loss = predictive_loss

                loss.backward()
                optimizer.step()

                Predict_loss += loss.item()

        # if True:
        #     info = 'User predict Loss={:.4f} Teacher Loss={:.4f} Latent Loss={:.4f}'.format(
        #         Predict_loss / (self.args.local_ep * len(self.ldr_train)),
        #         Teacher_loss / (self.args.local_ep * len(self.ldr_train)),
        #         Latent_loss / (self.args.local_ep * len(self.ldr_train)))
        #     print(info)

        net.to('cpu')
        
        return net
    
