#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchvision import datasets, transforms
from utils.sampling import *
from utils.dataset_utils import separate_data,read_record


from utils import mydata
from torch.autograd import Variable
import torch.nn.functional as F
import os
import json
import torch
from torch.utils.data import WeightedRandomSampler
import pandas as pd
from PIL import Image
import torch.utils.data as data
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader


def get_tiny_imagenet_data(args):

    trans_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    trans_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


    dataset_train = TinyImageNet(root='./data/tiny-imagenet-200', 
                                train=True, 
                                transform=trans_train)
    dataset_test = TinyImageNet(root='./data/tiny-imagenet-200', 
                               train=False, 
                               transform=trans_val)

    
    if args.generate_data:
        if args.iid:
         
            dict_users = tiny_imagenet_iid(dataset_train, args.num_users)
        else:
         
            dict_users = separate_data(dataset_train,
                                     args.num_users,
                                     200,  
                                     args.data_beta)
    else:
        dict_users = read_record(args.record_file)
        
    return dataset_train, dataset_test, dict_users

def tiny_imagenet_iid(dataset, num_users):

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def tiny_imagenet_noniid(dataset, num_users, noniid_case):

    num_shards = num_users * 2
    num_imgs = len(dataset) // num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    

    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]


    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


def get_dataset(args):

    file = os.path.join("data", args.dataset + "_" + str(args.num_users))
    if args.iid:
        file += "_iid"
    else:
        file += "_noniidCase" + str(args.noniid_case)

    if args.noniid_case > 4:
        file += "_beta" + str(args.data_beta)

    file += ".json"
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        if args.generate_data:
            # sample users
            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'cifar10':

        trans_cifar10_train = transforms.Compose([
            # transforms.RandomHorizontalFlip(),  
            # transforms.RandomCrop(32, padding=4),  
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]
                                                  
                                                  
                                                  )
        trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        # Support both the torchvision-standard ``./data/cifar-10-batches-py``
        # layout and the legacy ``./data/cifar10`` root used by older runs.
        legacy_root = './data/cifar10'
        standard_root = './data'
        if os.path.isdir(os.path.join(legacy_root, 'cifar-10-batches-py')):
            cifar_root = legacy_root
        else:
            cifar_root = standard_root
        dataset_train = datasets.CIFAR10(cifar_root, train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10(cifar_root, train=False, download=True, transform=trans_cifar10_val)

        if args.generate_data:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            elif args.noniid_case < 5:
                dict_users = cifar_noniid(dataset_train,args.num_users,args.noniid_case)
            else:
                dict_users = separate_data(dataset_train,args.num_users,args.num_classes,args.data_beta)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'cifar100':
        trans_cifar100 = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset_train = mydata.CIFAR100_coarse('./data/cifar100_coarse', train=True, download=True,
                                               transform=trans_cifar100)
        dataset_test = mydata.CIFAR100_coarse('./data/cifar100_coarse', train=False, download=True,
                                              transform=trans_cifar100)
        if args.generate_data:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            elif args.noniid_case < 5:
                dict_users = cifar_noniid(dataset_train, args.num_users, args.noniid_case)
            else:
                dict_users = separate_data(dataset_train, args.num_users, args.num_classes, args.data_beta)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'fashion-mnist':
        trans = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist/', train=True, download=True, transform=trans)
        dataset_test = datasets.FashionMNIST('./data/fashion-mnist/', train=False, download=True, transform=trans)
        if args.generate_data:
            if args.iid:
                dict_users = fashion_mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = fashion_mnist_noniid(dataset_train, args.num_users, case=args.noniid_case)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'femnist':
        dataset_train = FEMNIST(True)
        dataset_test = FEMNIST(False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
    else:
        exit('Error: unrecognized dataset')

    if args.generate_data: 
        with open(file,'w') as f:
            dataJson = {"dataset":args.dataset,"num_users":args.num_users,"iid":args.iid,"noniid_case":args.noniid_case,"data_beta":args.data_beta,"train_data":dict_users}
            json.dump(dataJson,f)

    


    return dataset_train, dataset_test, dict_users
def compute_dataset_stats(dataset):
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    num_samples = len(dataset)
    
    for img, _ in DataLoader(dataset, batch_size=64, num_workers=4):
        channel_sum += img.mean(dim=[0,2,3])
        channel_sq_sum += (img**2).mean(dim=[0,2,3])
    
    mean = channel_sum / num_samples
    std = (channel_sq_sum / num_samples - mean**2).sqrt()
    
    return mean.tolist(), std.tolist()


class TinyImageNet(data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        
   
        wnids_path = os.path.join(root, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            self.wnids = [x.strip() for x in f]
        
        self.wnid_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        
        
        self.samples = []
        self.targets = [] 
        
        if train:
        
            for idx, wnid in enumerate(self.wnids):
                img_dir = os.path.join(root, 'train', wnid, 'images')
                for img_name in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, img_name)
                    self.samples.append((img_path, idx))
                    self.targets.append(idx)  
        else:
            val_anno_path = os.path.join(root, 'val', 'val_annotations.txt')
            with open(val_anno_path, 'r') as f:
                for line in f:
                    img_name, wnid, _ = line.split('\t')[:3]
                    img_path = os.path.join(root, 'val', 'images', img_name)
                    idx = self.wnid_to_idx[wnid]
                    self.samples.append((img_path, idx))
                    self.targets.append(idx)  
                    
  
        self.targets = np.array(self.targets)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label

    def __len__(self):
        return len(self.samples)
