#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.utils import *
import copy
import matplotlib
matplotlib.use('Agg')
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap



def zero_out_next_layer_weights(model, reset_percent):
    def get_conv_linear_layers(module, layers):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append(module)
        elif hasattr(module, 'children') and len(list(module.children())) > 0:
            for child in module.children():
                get_conv_linear_layers(child, layers)
        return layers

    layers = get_conv_linear_layers(model, [])
    #print(f"Total Conv/Linear layers: {len(layers)}")

    for i in range(len(layers) - 1):
        current_layer = layers[i]
        next_layer = layers[i + 1]
        
        if isinstance(current_layer, nn.Linear):
            num_neurons = current_layer.out_features
        elif isinstance(current_layer, nn.Conv2d):
            num_neurons = current_layer.out_channels

        num_to_zero_out = max(1, int(np.ceil(reset_percent* num_neurons)))
        indices_to_zero_out = np.random.choice(num_neurons, num_to_zero_out, replace=False)

        # print(f"Layer {i} zeros out {num_to_zero_out} neurons: {indices_to_zero_out}")

        if isinstance(next_layer, nn.Linear):
            if max(indices_to_zero_out) < next_layer.weight.data.shape[1]:
                next_layer.weight.data[:, indices_to_zero_out] = 0.0
            # else:
            #     print(f"Skipping layer {i+1} due to index out of bounds")
        elif isinstance(next_layer, nn.Conv2d):
            if max(indices_to_zero_out) < next_layer.weight.data.shape[1]:
                next_layer.weight.data[:, indices_to_zero_out, :, :] = 0.0


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0


    loader_args = {
        'batch_size': args.bs,
        'num_workers': args.num_workers,
        'shuffle': True
    }

 
    if args.num_workers > 0:
        loader_args.update(
                {
                # 'shuffle': True,
                'pin_memory':True,
                'persistent_workers': True  
            })

        data_loader = DataLoader(
            datatest,
            **loader_args
        )
    data_loader = DataLoader(
        datatest, 
        batch_size=args.bs
        )
    l = len(data_loader)
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = net_g(data)['output']
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def evaluate_round_accuracy(net_glob, dataset_test, args, round_number):
    """Evaluate and print one machine-readable accuracy line."""
    accuracy, _ = test_img(net_glob, dataset_test, args)
    accuracy_value = float(
        accuracy.item() if hasattr(accuracy, "item") else accuracy
    )
    print(
        f"ROUND_ACCURACY method={args.algorithm} round={int(round_number)} "
        f"accuracy={accuracy_value:.6f}",
        flush=True,
    )
    return accuracy_value


def print_peak_accuracy(accuracies, method):
    """Print the maximum round accuracy and its one-based round number."""
    if not accuracies:
        raise ValueError("No round accuracies were recorded")
    peak_index = int(np.argmax(np.asarray(accuracies, dtype=np.float64)))
    peak_accuracy = float(accuracies[peak_index])
    peak_round = peak_index + 1
    print(
        f"PEAK_ACCURACY method={method} round={peak_round} "
        f"accuracy={peak_accuracy:.6f}",
        flush=True,
    )
    return peak_accuracy, peak_round




def calculate_max_average(acc, window_size=10):
    if len(acc) < window_size:
        raise ValueError("输入的准确度列表长度必须大于或等于窗口大小。")
    
    # 计算每个窗口的平均值
    averages = [sum(acc[i:i + window_size]) / window_size for i in range(len(acc) - window_size + 1)]
    
    # 求出最大平均值及其索引
    max_average = max(averages)
    max_average_index = averages.index(max_average)
    
    print("每连续十轮准确度的平均值的最大值为:", max_average)
    print(f"这个最大平均值出现在第 {max_average_index + 1} 轮到第 {max_average_index + window_size} 轮")

    return max_average, (max_average_index + 1, max_average_index + window_size)



def test(net_glob, dataset_test, args):
    
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()









from torch.utils.data import DataLoader, Subset
def get_features(net, dataset, args):
    """Extract features"""
    net.eval()
    features = []
    labels = []
    if len(dataset) > 12000:
        # Calculate 20% of the dataset size
        subset_size = int(0.2 * len(dataset))
        
        # Create a subset by randomly sampling 20% of the data
        indices = torch.randperm(len(dataset))[:subset_size]  # Randomly shuffle and select the first 20%
        dataset = Subset(dataset, indices)

    data_loader = DataLoader(
        dataset, 
        batch_size=500, 
        shuffle=False,
        num_workers=12,  # Increase the number of worker processes
        pin_memory=True  # Load data directly into pinned memory
    )
    
    with torch.no_grad():
        for batch_idx, (images, y) in enumerate(data_loader):
            images = images.to(args.device)
            feat = net(images)['representation'].cpu().numpy()
            features.append(feat)
            labels.append(y.numpy())
            
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def visualize_decision_boundary(features, labels, net, args, epoch):
    """Visualize decision boundary"""
    # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=args.n_components, random_state=args.seed, perplexity=args.perplexity)
    features_2d = tsne.fit_transform(features)
    
    # Create grid
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    # Plot decision boundary
    plt.figure(figsize=(8,7))
    
    # Plot scatter plot
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], s=10,
                         c=labels, cmap='tab10', alpha=0.6)
    
    plt.colorbar(scatter)
    # plt.title(f'Feature Distribution')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    
  

    # Create folder if it doesn't exist
    os.makedirs(f'./vis/{args.algorithm}', exist_ok=True)
    save_path = f'./vis/{args.algorithm}/tsne.pdf'
    plt.savefig(save_path)
    print('Decision boundary has been saved')
    plt.close()
