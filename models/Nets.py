#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(800, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        result = {}
        result['activation'] = x
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        result['hint'] = x
        x = F.relu(self.fc1(x))
        result['representation'] = x
        x = self.fc2(x)
        result['output'] = x
        return result



    
class CNNCifarDrop(nn.Module):
    def __init__(self, args, dropout_rate=0.3):
        super(CNNCifarDrop, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate)  # Dropout for conv layer
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dropout2 = nn.Dropout2d(dropout_rate)  # Dropout for conv layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout3 = nn.Dropout(0.3)  # Dropout after first fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.dropout4 = nn.Dropout(0.3)  # Dropout after second fully connected layer
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx < 0:
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
        
        x = self.conv1(x)
        x = self.dropout1(x)  # Applying dropout after first conv layer
        x = self.pool(F.relu(x))
        
        x = self.conv2(x)
        x = self.dropout2(x)  # Applying dropout after second conv layer
        x = self.pool(F.relu(x))
        
        result = {'activation' : x}
        
        x = x.view(-1, 16 * 5 * 5)
        result['hint'] = x
        
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)  # Applying dropout after first fully connected layer
        x = F.relu(self.fc2(x))
        x = self.dropout4(x)  # Applying dropout after second fully connected layer
        result['representation'] = x
        
        x = self.fc3(x)
        result['output'] = x
        return result

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        z = self.fc3(z)

        result = {'output': z}
        if logit:
            result['logit'] = z
        return result


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx < 0:  #
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        result = {'activation' : x}
        x = x.view(-1, 16 * 5 * 5)
        result['hint'] = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        result['representation'] = x
        x = self.fc3(x)
        result['output'] = x
        return result

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        z = self.fc3(z)

        result = {'output': z}
        if logit:
            result['logit'] = z
        return result


class CNNFashionMnist(nn.Module):
    def __init__(self, args):
        super(CNNFashionMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        results = {'representation' : x}
        x = self.fc2(x)
        results['output'] = x
        return results




import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
from models.resnetcifar import ResNet18_cifar10, ResNet50_cifar10,ResNet18_cifar10_drop

#import pytorch_lightning as pl


class MLP_header(nn.Module):
    def __init__(self,):
        super(MLP_header, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        #projection
        # self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


class FcNet(nn.Module):
    """
    Fully connected network for MNIST classification
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_p=0.0):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_p = dropout_p

        self.dims = [self.input_dim]
        self.dims.extend(hidden_dims)
        self.dims.append(self.output_dim)

        self.layers = nn.ModuleList([])

        for i in range(len(self.dims) - 1):
            ip_dim = self.dims[i]
            op_dim = self.dims[i + 1]
            self.layers.append(
                nn.Linear(ip_dim, op_dim, bias=True)
            )

        self.__init_net_weights__()

    def __init_net_weights__(self):

        for m in self.layers:
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0.1)

    def forward(self, x):

        x = x.view(-1, self.input_dim)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Do not apply ReLU on the final layer
            if i < (len(self.layers) - 1):
                x = nn.ReLU(x)

            # if i < (len(self.layers) - 1):  # No dropout on output layer
            #     x = F.dropout(x, p=self.dropout_p, training=self.training)

        return x


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        return x


class FCBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(FCBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGGConvBlocks(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features, num_classes=10):
        super(VGGConvBlocks, self).__init__()
        self.features = features
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class FCBlockVGG(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(FCBlockVGG, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = F.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNN_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        #self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        #out = self.conv1(x)
        #out = self.relu(out)
        #out = self.pool(out)
        #out = self.conv2(out)
        #out = self.relu(out)
        #out = self.pool(out)
        #out = out.view(-1, 16 * 5 * 5)

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# a simple perceptron model for generated 3D data
class PerceptronModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=2):
        super(PerceptronModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):

        x = self.fc1(x)
        return x


class SimpleCNNMNIST_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST_header, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        #self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

class SimpleCNNMNIST(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return x, 0, y


class SimpleCNNContainer(nn.Module):
    def __init__(self, input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNContainer, self).__init__()
        '''
        A testing cnn container, which allows initializing a CNN with given dims

        num_filters (list) :: number of convolution filters
        hidden_dims (list) :: number of neurons in hidden layers

        Assumptions:
        i) we use only two conv layers and three hidden layers (including the output layer)
        ii) kernel size in the two conv layers are identical
        '''
        self.conv1 = nn.Conv2d(input_channel, num_filters[0], kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


############## LeNet for MNIST ###################
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.ceriation = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class LeNetContainer(nn.Module):
    def __init__(self, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(LeNetContainer, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size, 1)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size, 1)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = self.fc1(x)
        x = self.fc2(x)
        return x



### Moderate size of CNN for CIFAR-10 dataset
class ModerateCNN(nn.Module):
    def __init__(self, output_dim=10):
        super(ModerateCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            # nn.Linear(4096, 1024),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


### Moderate size of CNN for CIFAR-10 dataset
class ModerateCNNCeleba(nn.Module):
    def __init__(self):
        super(ModerateCNNCeleba, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            # nn.Linear(4096, 1024),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        # x = x.view(x.size(0), -1)
        x = x.view(-1, 4096)
        x = self.fc_layer(x)
        return x


class ModerateCNNMNIST(nn.Module):
    def __init__(self):
        super(ModerateCNNMNIST, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(2304, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


class ModerateCNNContainer(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(ModerateCNNContainer, self).__init__()

        ##
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=input_channels, out_channels=num_filters[0], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

    def forward_conv(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        return x


class ModelFedCon(nn.Module):

    def __init__(self, base_model, out_dim, n_classes, net_configs=None):
        super(ModelFedCon, self).__init__()

        if base_model == "resnet50":
            basemodel = ResNet50_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18":
            basemodel = ResNet18_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "mlp":
            self.features = MLP_header()
            num_ftrs = 512
        elif base_model == 'cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = nn.Linear(out_dim, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            #print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x, start_layer_idx = 0, logit = False):

        if start_layer_idx < 0:  #
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)

        h = self.features(x)
        h = h.squeeze()

        result = {}
        result['feature'] = h
        #print("h after:", h)
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        result['representation'] = x

        y = self.l3(x)

        result['logit'] = y
        result['output'] = y

        return result

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        x = self.l1(z)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)

        x = self.l3(x)

        result = {'output': x}
        if logit:
            result['logit'] = x
        return result


class ModelFedCon_noheader(nn.Module):

    def __init__(self, base_model, out_dim, n_classes, net_configs=None):
        super(ModelFedCon_noheader, self).__init__()

        if base_model == "resnet50":
            basemodel = models.resnet50(pretrained=False)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18":
            basemodel = models.resnet18(pretrained=False)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet50":
            basemodel = ResNet50_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "mlp":
            self.features = MLP_header()
            num_ftrs = 512
        elif base_model == 'cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84

        self.num_ftrs = num_ftrs

        #summary(self.features.to('cuda:0'), (3,32,32))
        #print("features:", self.features)
        # projection MLP
        # self.l1 = nn.Linear(num_ftrs, num_ftrs)
        # self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = nn.Linear(num_ftrs, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            #print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x, start_layer_idx=0):
        if start_layer_idx < 0:
            return self.mapping(x)

        result = {}

        h = self.features(x)
        #print("h before:", h)
        #print("h size:", h.size())
        h = h.view(-1, self.num_ftrs)
        #print("h after:", h)
        # x = self.l1(h)
        # x = F.relu(x)
        # x = self.l2(x)
        result['representation'] = h

        y = self.l3(h)

        result['output'] = y

        return result

    def mapping(self, x):
        y = self.l3(x)

        result = {'output' : y}

        return result


class VGG16_Drop(nn.Module):
    def __init__(self, args):
        super(VGG16_Drop, self).__init__()
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(args.num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),  # 保留池化后Dropout1/4

            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),  # 保留池化后Dropout2/4

            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),  # 保留池化后Dropout3/4

            # 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # 9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),  # 保留池化后Dropout4/4

            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AvgPool2d(kernel_size=1, stride=1),
        )
        self.classifier = nn.Sequential(
            # 14
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),  # 修改Dropout概率

            # 15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),  # 修改Dropout概率
        )
        self.fc = nn.Linear(4096, args.num_classes)

    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx < 0:
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.fc(out)
        result = {}
        result['logit'] = out
        result['output'] = out
        return result

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        z = self.fc(z)
        result = {'output': z}
        if logit:
            result['logit'] = z
        return result

class VGG16_timage(nn.Module):
    def __init__(self, args):
        super(VGG16_timage, self).__init__()
        # Keep adaptive pooling to 4x4 for TinyImageNet
        adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
            
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Fixed to 3 channels
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            adaptive_pool
        )
        
        # Modified classifier for 4x4 feature maps (512*16 = 8192 input features)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 16, 4096),  # 8192 -> 4096
            nn.ReLU(True),
            nn.Dropout(),
            
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
        
        # Final layer for 200 TinyImageNet classes
        self.fc = nn.Linear(4096, 200)  # Fixed to 200 classes

    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx < 0:
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
            
        out = self.features(x)
        out = out.view(out.size(0), -1)  # Flatten 512x4x4 to 8192
        out = self.classifier(out)
        out = self.fc(out)
        
        result = {}
        result['logit'] = out
        result['output'] = out
        return result

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        z = self.fc(z)
        
        result = {'output': z}
        if logit:
            result['logit'] = z
        return result     





class VGG16_timage_Drop(nn.Module):
    def __init__(self, args):
        super(VGG16_timage_Drop, self).__init__()
        # Keep adaptive pooling to 4x4 for TinyImageNet
        adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
            
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),  # 添加第1个Dropout

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),  # 添加第2个Dropout

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),  # 添加第3个Dropout

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),  # 添加第4个Dropout

            # Block 5 (保持无Dropout)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            adaptive_pool
        )
        
        # Modified classifier for 4x4 feature maps (512*16 = 8192 input features)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 16, 4096),  # 8192 -> 4096
            nn.ReLU(True),
            nn.Dropout(),
            
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
        
        # Final layer for 200 TinyImageNet classes
        self.fc = nn.Linear(4096, 200)  # Fixed to 200 classes

    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx < 0:
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
            
        out = self.features(x)
        out = out.view(out.size(0), -1)  # Flatten 512x4x4 to 8192
        out = self.classifier(out)
        out = self.fc(out)
        
        result = {}
        result['logit'] = out
        result['output'] = out
        return result

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        z = self.fc(z)
        
        result = {'output': z}
        if logit:
            result['logit'] = z
        return result     












class VGG16(nn.Module):
    def __init__(self, args):
        super(VGG16, self).__init__()
        if args.dataset == 'timage':
            adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        else:
            adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(args.num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            adaptive_pool
        )
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
        
        # 最后的分类层根据数据集类别数自动调整
        self.fc = nn.Linear(4096, args.num_classes)
        


    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx < 0:
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
            
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.fc(out)
        
        result = {}
        result['logit'] = out
        result['output'] = out
        return result

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        z = self.fc(z)
        
        result = {'output': z}
        if logit:
            result['logit'] = z
        return result
# class VGG16(nn.Module):
#     def __init__(self, args):
#         super(VGG16, self).__init__()
#         self.features = nn.Sequential(
#             # 1
#             nn.Conv2d(args.num_channels, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             # 2
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             # 3
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             # 4
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             # 5
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             # 6
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             # 7
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             # 8
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             # 9
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             # 10
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             # 11
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             # 12
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             # 13
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.AvgPool2d(kernel_size=1, stride=1),
#         )
#         self.classifier = nn.Sequential(
#             # 14
#             nn.Linear(512, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=0),
#             # 15
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=0),
#             # 16
#             # nn.Linear(4096, args.num_classes),
#         )
#         self.fc = nn.Linear(4096, args.num_classes)

#     def forward(self, x, start_layer_idx=0, logit=False):
#         if start_layer_idx < 0:  #
#             return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         #        print(out.shape)
#         out = self.classifier(out)
#         out = self.fc(out)
#         #        print(out.shape)
#         result = {}
#         result['logit'] = out
#         result['output'] = out
#         return result

#     def mapping(self, z_input, start_layer_idx=-1, logit=True):
#         z = z_input
#         z = self.fc(z)

#         result = {'output': z}
#         if logit:
#             result['logit'] = z
#         return result



class VGG16_mnist(nn.Module):
    def __init__(self, args):
        super(VGG16_mnist, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
                     nn.ReLU(inplace=True),
                     nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                     nn.ReLU(inplace=True),
                     nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),
                     nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                     nn.ReLU(inplace=True),
                     nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),
                     nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                     nn.ReLU(inplace=True),
                     nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                     nn.ReLU(inplace=True),
                     nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                     nn.AdaptiveAvgPool2d(output_size=(7, 7)),
                     Flatten(),
                     nn.Linear(in_features=25088, out_features=256, bias=True),
                     #nn.Linear(in_features=256, out_features=args.num_classes, bias=True)
                     )
        self.fc = nn.Linear(256, args.num_classes)

    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx < 0:  #
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        #        print(out.shape)
        out = self.fc(out)
        #        print(out.shape)
        result = {}
        result['logit'] = out
        result['output'] = out
        return result

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        z = self.fc(z)

        result = {'output': z}
        if logit:
            result['logit'] = z
        return result


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self,x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.reshape(-1,shape)

class ResNet_cifar(nn.Module):
    def __init__(self, args, block, num_blocks, num_classes=10):
        super(ResNet_cifar, self).__init__()
        self.inplanes = 16
        print('num_classes:', num_classes)
        # declare lambda array
        num_block = 0
        for i in num_blocks:
            num_block += i
        print('block number:', num_block)

        self.conv1 = nn.Conv2d(args.num_channels, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        result = {}
        x = self.layer1(x)
        result['activation1'] = x
        x = self.layer2(x)
        result['activation2'] = x
        x = self.layer3(x)
        result['activation3'] = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        result['representation'] = x
        x = self.fc(x)
        result['output'] = x
        return result
        # x = self.conv1(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # return x


class ServerContinualBackprop:
    def __init__(self, net_glob, replacement_rate=0.005, maturity_threshold=8, utility_decay=0.5):
        self.model = net_glob
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.utility_decay = utility_decay
        
        # 在服务器端维护效用和年龄追踪50 33 54
        self.utilities = {}
        self.ages = {}
        self.initialize_tracking()
        
    def initialize_tracking(self):
        """初始化效用和年龄追踪"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                out_channels = module.weight.size(0)
                self.utilities[name] = torch.zeros(out_channels)
                self.ages[name] = torch.zeros(out_channels)
    
    def update_utilities(self, activations):
        """更新效用值"""
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if name in activations:
                        act = activations[name]
                        if isinstance(module, nn.Conv2d):
                            channel_contrib = act.abs().mean(dim=[0,2,3])
                        else:
                            channel_contrib = act.abs().mean(dim=0)
                        
                        self.utilities[name] = (
                            self.utility_decay * self.utilities[name] + 
                            (1 - self.utility_decay) * channel_contrib
                        )
                        self.ages[name] += 1
    
    def reset_units(self,scale_factor=0.5):
        """执行单元重置"""
        # 构建层之间的连接关系字典
        layer_connections = {}
        prev_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if prev_layer is not None:
                    layer_connections[prev_layer] = name
                prev_layer = name
        
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d,nn.Linear)):
                    out_channels = module.weight.size(0)
                    mature_units = (self.ages[name] > self.maturity_threshold)
                    num_mature = mature_units.sum().item()
                    
                    num_reset = int(num_mature * self.replacement_rate)
                    if num_reset == 0:num_reset=1
                    # print(f'num_reset:{num_reset}')
                    
                    if num_reset > 0:
                        utilities_mature = self.utilities[name].clone()
                        utilities_mature[~mature_units] = float('inf')
                        _, indices = torch.topk(utilities_mature, 
                                            k=num_reset,
                                            largest=False)
                        print (f'卷积核编号:{indices}')
                        
                        for idx in indices:
                            # 获取当前层参数的统计信息
                            current_mean = module.weight.data[idx:idx+1].mean().item()
                            current_std = module.weight.data[idx:idx+1].std().item()
                            
                            # 1. 重新初始化输入权重
                            if isinstance(module, nn.Conv2d):
                                # 生成与原分布相似的新参数
                                new_weights = torch.randn_like(module.weight[idx:idx+1]) * current_std * scale_factor + current_mean
                                module.weight.data[idx:idx+1] = new_weights
                            else:
                                # 对其他类型的层使用相同的初始化方法
                                new_weights = torch.randn_like(module.weight[idx:idx+1]) * current_std * scale_factor + current_mean
                                module.weight.data[idx:idx+1] = new_weights
                            
                            if module.bias is not None:
                                module.bias.data[idx] = 0
                                
                            # 2. 将输出权重置为0
                            if name in layer_connections:
                                next_layer_name = layer_connections[name]
                                next_module = dict(self.model.named_modules())[next_layer_name]
                                
                                try:
                                    if isinstance(module, nn.Conv2d):
                                        if isinstance(next_module, nn.Conv2d):
                                            # Conv2d -> Conv2d
                                            if idx < next_module.weight.size(1):
                                                next_module.weight.data[:, idx] = 0
                                        elif isinstance(next_module, nn.Linear):
                                            # Conv2d -> Linear (ResNet with Global Average Pooling)
                                            # 在ResNet中,由于全局平均池化,
                                            # Conv2d的每个通道直接对应Linear层的一个输入
                                            if idx < next_module.weight.size(1):
                                                next_module.weight.data[:, idx] = 0
                                    else:  # Linear
                                        if isinstance(next_module, nn.Linear):
                                            # Linear -> Linear
                                            if idx < next_module.weight.size(1):
                                                next_module.weight.data[:, idx] = 0
                                except IndexError as e:
                                    print(f"Warning: Index error when resetting weights. Layer: {name}, "
                                        f"Index: {idx}, Next layer: {next_layer_name}")
                                    continue
                            
                            # 3. 重置效用值和年龄
                            # self.utilities[name][idx] = 0
                            self.ages[name][idx] = 0
    #     """执行单元重置"""
    #     # 构建层之间的连接关系字典
    #     layer_connections = {}
    #     prev_layer = None
    #     for name, module in self.model.named_modules():
    #         if isinstance(module, (nn.Conv2d, nn.Linear)):
    #             if prev_layer is not None:
    #                 layer_connections[prev_layer] = name
    #             prev_layer = name
        
    #     with torch.no_grad():
    #         for name, module in self.model.named_modules():
    #             if isinstance(module, (nn.Conv2d, nn.Linear)):
    #                 out_channels = module.weight.size(0)
    #                 mature_units = (self.ages[name] > self.maturity_threshold)
    #                 num_mature = mature_units.sum().item()
                    
    #                 num_reset = int(num_mature * self.replacement_rate)
    #                 print(f'num_reset:{num_reset}')
                    
    #                 if num_reset > 0:
    #                     utilities_mature = self.utilities[name].clone()
    #                     utilities_mature[~mature_units] = float('inf')
    #                     _, indices = torch.topk(utilities_mature, 
    #                                         k=num_reset,
    #                                         largest=False)
                        
    #                     for idx in indices:
    #                         # 1. 重新初始化输入权重
    #                         if isinstance(module, nn.Conv2d):
    #                             nn.init.kaiming_normal_(
    #                                 module.weight[idx:idx+1],
    #                                 mode='fan_out',
    #                                 nonlinearity='relu'
    #                             )
    #                         else:
    #                             nn.init.kaiming_normal_(
    #                                 module.weight[idx:idx+1],
    #                                 mode='fan_out',
    #                                 nonlinearity='relu'
    #                             )
                            
    #                         if module.bias is not None:
    #                             module.bias.data[idx] = 0
                                
    #                         # 2. 将输出权重置为0
    #                         if name in layer_connections:
    #                             next_layer_name = layer_connections[name]
    #                             next_module = dict(self.model.named_modules())[next_layer_name]
                                
    #                             try:
    #                                 if isinstance(module, nn.Conv2d):
    #                                     if isinstance(next_module, nn.Conv2d):
    #                                         # Conv2d -> Conv2d
    #                                         if idx < next_module.weight.size(1):
    #                                             next_module.weight.data[:, idx] = 0
    #                                     elif isinstance(next_module, nn.Linear):
    #                                         # Conv2d -> Linear
    #                                         # 计算特征图大小
    #                                         feat_h = (module.weight.size(2) + 
    #                                                 2 * module.padding[0] - 
    #                                                 module.kernel_size[0]) // module.stride[0] + 1
    #                                         feat_w = (module.weight.size(3) + 
    #                                                 2 * module.padding[1] - 
    #                                                 module.kernel_size[1]) // module.stride[1] + 1
    #                                         feat_size = feat_h * feat_w
                                            
    #                                         # 计算在展平后的索引范围
    #                                         start_idx = idx * feat_size
    #                                         end_idx = (idx + 1) * feat_size
                                            
    #                                         # 确保索引在有效范围内
    #                                         if start_idx < next_module.weight.size(1):
    #                                             end_idx = min(end_idx, next_module.weight.size(1))
    #                                             next_module.weight.data[:, start_idx:end_idx] = 0
    #                                 else:  # Linear
    #                                     if isinstance(next_module, nn.Linear):
    #                                         # Linear -> Linear
    #                                         if idx < next_module.weight.size(1):
    #                                             next_module.weight.data[:, idx] = 0
    #                             except IndexError as e:
    #                                 print(f"Warning: Index error when resetting weights. Layer: {name}, "
    #                                     f"Index: {idx}, Next layer: {next_layer_name}")
    #                                 continue
                            
    #                         # 3. 重置效用值和年龄
    #                         self.utilities[name][idx] = 0
    #                         self.ages[name][idx] = 0


class ResNet_mnist(nn.Module):
    def __init__(self, args, block, num_blocks, num_classes=10):
        super(ResNet_mnist, self).__init__()
        self.inplanes = 16
        print('num_classes:', num_classes)
        # declare lambda array
        num_block = 0
        for i in num_blocks:
            num_block += i
        print('block number:', num_block)

        self.conv1 = nn.Conv2d(args.num_channels, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 1, padding =1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual
        return out



# def conv_dw(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
#         nn.BatchNorm2d(inp),
#         nn.ReLU(inplace=True),

#         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU(inplace=True),
#     )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

# class MobileNet(nn.Module):
#     def __init__(self, args):
#         super(MobileNet, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 32, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),

#             conv_dw(32, 64, 1),
#             conv_dw(64, 128, 2),
#             conv_dw(128, 128, 1),
#             conv_dw(128, 256, 2),
#             conv_dw(256, 256, 1),
#             conv_dw(256, 512, 2),
#             conv_dw(512, 512, 1),
#             conv_dw(512, 512, 1),
#             conv_dw(512, 512, 1),
#             conv_dw(512, 512, 1),
#             conv_dw(512, 512, 1),
#             conv_dw(512, 1024, 2),
#             conv_dw(1024, 1024, 1),

#             nn.AdaptiveAvgPool2d(1)
#         )
#         self.fc = nn.Linear(1024, args.num_classes)

#     def forward(self, x):
#         x = self.model(x)
#         x = x.view(-1, 1024)
#         x = self.fc(x)
#         return {'output': x}  # 返回一个字典，其中包含 'output' 键
class MobileNet(nn.Module):
    def __init__(self, args):
        super(MobileNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
       
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, args.num_classes)
        )

    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx < 0:
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)

        out = self.features(x)
        out = out.view(out.size(0), -1)  # Flatten for the fully connected layer
        out = self.classifier(out)

        result = {
            'logit': out,
            'output': out
        }
        return result

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = self.classifier(z_input)
        result = {'output': z}
        if logit:
            result['logit'] = z
        return result
    
class MobileNet_Drop(nn.Module):
    def __init__(self, args):
        super(MobileNet_Drop, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            nn.Dropout(p=0.5),  # 第一个Dropout层，放在早期层之后
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            nn.Dropout(p=0.5),  # 第二个Dropout层，放在中间层之后
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            nn.Dropout(p=0.5),  # 第三个Dropout层，放在较深层之后
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            nn.Dropout(p=0.5),  # 第四个Dropout层，放在接近末尾层之后
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # 全连接层前的Dropout
            nn.Linear(1024, args.num_classes)
        )

    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx < 0:
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)

        out = self.features(x)
        out = out.view(out.size(0), -1)  # Flatten for the fully connected layer
        out = self.classifier(out)

        result = {
            'logit': out,
            'output': out
        }
        return result

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = self.classifier(z_input)
        result = {'output': z}
        if logit:
            result['logit'] = z
        return result
    
# class MobileNet(nn.Module):
#     def __init__(self, args):
#         super(MobileNet, self).__init__()
        
#         # 初始化模块列表
#         self.conv_blocks = nn.ModuleList()
        
#         # 初始卷积层（非DW结构）
#         self.features_init = nn.Sequential(
#             nn.Conv2d(3, 32, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True)
#         )
        
#         # 添加DW卷积块（参数自动命名）
#         self._add_dw_block(32, 64, 1)    # conv_blocks.0
#         self._add_dw_block(64, 128, 2)   # conv_blocks.1
#         self._add_dw_block(128, 128, 1)  # conv_blocks.2
#         self._add_dw_block(128, 256, 2)  # conv_blocks.3
#         self._add_dw_block(256, 256, 1)  # conv_blocks.4
#         self._add_dw_block(256, 512, 2)  # conv_blocks.5
#         self._add_dw_block(512, 512, 1)  # conv_blocks.6
#         self._add_dw_block(512, 512, 1)  # conv_blocks.7
#         self._add_dw_block(512, 512, 1)  # conv_blocks.8
#         self._add_dw_block(512, 512, 1)  # conv_blocks.9
#         self._add_dw_block(512, 1024, 2) # conv_blocks.10
#         self._add_dw_block(1024, 1024, 1)# conv_blocks.11
        
#         # 最终分类层
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(1024, args.num_classes)

#     def _add_dw_block(self, inp, oup, stride):
#         """添加深度可分离卷积块"""
#         self.conv_blocks.append(
#             conv_dw(inp, oup, stride)
#         )

#     def forward(self, x):
#         x = self.features_init(x)
#         for block in self.conv_blocks:
#             x = block(x)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return {'output': x, 'logit': x}