"""
models.py: Model definitions and utilities
"""

import torch
import torch.nn as nn
import os
import pickle
import numpy as np

num_rot_conv_layers = 7
num_rot_batch_norm_layers = 7

class Stage2CountingNet(nn.Module):
    """
    Counting net to use for stage2 training
    """
    def __init__(self, name='stage2'):
        """
        Initialise Stage2CountingNet class.

        Parameters
        ----------
        name: string
            an alias for the network
        """
        super(Stage2CountingNet, self).__init__()
        self.name = name
        if torch.cuda.is_available():
            self.rgb_means = torch.cuda.FloatTensor([104.008, 116.669, 122.675])
        else:
            self.rgb_means = torch.FloatTensor([104.008, 116.669, 122.675])
        self.rgb_means = torch.autograd.Variable(self.rgb_means,
                                                 requires_grad=False).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        layers = []
        in_channels = 3

        self.relu = nn.functional.relu

        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.batch_norm_1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.batch_norm_1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.batch_norm_2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.batch_norm_2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.batch_norm_3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.batch_norm_3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.batch_norm_3_3 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(384, 128, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01, mean=0.0)
                try:
                    m.bias.data.zero_()
                except:
                    continue
            elif isinstance(m, nn.Linear):
                assert(0)

    def forward(self, x):
        mean_sub_input = x
        mean_sub_input -= self.rgb_means

        main_out_block1 = self.relu(self.batch_norm_1_2(self.conv1_2(self.relu(self.batch_norm_1_1(self.conv1_1(mean_sub_input))))))
        main_out_pool1 = self.pool1(main_out_block1)

        main_out_block2 = self.relu(self.batch_norm_2_2(self.conv2_2(self.relu(self.batch_norm_2_1(self.conv2_1(main_out_pool1))))))
        main_out_pool2 = self.pool2(main_out_block2)

        main_out_block3 = self.relu(self.batch_norm_3_3(self.conv3_3(self.relu(self.batch_norm_3_2(self.conv3_2(self.relu(self.batch_norm_3_1(self.conv3_1(main_out_pool2)))))))))

        hyper_out = torch.cat((main_out_pool2, main_out_block3), dim=1)

        main_out_block4 = self.relu(self.conv4_2(self.relu(self.conv4_1(hyper_out))))
        main_out_pool4 = self.pool4(main_out_block4)

        main_out_block5 = self.relu(self.conv5_1(main_out_pool4))
        return main_out_block5

class Stage1CountingNet(nn.Module):
    """
    Counting net to use for stage1 training
    """
    def __init__(self, name='stage1'):
        """
        Initialise Stage1CountingNet class.

        Parameters
        ----------
        name: string
            an alias for the network
        """
        super(Stage1CountingNet, self).__init__()
        self.name = name
        if torch.cuda.is_available():
            self.rgb_means = torch.cuda.FloatTensor([104.008, 116.669, 122.675])
        else:
            self.rgb_means = torch.FloatTensor([104.008, 116.669, 122.675])
        self.rgb_means = torch.autograd.Variable(self.rgb_means,
                                                 requires_grad=False).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        layers = []
        in_channels = 3

        self.relu = nn.functional.relu

        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.batch_norm_1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.batch_norm_1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.batch_norm_2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.batch_norm_2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.batch_norm_3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.batch_norm_3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.batch_norm_3_3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.batch_norm_4_1 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.batch_norm_5_1 = nn.BatchNorm2d(64)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (B,C,1,1)
        self.fc = nn.Linear(64*1*1, 4)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                m.bias.data.zero_()

    def forward(self, x):
        mean_sub_input = x
        mean_sub_input -= self.rgb_means

        main_out_block1 = self.relu(self.batch_norm_1_2(self.conv1_2(self.relu(self.batch_norm_1_1(self.conv1_1(mean_sub_input))))))
        main_out_pool1 = self.pool1(main_out_block1)

        main_out_block2 = self.relu(self.batch_norm_2_2(self.conv2_2(self.relu(self.batch_norm_2_1(self.conv2_1(main_out_pool1))))))
        main_out_pool2 = self.pool2(main_out_block2)

        main_out_block3 = self.relu(self.batch_norm_3_3(self.conv3_3(self.relu(self.batch_norm_3_2(self.conv3_2(self.relu(self.batch_norm_3_1(self.conv3_1(main_out_pool2)))))))))
        main_out_pool3 = self.pool3(main_out_block3)

        main_out_block4 = self.relu(self.batch_norm_4_1(self.conv4_1(main_out_pool3)))
        main_out_pool4 = self.pool4(main_out_block4)

        main_out_block5 = self.relu(self.batch_norm_5_1(self.conv5_1(main_out_pool4)))

        global_avg_pool_out = self.avg_pool(main_out_block5)
        fc_out = self.fc(global_avg_pool_out.view(global_avg_pool_out.size(0), -1))
        return fc_out

def load_rot_model_blocks(network, snapshot_path, excluded_layers):
    """
    Loading Feature Extraction Network (FEN) for stage2 training

    Parameters
    ----------
    network: Stage2CountingNet object
        uninitialised random Stage2CountingNetwork
    snapshot_path: str
        directory path to load weights for FEN
    excluded_layers: list
        ignore loading particular layers
    """
    best_epoch_file_name = open(os.path.join(snapshot_path,'unsup_vgg_best_model_meta.pkl'),'rb')
    best_epoch_file_name = pickle.load(best_epoch_file_name)

    print('Loading Stage 1 best epoch model :{}'.format(best_epoch_file_name))
    model_checkpoint = torch.load(os.path.join(snapshot_path,best_epoch_file_name))
    count = 0
    parameter_count = 0

    for name, module in network.named_children():
        if name.startswith('conv') and name not in excluded_layers:
            module.weight.data.copy_(model_checkpoint['state_dict']['{}.weight'.format(name)])
            module.weight.requires_grad = False
            parameter_count +=1
            if module.bias != None:
                module.bias.data.copy_(model_checkpoint['state_dict']['{}.bias'.format(name)])
                module.bias.requires_grad = False
                parameter_count+=1
            count += 1
        elif name.startswith('batch_norm') and name not in excluded_layers:
            module.weight.data.copy_(model_checkpoint['state_dict']['{}.weight'.format(name)])
            parameter_count += 1
            module.bias.data.copy_(model_checkpoint['state_dict']['{}.bias'.format(name)])
            parameter_count += 1

            module.weight.requires_grad = False
            module.bias.requires_grad = False

            module.running_mean.requires_grad = False
            module.running_var.requires_grad = False

            module.running_mean.data.copy_(model_checkpoint['state_dict']['{}.running_mean'.format(name)])
            parameter_count += 1
            module.running_var.data.copy_(model_checkpoint['state_dict']['{}.running_var'.format(name)])
            parameter_count += 1

            module.eval() # freeze batch norm
            count += 1
    assert (count == (num_rot_conv_layers + num_rot_batch_norm_layers))
    assert (parameter_count == (num_rot_conv_layers*1 + num_rot_batch_norm_layers*4))
    return network

def check_BN_no_gradient_change(network, exclude_list=[]):
    """
    checking if BN weights are not being updated

    Parameters
    ----------
    network: Stage2CountingNet object
    excluded_layers: list
        ignore checking particular layers
    """
    s = []
    count = 0
    for name, module in network.named_children():
        if name.startswith('batch_norm') and name not in exclude_list:
            count += 1
            s.append(module.running_mean.data.cpu().detach().numpy().reshape(-1))
    assert (count == num_rot_batch_norm_layers)
    return np.concatenate(s)

def check_conv_no_gradient_change(network, exclude_list=[]):
    """
    checking if conv weights are not being updated

    Parameters
    ----------
    network: Stage2CountingNet object
    excluded_layers: list
        ignore checking particular layers
    """
    s = []
    count = 0
    for name, module in network.named_children():
        if name.startswith('conv') and name not in exclude_list:
            assert (module.weight.requires_grad == False)
            s.append(module.weight.data.cpu().detach().numpy().reshape(-1))
            count += 1
    assert (count == num_rot_conv_layers)
    return np.concatenate(s)

def set_batch_norm_to_eval(network):
    """
    setting all batch norm layers to eval mode

    Parameters
    ----------
    network: Stage2CountingNet object
    """
    count = 0
    for name, module in network.named_children():
        if name.startswith('batch_norm'):
            module.eval()
            count += 1
    assert (count == num_rot_batch_norm_layers)
    return network

def load_net(networks, fdir, name, set_epoch=True):
    """
    setting all batch norm layers to eval mode

    Parameters
    ----------
    networks: Stage2CountingNet object
    fdir: str
        Directory to load the network from
    name: str
        Name of the checkpoint to be loaded
    set_epoch: bool
        to resume training
    """
    net = networks

    filepath = os.path.join(fdir, name)
    print("Loading file...", filepath)

    if not os.path.isfile(filepath):
        print("Checkpoint file" + filepath + " not found!")
        raise IOError

    checkpoint_1 = torch.load(filepath)

    if set_epoch:
        try:
            args.start_epoch = checkpoint_1['epoch']
        except NameError:
            pass
    net.load_state_dict(checkpoint_1['state_dict'])
    print("=> loaded checkpoint '{}' ({} epochs over)".format(filepath, checkpoint_1['epoch']))
    return net
