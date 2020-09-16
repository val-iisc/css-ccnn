"""
test_model.py: testing script
"""

import argparse
import datetime
import os
import random

import cv2
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

from crowd_dataset import CrowdDataset
from models import Stage2CountingNet, load_net, set_batch_norm_to_eval

matplotlib.use('Agg')



parser = argparse.ArgumentParser(description='Test CSS-CCNN Model')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU number')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N',
                    help='mini-batch size (default: 4),only used for train')
parser.add_argument('--patches', default=1, type=int, metavar='N',
                    help='number of patches per image')
parser.add_argument('--dataset', default="parta", type=str,
                    help='dataset to train on')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--trained-model', default='', type=str, metavar='PATH', help='filename of model to load',
                    nargs='+')
parser.add_argument('--loss', default='sinkhorn', type=str, help="loss to use: mse or sinkhorn")
parser.add_argument('--kernel_size', default=8, type=int, help="kernel size for summing counts")
parser.add_argument('--sinkhorn_epsilon', default=0.1, type=float, help="entropy regularisation weight in sinkhorn")
parser.add_argument('--sinkhorn_batch_size', default=4, type=int, help="points to sample from distribution")
parser.add_argument('--sinkhorn_iterations', default=1000, type=int, help="no of iterations in sinkhorn")
parser.add_argument('--best_model_name', default=None, type=str, help="name of the best model checkpoint")

sampled_GT = None

def log(f, txt, do_print=1):
    txt = str(datetime.datetime.now()) + ': ' + txt
    if do_print == 1:
        print(txt)
    f.write(txt + '\n')

def get_filename(net_name, epochs_over):
    return net_name + "_epoch_" + str(epochs_over) + ".pth"

def print_graph(maps, title, save_path):
    fig = plt.figure()
    st = fig.suptitle(title)
    for i, (map, args) in enumerate(maps):
        plt.subplot(1, len(maps), i + 1)
        if len(map.shape) > 2 and map.shape[0] == 3:
            plt.imshow(map.transpose((1, 2, 0)).astype(np.uint8),aspect='equal', **args)
        else:
            plt.imshow(map, aspect='equal', **args)
            plt.axis('off')
    plt.savefig(save_path + ".png", bbox_inches='tight', pad_inches = 0)
    fig.clf()
    plt.clf()
    plt.close()


@torch.no_grad()
def test_function(X, Y, network):
    X = torch.autograd.Variable(torch.from_numpy(X)).cuda()
    Y = torch.autograd.Variable(torch.from_numpy(Y)).cuda()

    network = network.cuda()
    network.eval()
    output = network(X) # (B,1,h,w)
 
    count_error = torch.abs(torch.sum(Y.view(Y.size(0), -1), dim=1) - torch.sum(output.view(output.size(0), -1), dim=1))

    network.train()
    network = set_batch_norm_to_eval(network)
    return output.cpu().detach().numpy(), count_error.cpu().detach().numpy()


def test_network(dataset, set_name, network, print_output=False):
    if isinstance(print_output, str):
        print_path = print_output
    elif isinstance(print_output, bool) and print_output:
        print_path = './models_stage_2/dump'
    else:
        print_path = None

    count_error_list = []
    for idx, data in enumerate(dataset.test_get_data(set_name)):
        image_name, Xs, Ys = data
        image = Xs[0].transpose((1, 2, 0))
        image = cv2.resize(image, (image.shape[1] // output_downscale, image.shape[0] // output_downscale))

        pred_dmap, count_error = test_function(Xs, Ys, network)

        max_val = max(np.max(pred_dmap[0, 0].reshape(-1)), np.max(Ys[0, 0].reshape(-1)))
        maps = [(np.transpose(image,(2,0,1)), {}),
                (pred_dmap[0,0], {'cmap': 'jet', 'vmin': 0., 'vmax': max_val}),
                (Ys[0,0], {'cmap': 'jet', 'vmin': 0., 'vmax': max_val})]

        count_error_list.append(count_error)

        if print_path:
            print_graph(maps, "Gt:{},Pred:{}".format(np.sum(Ys),np.sum(pred_dmap)), os.path.join(print_path, image_name))

    mae = np.mean(count_error_list)
    mse = np.sqrt(np.mean(np.square(count_error_list)))
    return {'mae':mae, 'mse': mse}, mae

def train_network():
    network = Stage2CountingNet()
    model_save_dir = './models_stage_2'
    model_save_path = os.path.join(model_save_dir, 'train2')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        os.makedirs(os.path.join(model_save_path, 'snapshots'))
        os.makedirs(os.path.join(model_save_dir,'dump'))
        os.makedirs(os.path.join(model_save_dir,'dump_test'))
    global f
    snapshot_path = os.path.join(model_save_path, 'snapshots')

    network = load_net(network, snapshot_path, get_filename(network.name, args.best_model_name))
    print(network)
    epoch_test_losses, mae = test_network(dataset, 'test', network, print_output=os.path.join(model_save_dir,'dump_test'))
    print('TEST mae, mse:' + str(epoch_test_losses))
    return


if __name__ == '__main__':
    args = parser.parse_args()
    # -- Assign GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # -- Assertions
    assert (args.dataset)

    # -- Setting seeds for reproducability
    np.random.seed(11)
    random.seed(11)
    torch.manual_seed(11)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(11)
    torch.cuda.manual_seed_all(11)

    # -- Dataset paths
    if args.dataset == "parta":
        validation_set = 30
        output_downscale = 8
        path = '../../dataset/ST_partA/'
    elif args.dataset == "ucfqnrf":
        validation_set = 240
        output_downscale = 8
        path = "../../dataset/UCF-QNRF_ECCV18"
        
    model_save_dir = './models'

    batch_size = args.batch_size

    dataset = CrowdDataset(path, name=args.dataset, valid_set_size=validation_set,
                           gt_downscale_factor=output_downscale)
    print(dataset.data_files['test_valid'], len(dataset.data_files['test_valid']))
    print(dataset.data_files['train'], len(dataset.data_files['train']))

    # -- Train the model
    train_network()
