"""
stage2_main.py: stage 2 training script
"""

import argparse
import datetime
import os
import pickle
import random

import cv2
import matplotlib
import numpy as np
import powerlaw
import torch
from matplotlib import pyplot as plt
from mpmath import gammainc
from torch import nn as nn
from torch import optim as optim

from crowd_dataset import CrowdDataset
from models import (Stage2CountingNet, check_BN_no_gradient_change,
                    check_conv_no_gradient_change, load_net,
                    load_rot_model_blocks, set_batch_norm_to_eval)
from sinkhorn import SinkhornSolver

matplotlib.use('Agg')


parser = argparse.ArgumentParser(description='CSS-CSNN++ Stage-2 Training')
parser.add_argument('--epochs', default=600, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU number')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 32),only used for train')
parser.add_argument('--patches', default=1, type=int, metavar='N',
                    help='number of patches per image')
parser.add_argument('--dataset', default="parta", type=str,
                    help='dataset to train on')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--loss', default='sinkhorn', type=str,
                    help="loss to use: mse or sinkhorn")
parser.add_argument('--kernel_size', default=8, type=int,
                    help="kernel size for summing counts")
parser.add_argument('--sinkhorn_epsilon', default=0.1, type=float,
                    help="entropy regularisation weight in sinkhorn")
parser.add_argument('--sbs', '--sinkhorn_batch_size', default=32,
                    type=int, help="points to sample from distribution")
parser.add_argument('--sinkhorn_iterations', default=1000,
                    type=int, help="no of iterations in sinkhorn")
parser.add_argument('--seed', default=11, type=int, help="seed to use")
parser.add_argument('--alpha', default=2.0, type=float, help="shape parameter of power law distribution")
parser.add_argument('--cmax', default=3500, type=int, help="the maximum value")
parser.add_argument('--scrop', default=4, type=int, help="patch approximation parameter")
parser.add_argument('--num_samples', default=482, type=int, help="number of samples")
parser.add_argument('--patience', default=300, type=int, help="epochs to train before stopping")
parser.add_argument('--ma_window', default=10, type=int, help="window for computing moving average")
parser.add_argument('--percentile_thresh', default=0.3, type=float, help="percentile for splitting regios")
parser.add_argument('--dense_weight', default=0.1, type=float, help="weight for dense")
sampled_GT = None
blur_sigma = None

# -- Compute CDF for Truncated Power Law Distribution
def get_cdf(x, alpha, Lambda):
        CDF = ( (gammainc(1-alpha, Lambda*x)) /
                    Lambda**(1-alpha)
                        )
        return 1-CDF

# -- Obtain Lambda from max count
def get_lambda():
    m, n = 4, 4
    max_value = args.cmax / (args.scrop * m * n) 

    for Lambda_t in np.arange(0.001, 0.1, 0.001):
        cdf = get_cdf(max_value, args.alpha, Lambda_t)
        if cdf > 1 - 1. / args.num_samples:
            return Lambda_t

# -- Get shift thresh
def get_shift_thresh():
    Lambda = get_lambda()
    for value in np.arange(1.00001, 10, 0.01):
        cdf = get_cdf(value, args.alpha, Lambda)
        if cdf > 0.28:
            return value

def log(f, txt, do_print=1):
    txt = str(datetime.datetime.now()) + ': ' + txt
    if do_print == 1:
        print(txt)
    f.write(txt + '\n')

def get_filename(net_name, epochs_over):
    return net_name + "_epoch_" + str(epochs_over) + ".pth"


def save_checkpoint(state, fdir, name='checkpoint.pth'):
    filepath = os.path.join(fdir, name)
    torch.save(state, filepath)


def print_graph(maps, title, save_path):
    fig = plt.figure()
    st = fig.suptitle(title)
    for i, (map, args) in enumerate(maps):
        plt.subplot(1, len(maps), i + 1)
        if len(map.shape) > 2 and map.shape[0] == 3:
            plt.imshow(map.transpose((1, 2, 0)).astype(
                np.uint8), aspect='equal', **args)
        else:
            plt.imshow(map, aspect='equal', **args)
            plt.axis('off')
    plt.savefig(save_path + ".png", bbox_inches='tight', pad_inches=0)
    fig.clf()
    plt.clf()
    plt.close()


excluded_layers = ['conv4_1', 'conv4_2', 'conv5_1']


def get_loss_criterion():
    if args.loss == 'mse':
        return nn.MSELoss(size_average=True)
    elif args.loss == 'sinkhorn':
        return SinkhornSolver(epsilon=args.sinkhorn_epsilon, iterations=args.sinkhorn_iterations)
    else:
        raise NotImplementedError

# -- Create edge density maps
def create_pseudo_density(Xs):
    global blur_sigma
    kernal_size_from_actual = 5
   
    pseudo_density_maps = []
    for i in range(Xs.shape[0]):
        image = Xs[i].transpose((1,2,0)).astype('uint8')#(224,224,3)
    
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.Canny(gray, 225, 250)
        gray = cv2.resize(gray,(gray.shape[1]//output_downscale, gray.shape[0]//output_downscale))

        blur = cv2.GaussianBlur(gray,(kernal_size_from_actual,kernal_size_from_actual), sigmaX = blur_sigma)

        orig_blur = blur.copy()
        blur = blur.astype('float32') / 255
        blur = blur * 0.8 / 10
        
        pseudo_density_maps.append(blur[None,...])
    
    pseudo_density_maps = np.array(pseudo_density_maps)
    pseudo_density_maps = pseudo_density_maps / np.max(pseudo_density_maps)
    return pseudo_density_maps

def train_function(Xs, Ys, network, optimizer):
    network = network.cuda()
    optimizer.zero_grad()

    X = torch.autograd.Variable(torch.from_numpy(Xs)).cuda()
    Y = torch.autograd.Variable(torch.FloatTensor(Ys)).cuda()

    outputs = network(X)

    losses = []
    loss = 0.0
    loss_criterion = get_loss_criterion()

    pseudo_density_maps = create_pseudo_density(Xs)
    pseudo_density_maps = torch.from_numpy(pseudo_density_maps).cuda()
    
    avg_pool = nn.AvgPool2d(kernel_size=args.kernel_size,
                            stride=args.kernel_size)
    output_reshape_ = avg_pool(outputs) * (args.kernel_size * args.kernel_size)
    pseudo_reshape_ = avg_pool(pseudo_density_maps) * (args.kernel_size * args.kernel_size)
    output_reshape = output_reshape_.view(-1, 1)
    pseudo_reshape = pseudo_reshape_.view(-1, 1)

    # -- Split predictions into sparse, dense using percentile_thresh
    pseudo_median = pseudo_reshape.topk(int(args.percentile_thresh*len(pseudo_reshape)), dim=0)[0][-1:][0]
    Y_median = Y.topk(int(args.percentile_thresh*(len(Y))), dim=0)[0][-1:][0]
    a_output_indices = pseudo_reshape < pseudo_median
    a_Y_indices = Y < Y_median

    if a_output_indices.sum() > 2:
        loss_sparse = loss_criterion(output_reshape[a_output_indices].view(-1, 1), Y[a_Y_indices].view(-1, 1))
        loss_dense = loss_criterion(output_reshape[~a_output_indices].view(-1, 1), Y[~a_Y_indices].view(-1, 1))
        loss = (loss_sparse + loss_dense)* 0.01
    else:
        loss = loss_criterion(output_reshape, Y.view(-1, 1)) * 0.01

    assert(loss.grad_fn != None)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    return loss.item()


@torch.no_grad()
def test_function(Xs, Ys, network, set_name=None):
    assert(set_name is not None)

    X = torch.autograd.Variable(torch.from_numpy(Xs)).cuda()
    Y = torch.autograd.Variable(torch.from_numpy(Ys)).float().cuda()

    network = network.cuda()
    network.eval()
    output = network(X)  # (B,1,h,w)

    loss = 0.0
    loss_criterion = get_loss_criterion()

    avg_pool = nn.AvgPool2d(kernel_size=args.kernel_size,
                            stride=args.kernel_size)

    output_reshape_ = avg_pool(output) * (args.kernel_size * args.kernel_size)
    
    if set_name == 'test_valid':
        pseudo_density_maps = create_pseudo_density(Xs)
        pseudo_density_maps = torch.from_numpy(pseudo_density_maps).cuda()

        pseudo_reshape_ = avg_pool(pseudo_density_maps) * (args.kernel_size * args.kernel_size)
        output_reshape = output_reshape_.view(-1, 1)
        pseudo_reshape = pseudo_reshape_.view(-1, 1)

        pseudo_median = pseudo_reshape.topk(int(args.percentile_thresh*len(pseudo_reshape)), dim=0)[0][-1:][0]
        Y_median = Y.topk(int(args.percentile_thresh*(len(Y))), dim=0)[0][-1:][0]
        a_output_indices = pseudo_reshape < pseudo_median
        a_Y_indices = Y < Y_median

        if a_output_indices.sum() > 2:
            loss_sparse = loss_criterion(output_reshape[a_output_indices].view(-1, 1), Y[a_Y_indices].view(-1, 1))
            loss_dense = loss_criterion(output_reshape[~a_output_indices].view(-1, 1), Y[~a_Y_indices].view(-1, 1))
            loss = (loss_sparse + loss_dense)* 0.01
        else:
            loss = loss_criterion(output_reshape, Y.view(-1, 1)) * 0.01
    else:
        output_reshape = output_reshape_.view(-1, 1)        
        loss = loss_criterion(output_reshape, Y.view(-1, 1)) * 0.01
        count_error = torch.abs(torch.sum(Y.view(Y.size(0), -1), dim=1) - torch.sum(output.view(output.size(0), -1), dim=1))

    network.train()
    network = set_batch_norm_to_eval(network)

    if set_name == "test_valid":
        return loss.item(), loss_sparse.item(), loss_dense.item(), output.cpu().detach().numpy()
    else:
        return loss.item(), output.cpu().detach().numpy(), count_error.cpu().detach().numpy()

def test_network(dataset, set_name, network, print_output=False):
    assert(set_name == "test")
    if isinstance(print_output, str):
        print_path = print_output
    elif isinstance(print_output, bool) and print_output:
        print_path = './models_stage_2/dump'
    else:
        print_path = None

    loss_list = []
    count_error_list = []
    for idx, data in enumerate(dataset.test_get_data(set_name)):
        image_name, Xs, Ys = data
        image = Xs[0].transpose((1, 2, 0))
        image = cv2.resize(
            image, (image.shape[1] // output_downscale, image.shape[0] // output_downscale))

        loss, pred_dmap, count_error = test_function(Xs, Ys, network, set_name)
        max_val = max(
            np.max(pred_dmap[0, 0].reshape(-1)), np.max(Ys[0, 0].reshape(-1)))
        maps = [(np.transpose(image, (2, 0, 1)), {}),
                (pred_dmap[0, 0], {'cmap': 'jet',
                                   'vmin': 0., 'vmax': max_val}),
                (Ys[0, 0], {'cmap': 'jet', 'vmin': 0., 'vmax': max_val})]

        loss_list.append(loss)
        count_error_list.append(count_error)

        # -- Plotting visualisations
        if print_path:
            print_graph(maps, "Gt:{},Pred:{}".format(np.sum(Ys), np.sum(
                pred_dmap)), os.path.join(print_path, image_name))

    loss = np.mean(loss_list)
    mae = np.mean(count_error_list)
    return {'loss1': loss, 'new_mae': mae}, mae
        

def val_network(dataset, set_name, network, print_output=False):
    assert(set_name == "test_valid")

    loss_list = []
    loss_sparse_list, loss_dense_list = [], []
    count_error_list = []

    num_batches_per_epoch = 5 * len(dataset.data_files['test_valid']) // batch_size
    dataset.val_pos_counter = 0
    dataset.val_iterator = None
    
    for b_i in range(num_batches_per_epoch):
        Xs, _ = dataset.val_get_data(min(validation_set, args.batch_size))
    
        loss, loss_sparse, loss_dense, pred_dmap = test_function(Xs, sampled_GT, network, set_name)

        loss_list.append(loss)
        loss_sparse_list.append(loss_sparse)
        loss_dense_list.append(loss_dense)

    loss = np.mean(loss_list)
    loss_s = np.mean(loss_sparse_list)
    loss_d = np.mean(loss_dense_list)
    return {'loss1': loss, 'loss_sparse': loss_s, 'loss_dense': loss_d}, None
            

def train_network():
    network = Stage2CountingNet()
    model_save_dir = './models_stage_2'
    model_save_path = os.path.join(model_save_dir, 'train2')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        os.makedirs(os.path.join(model_save_path, 'snapshots'))
        os.makedirs(os.path.join(model_save_dir, 'dump'))
        os.makedirs(os.path.join(model_save_dir, 'dump_test'))
    global f
    snapshot_path = os.path.join(model_save_path, 'snapshots')
    f = open(os.path.join(model_save_path, 'train0.log'), 'w')

    # -- Logging Parameters
    log(f, 'args: ' + str(args))
    log(f, 'model: ' + str(network), False)
    log(f, 'Stage2...')
    log(f, 'LR: %.12f.' % (args.lr))

    start_epoch = 0
    num_epochs = args.epochs
    valid_losses = {}
    train_losses = {}
    for metric in ['loss1', 'loss_sparse', 'loss_dense']:
        valid_losses[metric] = []
        
    for metric in ['loss1']:
        train_losses[metric] = []

    batch_size = args.batch_size
    args.percentile_thresh = float("{0:.2f}".format(1 - args.percentile_thresh))
    num_train_images = len(dataset.data_files['train'])
    num_patches_per_image = args.patches
    assert(batch_size < (num_patches_per_image * num_train_images))
    num_batches_per_epoch = num_patches_per_image * num_train_images // batch_size
    assert(num_batches_per_epoch >= 1)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, network.parameters()),
                          lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    network = load_rot_model_blocks(
        network, snapshot_path='models_stage_1/train2/snapshots/', excluded_layers=excluded_layers)

    shift_thresh = get_shift_thresh()
    Lambda = get_lambda()

    log(f, "Shift thresh: {}, Lambda: {}".format(shift_thresh, Lambda))

    # -- Main Training Loop
    min_valid_loss = 100.
    min_valid_sparse_loss = 100.
    min_valid_epoch = -1
    min_valid_sparse_loss_epoch = -1

    before_BN_weights_sum = check_BN_no_gradient_change(
        network, exclude_list=excluded_layers)
    before_conv_weights_sum = check_conv_no_gradient_change(
        network, exclude_list=excluded_layers)

    stop_training = False

    global sampled_GT

    for e_i, epoch in enumerate(range(start_epoch, num_epochs)):
        avg_loss = []

        # b_i - batch index
        for b_i in range(num_batches_per_epoch):
            # Generate next training sample
            Xs, _Ys = dataset.train_get_data(batch_size=args.batch_size)

            after_conv_weights_sum = check_conv_no_gradient_change(
                network, exclude_list=excluded_layers)
            assert (np.all(before_conv_weights_sum == after_conv_weights_sum))

            sampled_GT = None
            sampled_GT_shape = args.sbs * 7 * 7 * \
                (8 // args.kernel_size) * (8 // args.kernel_size)

            sampling_parameters = [args.alpha, Lambda]
            sampled_GT = powerlaw.Truncated_Power_Law(
                parameters=sampling_parameters).generate_random(sampled_GT_shape)
            
            for s_i, s_val in enumerate(sampled_GT):
                if s_val < shift_thresh:
                    sampled_GT[s_i] = np.random.uniform(0, shift_thresh)
            assert(sampled_GT.shape[0] == (
                sampled_GT_shape) and sampled_GT.ndim == 1)

            train_loss = train_function(
                Xs, sampled_GT, network, optimizer)
            avg_loss.append(train_loss)

            # Logging losses after 1k iterations.
            if b_i % 1 == 0:
                log(f, 'Epoch %d [%d]: %s loss: %s.' %
                    (epoch, b_i, [network.name], train_loss))
            after_BN_weights_sum = check_BN_no_gradient_change(
                network, exclude_list=excluded_layers)
            after_conv_weights_sum = check_conv_no_gradient_change(
                network, exclude_list=excluded_layers)
            
            assert (np.all(before_BN_weights_sum == after_BN_weights_sum))
            assert (np.all(before_conv_weights_sum == after_conv_weights_sum))
        
        # -- Stats update
        avg_loss = np.mean(np.array(avg_loss))
        train_losses['loss1'].append(avg_loss)
        log(f, 'TRAIN epoch: ' + str(epoch) +
            ' train mean loss1:' + str(avg_loss))

        torch.cuda.empty_cache()

        log(f, 'Validating...')

        epoch_val_losses, valid_mae = val_network(
            dataset, 'test_valid', network)
        log(f, 'TEST valid epoch: ' + str(epoch) +
            ' test valid loss1, mae' + str(epoch_val_losses))

        for metric in ['loss1', 'loss_sparse', 'loss_dense']:
            valid_losses[metric].append(epoch_val_losses[metric])
        
        if e_i > args.ma_window:
            valid_losses_sparse_smooth = np.mean(valid_losses['loss_sparse'][-args.ma_window:])
            valid_losses_dense_smooth = np.mean(valid_losses['loss_dense'][-args.ma_window:])
            valid_losses_smooth = valid_losses_sparse_smooth + args.dense_weight * valid_losses_dense_smooth
            
            if valid_losses_sparse_smooth < min_valid_sparse_loss:
                min_valid_sparse_loss = valid_losses_sparse_smooth
                min_valid_sparse_loss_epoch = e_i
            #Check out for divergence in sparse loss
            if valid_losses_sparse_smooth > (min_valid_sparse_loss + 1.):
                stop_training = True
                min_valid_epoch = min_valid_sparse_loss_epoch
            elif valid_losses_smooth < min_valid_loss:
                min_valid_loss = valid_losses_smooth
                min_valid_epoch = e_i
                count = 0
            else:
                count = count + 1
                if count > args.patience:
                    stop_training = True

        log(f, 'Best valid so far epoch: {}, valid_loss: {}'.format(min_valid_epoch,
                                                                    valid_losses['loss1'][min_valid_epoch]))
        # Save networks
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': network.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, snapshot_path, get_filename(network.name, epoch + 1))

        print('saving graphs...')
        with open(os.path.join(snapshot_path, 'losses.pkl'), 'wb') as lossfile:
            pickle.dump((train_losses, valid_losses),
                        lossfile, protocol=2)

        for metric in train_losses.keys():
            if "maxima_split" not in metric:
                if isinstance(train_losses[metric][0], list):
                    for i in range(len(train_losses[metric][0])):
                        plt.plot([a[i] for a in train_losses[metric]])
                        plt.savefig(os.path.join(snapshot_path,
                                                 'train_%s_%d.png' % (metric, i)))
                        plt.clf()
                        plt.close()
                plt.plot(train_losses[metric])
                plt.savefig(os.path.join(
                    snapshot_path, 'train_%s.png' % metric))
                plt.clf()
                plt.close()

        for metric in valid_losses.keys():
            if isinstance(valid_losses[metric][0], list):
                for i in range(len(valid_losses[metric][0])):
                    plt.plot([a[i] for a in valid_losses[metric]])
                    plt.savefig(os.path.join(snapshot_path,
                                             'valid_%s_%d.png' % (metric, i)))
                    plt.clf()
                    plt.close()
            plt.plot(valid_losses[metric])
            plt.savefig(os.path.join(snapshot_path, 'valid_%s.png' % metric))
            plt.clf()
            plt.close()

        if stop_training:
            break

    network = load_net(network, snapshot_path, get_filename(
        network.name, min_valid_epoch))
    log(f, 'Testing on best model {}'.format(min_valid_epoch))
    epoch_test_losses, mae = test_network(
        dataset, 'test', network, print_output=os.path.join(model_save_dir, 'dump_test'))
    log(f, 'TEST epoch: ' + str(epoch) +
        ' test loss1, mae:' + str(epoch_test_losses) + ", " + str(mae))
    log(f, 'Exiting train...')
    f.close()
    return


if __name__ == '__main__':
    args = parser.parse_args()
    # -- Assign GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # -- Assertions
    assert (args.dataset)

    # -- Check if requirements satisfied
    assert(np.__version__=="1.15.4")
    assert(cv2.__version__=="3.4.3")
    assert(torch.__version__=="0.4.1")
    assert(powerlaw.__version__=="1.4.4")
    assert("9.0" in torch.version.cuda)
    
    # -- Setting seeds for reproducability
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # -- Dataset paths
    if args.dataset == "parta":
        validation_set = 30
        output_downscale = 8
        blur_sigma = 2
        path = '../../dataset/ST_partA/'
    elif args.dataset == "ucfqnrf":
        validation_set = 240
        output_downscale = 8
        blur_sigma = 2
        args.dense_weight = 0.01
        path = "../../dataset/UCF-QNRF_ECCV18"

    model_save_dir = './models'

    batch_size = args.batch_size

    dataset = CrowdDataset(path, name=args.dataset, valid_set_size=validation_set,
                           gt_downscale_factor=output_downscale)

    print(dataset.data_files['test_valid'],
          len(dataset.data_files['test_valid']))
    print(dataset.data_files['train'], len(dataset.data_files['train']))

    # -- Train the model
    train_network()
