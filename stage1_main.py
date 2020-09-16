"""
stage1_main.py: stage 1 training script
"""

import argparse
import datetime
import os
import pickle
import random

import cv2
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn as nn
from torch import optim as optim

from crowd_dataset import CrowdDataset
from models import Stage1CountingNet

matplotlib.use('Agg')

rotation_angles = [0, 90, 180, 270]
rotation_angles_cv2 = [0, cv2.ROTATE_90_COUNTERCLOCKWISE,
                       cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE]
num_rotations = len(rotation_angles)
image_new_crop_size = 112

parser = argparse.ArgumentParser(description='CSS-CCNN Stage-1 Training')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--gpu', default=0, type=int,
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

def train_function(Xs, Ys, network, optimizer):
    network = network.cuda()
    optimizer.zero_grad()

    X = torch.autograd.Variable(torch.from_numpy(Xs)).cuda()
    Y = torch.autograd.Variable(torch.LongTensor(Ys)).cuda()

    outputs = network(X)
    assert(outputs.shape == (X.shape[0], num_rotations))  # (B,4)
    losses = []

    loss_criterion = nn.CrossEntropyLoss(size_average=True)
    loss_ = loss_criterion(outputs, Y)
    loss = loss_
    assert(loss.grad_fn != None)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    matches, actual_angle_dist, matches_by_angle = calculate_per_rot_acc(
        outputs, Y)
    return losses, matches, actual_angle_dist, matches_by_angle


@torch.no_grad()
def test_function(X, Y, network):
    X = torch.autograd.Variable(torch.from_numpy(X)).cuda()
    Y = torch.autograd.Variable(torch.from_numpy(Y)).cuda().long()

    network = network.cuda()
    network.eval()
    output = network(X)

    loss_criterion = nn.CrossEntropyLoss(size_average=True)
    loss_ = loss_criterion(output, Y)
    loss = loss_

    matches, actual_angle_dist, matches_by_angle = calculate_per_rot_acc(
        output, Y)

    network.train()
    return loss.data, matches, actual_angle_dist, matches_by_angle


def calculate_per_rot_acc(rotation_prediction, rotation_gt):
    out_argmax = torch.argmax(nn.functional.softmax(
        rotation_prediction, dim=1), dim=1)  # (B,)
    Yss_argmax = rotation_gt  # (B,)

    equat_mat = out_argmax == Yss_argmax
    matches = torch.sum(out_argmax == Yss_argmax).item()

    actual_angle_dist = np.array([torch.sum(Yss_argmax == rot_idx).item(
    ) for rot_idx in range(num_rotations)])  # len of n
    matches_by_angle = np.array([torch.sum(
        equat_mat[out_argmax == rot_idx]).item() for rot_idx in range(num_rotations)])

    assert(np.sum(matches_by_angle) == matches)
    return matches, actual_angle_dist, matches_by_angle


def test_network(dataset, set_name, network, print_output=False):
    global test_loss
    global counter
    test_loss = 0.
    counter = 0.
    metrics_test = {}
    metrics_ = ['new_mae', 'mle', 'mse', 'loss1']
    for k in metrics_:
        metrics_test[k] = 0.0

    if isinstance(print_output, str):
        print_path = print_output
    elif isinstance(print_output, bool) and print_output:
        print_path = './models/dump'
    else:
        print_path = None

    total_matches_count = 0
    total_per_angle_count = np.zeros(num_rotations)
    total_per_angle_match_count = np.zeros(num_rotations)

    for idx, data in enumerate(dataset.test_get_data(set_name)):
        image_name, Xs, _ = data
        image = Xs[0].transpose((1, 2, 0))

        # 1. Crop out the 112x112 image, Xs[0] (3,h,w)
        image_h, image_w = Xs[0].shape[-2:]
        image_center = np.array([image_h // 2, image_w // 2])
        image_crop_start_loc = image_center - (image_new_crop_size//2)
        image_crop_start_loc[image_crop_start_loc < 0] = 0
        assert(image_h >= image_new_crop_size and image_w >= image_new_crop_size)
        cropped_Xs = Xs[0][:, image_crop_start_loc[0]: image_crop_start_loc[0] + image_new_crop_size,
                           image_crop_start_loc[1]: image_crop_start_loc[1] + image_new_crop_size]  # (3,h',w')
        assert(cropped_Xs.shape == (3, image_new_crop_size, image_new_crop_size))

        # 2. Do all the rotations for image and form the batch of rotation
        new_images_input = np.zeros(
            (num_rotations,) + cropped_Xs.shape, dtype=Xs.dtype)  # (num_rotations,3,h',w')
        new_image_rotation_gt = np.zeros(
            (num_rotations, ), dtype=np.int32)  # (B, )
        cropped_image = np.transpose(cropped_Xs, (1, 2, 0))  # (h',w',3)
        for i in range(num_rotations):
            rot_cropped_image = cropped_image.copy()
            if i != 0:
                rot_cropped_image = cv2.rotate(
                    rot_cropped_image, rotation_angles_cv2[i])
            new_images_input[i] = np.transpose(rot_cropped_image, (2, 0, 1))
            assert (np.sum(cropped_Xs) == np.sum(rot_cropped_image))
            new_image_rotation_gt[i] = i

        assert(new_images_input.shape == (num_rotations, 3,
                                          image_new_crop_size, image_new_crop_size))

        loss, num_matches,  actual_angle_dist, matches_by_angle = test_function(new_images_input, new_image_rotation_gt,
                                                                                network)
        total_matches_count += num_matches
        total_per_angle_count += actual_angle_dist
        total_per_angle_match_count += matches_by_angle

        test_loss += loss
        counter += 1

    rotation_match_acc = total_matches_count/(counter * num_rotations)
    per_rot_match_acc = total_per_angle_match_count/total_per_angle_count

    assert (np.sum(total_per_angle_count) == (counter * num_rotations))

    metrics_test['loss1'] = test_loss / float(counter)
    txt = ''
    txt += '%s: %s ' % ('loss1', metrics_test['loss1'])

    return metrics_test, txt, rotation_match_acc, per_rot_match_acc


def train_network():
    network = Stage1CountingNet()
    model_save_dir = './models_stage_1'
    model_save_path = os.path.join(model_save_dir, 'train2')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        os.makedirs(os.path.join(model_save_path, 'snapshots'))
    global f
    snapshot_path = os.path.join(model_save_path, 'snapshots')
    f = open(os.path.join(model_save_path, 'train0.log'), 'w')

    # -- Logging Parameters
    log(f, 'args: ' + str(args))
    log(f, 'model: ' + str(network), False)
    log(f, 'Stage1..')
    log(f, 'LR: %.12f.' % (args.lr))

    start_epoch = 0
    num_epochs = args.epochs
    valid_losses = {}
    train_losses = {}
    for metric in ['loss1', 'new_mae']:
        valid_losses[metric] = []

    for metric in ['loss1']:
        train_losses[metric] = []

    batch_size = args.batch_size
    num_train_images = len(dataset.data_files['train'])
    num_patches_per_image = args.patches
    num_batches_per_epoch = num_patches_per_image * num_train_images // batch_size

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, network.parameters()),
                          lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # -- Main Training Loop
    all_epoch_test_valid_accs = []
    all_epoch_test_valid_per_rot_accs = []
    for e_i, epoch in enumerate(range(start_epoch, num_epochs)):
        avg_loss = [0.0 for _ in range(1)]

        # b_i - batch index
        total_match_count = 0
        total_count = 0
        total_per_angle_count = np.zeros(num_rotations)
        total_per_angle_match_count = np.zeros(num_rotations)
        for b_i in range(num_batches_per_epoch):
            # Generate next training sample
            Xs, _ = dataset.train_get_data(batch_size=args.batch_size)

            # 1. Crop image to 112x112 . Xs shape: (B,3,h,w)
            image_size = Xs.shape[-1]
            crop_start_loc = [image_size // 4, image_size // 4]

            Xs = Xs[:, :, crop_start_loc[0]: crop_start_loc[0] + image_new_crop_size,
                    crop_start_loc[1]: crop_start_loc[1] + image_new_crop_size]

            # 2 . Randomly rotate each image
            new_images_input = np.zeros_like(Xs, dtype=Xs.dtype)  # (B,3,h',w')
            new_image_rotation_gt = np.zeros(
                (Xs.shape[0], ), dtype=np.int32)  # (B,4)
            images = np.transpose(Xs, (0, 2, 3, 1))  # (B,h',w',3)
            for i in range(images.shape[0]):
                image = images[i]  # (h',w',3)
                chosen_index = np.random.choice(num_rotations, 1)[0]
                chosen_angle = rotation_angles[chosen_index]
                if chosen_angle != 0:
                    image = cv2.rotate(
                        image, rotation_angles_cv2[chosen_index])
                new_images_input[i, :, :, :] = np.transpose(image, (2, 0, 1))
                new_image_rotation_gt[i] = chosen_index

            losses, matches, actual_angle_dist, matches_by_angle = train_function(new_images_input,
                                                                                  new_image_rotation_gt,
                                                                                  network, optimizer)
            total_match_count += matches
            total_count += args.batch_size
            assert(total_match_count <= total_count)

            total_per_angle_count += actual_angle_dist
            total_per_angle_match_count += matches_by_angle

            assert(np.sum(total_per_angle_count) == total_count)
            for scale_idx in range(1):
                avg_loss[scale_idx] = avg_loss[scale_idx] + losses[scale_idx]

            # Logging losses after 1k iterations.
            if b_i % 100 == 0:
                log(f, 'Epoch %d [%d]: %s loss: %s.' %
                    (epoch, b_i, [network.name], losses))
                log(f, 'Epoch %d [%d]: %s rot acc: %s.' % (
                    epoch, b_i, [network.name], (total_match_count/total_count)))
                log(f, 'Epoch %d [%d]: %s rot acc(0,90,180,270): %s.' % (epoch, b_i, [network.name],
                                                                         (total_per_angle_match_count / total_per_angle_count)))

        # -- Stats update
        avg_loss = [al / num_batches_per_epoch for al in avg_loss]
        avg_loss = [av for av in avg_loss]

        train_losses['loss1'].append(avg_loss)

        torch.cuda.empty_cache()
        log(f, 'Validating...')

        epoch_val_losses, txt, rot_acc_valid, per_rot_acc_valid = test_network(
            dataset, 'test_valid', network, False)
        log(f, 'Valid epoch: ' + str(epoch) + ' ' + txt)
        log(f, 'Valid epoch: ' + str(epoch) +
            'total rotation acc:' + str(rot_acc_valid))
        log(f, 'Valid epoch: ' + str(epoch) +
            'per rotation acc:' + str(per_rot_acc_valid))
        all_epoch_test_valid_accs.append(rot_acc_valid)
        all_epoch_test_valid_per_rot_accs.append(per_rot_acc_valid)

        best_epoch = np.argmax(np.array(all_epoch_test_valid_accs))
        best_valid_test_acc = np.array(all_epoch_test_valid_accs).max()
        log(f, 'Best valid rot acc so far epoch : {} , acc : {}'.format(
            best_epoch, best_valid_test_acc))

        for metric in ['loss1', 'new_mae']:
            valid_losses[metric].append(epoch_val_losses[metric])

        min_valid_epoch = np.argmin(valid_losses['new_mae'])

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

    all_epoch_test_valid_accs = np.array(all_epoch_test_valid_accs)
    best_epoch = np.argmax(all_epoch_test_valid_accs)
    best_valid_test_acc = all_epoch_test_valid_accs.max()

    log(f, 'Best valid rot acc epoch : {} , acc : {}'.format(
        best_epoch, best_valid_test_acc))

    # Plotting the valid accuracies
    plt.plot(np.array(all_epoch_test_valid_accs))
    for i in range(num_rotations):
        plt.plot(np.array(all_epoch_test_valid_per_rot_accs)[:, i])
    plt.legend(['overall acc', '0 deg acc', '90 deg acc',
                '180 deg acc', '270 deg acc'], loc='upper right')
    plt.savefig(os.path.join(snapshot_path, 'test_valid_all_rot_acc.png'))
    plt.clf()
    plt.close()

    # this is to be consistent with the file name written
    filename = get_filename(network.name, best_epoch + 1)
    with open(os.path.join(snapshot_path, 'unsup_vgg_best_model_meta.pkl'), 'wb') as unsup_file:
        pickle.dump(filename, unsup_file, protocol=2)
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
    assert("9.0" in torch.version.cuda)
    
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
        path = "../../dataset/ST_partA/"
        output_downscale = 4
        dataset = CrowdDataset(path, name=args.dataset, valid_set_size=validation_set,
                           gt_downscale_factor=output_downscale, stage_1=True)
    elif args.dataset == "ucfqnrf":
        validation_set = 240
        output_downscale = 4
        path = "../../dataset/UCF-QNRF_ECCV18"
        dataset = CrowdDataset(path, name=args.dataset, valid_set_size=validation_set,
                           gt_downscale_factor=output_downscale, stage_1=True, image_size_max=768)

    model_save_dir = './models'
    batch_size = args.batch_size

    print(dataset.data_files['test_valid'],
          len(dataset.data_files['test_valid']))
    print(dataset.data_files['train'], len(dataset.data_files['train']))

    # -- Train the model
    train_network()
