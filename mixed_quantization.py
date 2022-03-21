# Mixed Quantization (Fusion Quantization)
# Step1: Acquire the codebook and index matrix using PQ
# Step2: Insert presudo quantize unit and finetune the network recovered from codebook and indexmatrix.
# Step3: Finetune the whole network.
# 
# Author: Mingyuan Liu

import os
import sys
import time
import math
import argparse
from operator import attrgetter
from bisect import bisect_left

import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn

import models
from data import load_dataset
from pq import CentroidSGD
from pq import PQ
from utils.training import finetune_centroids, evaluate
from utils.watcher import ActivationWatcher
from utils.dynamic_sampling import dynamic_sampling
from utils.statistics import compute_size
from utils.utils import centroids_from_weights, weight_from_centroids
from distillation_data.distillation_data import getDistilData

from qat import prepare


def parse_args():
    parser = argparse.ArgumentParser(description='Mixed Quantization: Combine scalar quantization with product quantization.')

    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'resnet50', 'mobilenetv2', 
                                                            'shufflenetv2', 'efficientnet_b0', 'efficientnet_b3',
                                                            'efficientnet_b7', 'densenet-201'],
                        help='Pretrained model to quantize')
    
    # Parameters for product quantization.
    # 
    parser.add_argument('--except_blocks', default='no', type=str,
                        help='Excepted block to quantize (if no, quantizes whole network)')

    parser.add_argument('--n-iter', default=100, type=int,
                        help='Number of EM iterations for quantization')
    parser.add_argument('--n-activations', default=512, type=int,
                        help='Size of the batch of activations to sample from')

    parser.add_argument('--block-size-cv', default=9, type=int,
                        help='Quantization block size for 3x3 standard convolutions')
    parser.add_argument('--block-size-pw', default=4, type=int,
                        help='Quantization block size for 1x1 convolutions')
    parser.add_argument('--block-size-fc', default=4, type=int,
                        help='Quantization block size for fully-connected layers')
    
    parser.add_argument('--n-centroids-cv', default=256, type=int,
                        help='Number of centroids')
    parser.add_argument('--n-centroids-pw', default=128, type=int,
                        help='Number of centroids for pointwise convolutions')
    parser.add_argument('--n-centroids-fc', default=2048, type=int,
                        help='Number of centroids for classifier')
    parser.add_argument('--n-centroids-threshold', default=4, type=int,
                        help='Threshold for reducing the number of centroids') #  we clamp the number of centroids to min(k, Cout × m/4) for stability.
    parser.add_argument('--eps', default=1e-8, type=float,
                        help='For empty cluster resolution') # resolve the empty clusters

    # data relevant parameters
    parser.add_argument('--data-type', default='imagenet', type=str,
                        help='Dataset used by quantization, imagenet or cifar10')
    parser.add_argument('--data-path', default='dataset/imagenet', type=str,
                        help='Path to ImageNet dataset')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size for fiuetuning steps')
    parser.add_argument('--n-workers', default=20, type=int,
                        help='Number of workers for data loading')
    # whether distillation data used
    parser.add_argument('--use-distillation-data', action='store_true',
                        help='Is distillation data used.')

    # parameters for findtuning the centroids
    parser.add_argument('--finetune-epochs', default=2, type=int,
                        help='Number of iterations for layer-wise finetuning of the centroids')   
    parser.add_argument('--finetune-centroids', default=5000, type=int,
                        help='Number of iterations for layer-wise finetuning of the centroids')
    parser.add_argument('--lr-centroids', default=0.01, type=float,
                        help='Learning rate to finetune centroids')
    parser.add_argument('--momentum-centroids', default=0.9, type=float,
                        help='Momentum when using SGD')
    parser.add_argument('--weight-decay-centroids', default=1e-4, type=float,
                        help='Weight decay')

    parser.add_argument('--finetune-whole', default=10000, type=int,
                        help='Number of iterations for global finetuning of the centroids')
    parser.add_argument('--lr-whole', default=0.01, type=float,
                        help='Learning rate to finetune classifier')
    parser.add_argument('--momentum-whole', default=0.9, type=float,
                        help='Momentum when using SGD')
    parser.add_argument('--weight-decay-whole', default=1e-4, type=float,
                        help='Weight decay')
    parser.add_argument('--finetune-whole-epochs', default=9, type=int,
                        help='Number of epochs for global finetuning of the centroids')
    parser.add_argument('--finetune-whole-step-size', default=3, type=int,
                        help='Learning rate schedule for global finetuning of the centroids')

    parser.add_argument('--restart', default='', type=str,
                        help='Already stored centroids')
    parser.add_argument('--save', default='./', type=str,
                        help='Path to save the finetuned models')
    
    # parameters for error correction(QAT).
    # 
    parser.add_argument('--bn-fuse', action='store_true',
                        help='batch-normalization fuse')
    parser.add_argument('--bn-fuse-calib', action='store_true',
                        help='batch-normalization fuse calibration')
    parser.add_argument('--q-type', type=int, default=0,
                        help='quant_type:0-symmetric, 1-asymmetric')
    parser.add_argument('--q-level', type=int, default=0,
                        help='quant_level:0-per_channel, 1-per_layer')
    parser.add_argument('--weight-observer', type=int, default=0,
                        help='quant_weight_observer:0-MinMaxObserver, 1-MovingAverageMinMaxObserver')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.except_blocks = '' if args.except_blocks == 'no' else args.except_blocks
    # student model to quantize
    student = models.__dict__[args.model](pretrained=True).cuda()
    student.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    print(' ===> prepare dataset ')
    if not args.use_distillation_data:
        # load data
        train_loader, test_loader = load_dataset(type=args.data_type, data_path=args.data_path, 
                                                batch_size=args.batch_size, nb_workers=args.n_workers)
    else:
        print(' ======> use distillation data ')
        print(' ======> start generating distillation data')
        _, test_loader = load_dataset(type=args.data_type, data_path=args.data_path, 
                                                batch_size=args.batch_size, nb_workers=args.n_workers)
        train_loader = getDistilData(student.cuda(), args.data_type, batch_size=args.batch_size, 
                                    num_batch=1, num_workers=args.n_workers, for_inception=args.model.startswith('inception'))


    # layers to quantize 
    watcher = ActivationWatcher(student)
    # exclude do not quantized layers
    if 'resnet' in args.model:
        layers = [layer for layer in watcher.layers[1:] if args.except_blocks in layer ]
    elif 'mobilenet' in args.model:
        layers = [layer for layer in watcher.layers if args.except_blocks in layer]
    elif 'efficientnet' in args.model:
        layers = [layer for layer in watcher.layers if args.except_blocks in layer and not 'se' in layer]
        
    print(' ===> layers: ', layers)

    # teacher model
    teacher = models.__dict__[args.model](pretrained=True).cuda()
    teacher.eval()

    # parameters for the centroids optimizer
    opt_centroids_params_all = []

    # book-keeping for compression statistics (in MB)
    size_uncompressed = compute_size(student)
    size_index = 0
    size_centroids = 0
    size_other = size_uncompressed

    t = time.time()
    top_1 = 0
    pesudo_quantization_layers = []
    for layer in layers:
        # print information
        print(' ===> Start Quantize layer: {}'.format(layer))
        pesudo_quantization_layers.append(layer)

        # Step 1: iteratively quantize the network layers
        print(' ===> Step 1: Quantize network using product quantization')

        #  gather input activations
        n_iter_activations = math.ceil(args.n_activations / args.batch_size)
        watcher = ActivationWatcher(student, layer=layer)
        in_activations_current = watcher.watch(train_loader, criterion, n_iter_activations)
        in_activations_current = in_activations_current[layer]
        # get weight matrix and detach it from the computation graph (.data should be enough, adding .detach() as a safeguard)
        M = attrgetter(layer + '.weight.data')(student).detach()
        sizes = M.size()
        is_conv = len(sizes) == 4

        # get padding and stride attributes
        padding = attrgetter(layer)(student).padding if is_conv else 0
        stride = attrgetter(layer)(student).stride if is_conv else 1
        groups = attrgetter(layer)(student).groups if is_conv else 1

        # block size, distinguish between fully connected and convolutional case
        if is_conv:
            out_features, in_features, k, _ = sizes
            block_size = args.block_size_cv if k > 1 else args.block_size_pw
            n_centroids = args.n_centroids_cv if k > 1 else args.n_centroids_pw
            n_blocks = in_features * k * k // block_size
        else:
            k = 1
            out_features, in_features = sizes
            block_size = args.block_size_fc
            n_centroids = args.n_centroids_fc
            n_blocks = in_features // block_size

        # clamp number of centroids for stability
        powers = 2 ** np.arange(0, 16, 1) # [1 ~ 65536]
        n_vectors = np.prod(sizes) / block_size # the total number of subvector, eg: layer1.conv1(64*64*3*3) ==> 64*64*3*3/9=4096
        idx_power = bisect_left(powers, n_vectors / args.n_centroids_threshold) # 4096/4=2^10
        n_centroids = min(n_centroids, powers[idx_power - 1]) # clip centriods to 2^9
        
        # compression rations
        bits_per_weight = np.log2(n_centroids) / block_size

        # number of bits per weight
        size_index_layer = bits_per_weight * M.numel() / 8 / 1024 / 1024
        size_index += size_index_layer

        # centroids stored in float16
        size_centroids_layer = n_centroids * block_size * 2 / 1024 / 1024 / 4
        size_centroids += size_centroids_layer

        # size of non-compressed layers, e.g. BatchNorms or first 7x7 convolution
        size_uncompressed_layer = M.numel() * 4 / 1024 / 1024
        size_other -= size_uncompressed_layer

        compressed_size = size_index_layer + size_centroids_layer
        # print layer size 
        print('Quantizing layer: {}, size: {}, n_blocks: {}, block size: {}, ' \
              'centroids: {}, bits/weight: {:.2f}, compressed size: {:.2f} MB, compress ratio: {:.2f}'.format(
               layer, list(sizes), n_blocks, block_size, n_centroids,
               bits_per_weight,compressed_size, compressed_size/size_uncompressed))

        
        # number of activation samples
        n_samples = dynamic_sampling(layer)

        # product quantization
        quantizer = PQ(in_activations_current, M, n_activations=args.n_activations,
                       n_samples=n_samples, eps=args.eps, n_centroids=n_centroids,
                       n_iter=args.n_iter, n_blocks=n_blocks, k=k,
                       stride=stride, padding=padding, groups=groups)
        if len(args.restart) > 0:
            # do not quantize already quantized layers
            try:
                # load centroids and assignments if already stored
                quantizer.load(args.restart, layer)
                centroids = quantizer.centroids
                assignments = quantizer.assignments

                # quantize weight matrix
                M_hat = weight_from_centroids(centroids, assignments, n_blocks, k, is_conv)
                attrgetter(layer + '.weight')(student).data = M_hat
                quantizer.save(args.save, layer)

                # optimizer for global finetuning
                prepare(student, 
                    pesudo_quantization_layers, 
                    inplace=True, 
                    a_bits=8, 
                    w_bits=8, 
                    q_type=args.q_type,
                    q_level=args.q_level,
                    weight_observer=args.weight_observer,
                    bn_fuse=args.bn_fuse,
                    pretrained_model=True,
                    qaft=False,
                    ptq=False)
                parameters = [p for (n, p) in student.named_parameters() if layer in n and 'bias' not in n]
                centroids_params = {'params': parameters,
                                    'assignments': assignments,
                                    'kernel_size': k,
                                    'n_centroids': n_centroids,
                                    'n_blocks': n_blocks}
                opt_centroids_params_all.append(centroids_params)

                # proceed to next layer
                print('Layer already quantized, proceeding to next layer\n')
                continue

            # otherwise, quantize layer
            except FileNotFoundError:
                print('Quantizing layer')
            # quantize layer
        quantizer.encode() 

        # assign quantized weight matrix
        M_hat = quantizer.decode()
        attrgetter(layer + '.weight')(student).data = M_hat

        # top1
        top_1 = evaluate(test_loader, student, criterion).item()

        # book-keeping
        print('Quantizing time: {:.0f}min, Top1 after quantization: {:.2f}\n'.format((time.time() - t) / 60, top_1))
        t = time.time()

        print(' ===> Step2: Finetune the centroids and do error correction using QAT')
        # place pesudo quantization unit 
        prepare(student, 
                pesudo_quantization_layers, 
                inplace=True, 
                a_bits=8, 
                w_bits=8, 
                q_type=args.q_type,
                q_level=args.q_level,
                weight_observer=args.weight_observer,
                bn_fuse=args.bn_fuse,
                pretrained_model=True,
                qaft=False,
                ptq=False)
        # optimizer for centroids
        parameters = [p for (n, p) in student.named_parameters() if layer in n and 'bias' not in n]
        assignments = quantizer.assignments
        centroids_params = {'params': parameters,
                            'assignments': assignments,
                            'kernel_size': k,
                            'n_centroids': n_centroids,
                            'n_blocks': n_blocks}
 
        # remember centroids parameters to finetuning at the end
        opt_centroids_params = [centroids_params]
        opt_centroids_params_all.append(centroids_params)

        # custom optimizer
        optimizer_centroids = CentroidSGD(opt_centroids_params, lr=args.lr_centroids,
                                          momentum=args.momentum_centroids,
                                          weight_decay=args.weight_decay_centroids)
        # standard training loop
        n_iter = args.finetune_centroids
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_centroids, step_size=5, gamma=0.1)
        epochs = args.finetune_epochs
        # print(student.state_dict())

        # start training 
        print('===> Start finetune ... ')
        for epoch in range(epochs):
            finetune_centroids(train_loader, student, teacher, criterion, optimizer_centroids, n_iter=n_iter)
            top1 = evaluate(test_loader, student, criterion)
            scheduler.step()
            print('Epoch: {}, Top1: {}'.format(epoch, top1))

        print('After finetune, cost time: {}min, iterations with learnrate: {} and top1: {}'.format((time.time() - t) / 60, args.lr_centroids, top1))
        
        t = time.time()

        M_hat = attrgetter(layer + '.weight')(student).data
        centroids = centroids_from_weights(M_hat, assignments, n_centroids, n_blocks)
        quantizer.centroids = centroids
        quantizer.save(args.save, layer)
    
    size_compressed = size_index + size_centroids + size_other
    print('End of compression, non-compressed teacher model: {:.2f}MB, compressed student model ' \
          '(indexing + centroids + other): {:.2f}MB + {:.2f}MB + {:.2f}MB = {:.2f}MB, compression ratio: {:.2f}x\n'.format(
          size_uncompressed, size_index, size_centroids, size_other, size_compressed, size_uncompressed / size_compressed))

    print('===> finetune the whole network')
    optimizer_centroids_all = CentroidSGD(opt_centroids_params_all, lr=args.lr_whole,
                                          momentum=args.momentum_whole,
                                          weight_decay=args.weight_decay_whole)
    n_iter = args.finetune_whole
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_centroids_all, step_size=args.finetune_whole_step_size, gamma=0.1)
    for epoch in range(args.finetune_whole_epochs):
        student.train()
        finetune_centroids(train_loader, student, teacher, criterion, optimizer_centroids_all, n_iter=n_iter)
        top1 = evaluate(test_loader, student, criterion)
        scheduler.step()
        print('Epoch:{}, Top1:{}'.format(epoch, top1))
    
    # save quantization model
    compressed_model_state_dict = {}
    # copy the first layer's weights
    compressed_model_state_dict['conv1'] = student.conv1.state_dict()
    compressed_model_state_dict['fc_bias'] = {'bias': student.fc.bias}

    # save bn layers
    bn_layers = watcher._get_bn_layers()

    for bn_layer in bn_layers:
        compressed_model_state_dict[bn_layer] = attrgetter(bn_layer)(student).state_dict()

    # save layers
    # stats
    for layer in layers:
        M = attrgetter(layer + '.weight.data')(student).detach()
        sizes = M.size()
        is_conv = len(sizes) == 4
        # get padding and stride attributes
        padding = attrgetter(layer)(student).padding if is_conv else 0
        stride = attrgetter(layer)(student).stride if is_conv else 1
        groups = attrgetter(layer)(student).groups if is_conv else 1
        # block size, distinguish between fully connected and convolutional case
        if is_conv:
            out_features, in_features, k, _ = sizes
            block_size = args.block_size_cv if k > 1 else args.block_size_pw
            n_centroids = args.n_centroids_cv if k > 1 else args.n_centroids_pw
            n_blocks = in_features * k * k // block_size
        else:
            k = 1
            out_features, in_features = sizes
            block_size = args.block_size_fc
            n_centroids = args.n_centroids_fc
            n_blocks = in_features // block_size
        
        # clamp number of centroids for stability
        powers = 2 ** np.arange(0, 16, 1)
        n_vectors = np.prod(sizes) / block_size
        idx_power = bisect_left(powers, n_vectors / args.n_centroids_threshold)
        n_centroids = min(n_centroids, powers[idx_power - 1])
        
        pesudo_layer_weight_scale = attrgetter(layer + '.weight_quantizer')(student).scale
        pesudo_layer_weight_zero_point = attrgetter(layer + '.weight_quantizer')(student).zero_point
        layer_activation_scale = attrgetter(layer + '.activation_quantizer')(student).scale
        layer_activation_zero_point = attrgetter(layer + '.activation_quantizer')(student).zero_point

        quantizer.load(args.save, layer)
        assignments = quantizer.assignments
        M_hat = attrgetter(layer + '.weight')(student).data
        centroids = centroids_from_weights(M_hat, assignments, n_centroids, n_blocks)
        quantizer.centroids = centroids
        quantizer.save(args.save, layer)

        # quantize centroids
        
        # define closure
        def quant_centroids():
            # centroids_int = torch.zeros(centroids.size(), dtype=torch.short)
            # Symmetric quantization
            if args.q_type == 0:
                centroids_int = torch.clamp(torch.round(centroids / pesudo_layer_weight_scale), -128, 127).short()
            # Asymmetric quantization
            else:
                centroids_int = torch.clamp(torch.round(centroids / pesudo_layer_weight_scale) + pesudo_layer_weight_zero_point, 0, 255).short()

            return centroids_int
        
        centroids_int = quant_centroids()
        compressed_layer_state_dict = {
                'centroids': centroids_int,
                'assignments': assignments.short() if 'fc' in layer else assignments.byte(),
                'n_blocks': n_blocks,
                'is_conv': is_conv,
                'k': k,
                'w_scale': pesudo_layer_weight_scale,
                'w_zero_point': pesudo_layer_weight_zero_point,
                'act_scale': layer_activation_scale,
                'act_zero_point': layer_activation_zero_point
        }
        compressed_model_state_dict[layer] = compressed_layer_state_dict

    torch.save(compressed_model_state_dict, os.path.join(args.save, 'compressed_'+args.model+'.pth'))
    print('Finetune the whole network, cost time:{} min, Top1 after finetune centroids: {}\n'.format((time.time() - t) / 60, top1))
       

if __name__ == '__main__':
    main()
        
