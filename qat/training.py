import os
import random
import sys
sys.path.append("../")
import models

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import torchvision

import time
import copy
import argparse
import numpy as np
import qat.quantization as quantization

from progress.bar import Bar
from operator import attrgetter
from data import load_dataset
from models import resnet18

def set_random_seed(random_seed = 0):
    """
    Set torch.random and random package's random seed.

    Args:
        random_seed -> int: seed 
    """
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def evaluate_model(model, test_loader, device, criterion=None):
    """
    Evaluate the model using test dataset.

    Args:
        model -> nn.Module
        test_loader -> torch.datasets
        device: cpu or cuda
        criterion: loss func
    """
    model.eval()
    # model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = torch.true_divide(running_loss, len(test_loader.dataset))
    eval_accuracy = torch.true_divide(running_corrects, len(test_loader.dataset))

    return eval_loss, eval_accuracy


def train_model(model,
                model_name,
                train_loader,
                test_loader,
                device,
                learning_rate=1e-1,
                num_epochs=200):
    """
    Train a model.

    Args:
        model -> nn.Module
        train_loader -> train dataset
        test_loader -> test dataset
        device -> cpu or cuda
    """
    # The training configurations were not carefully selected.
    best_eval_acc = 0

    criterion = nn.CrossEntropyLoss().cuda()
    
    # model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=0.9,
                          weight_decay=1e-4)
    # Cosine scheduler or MultiStep scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[100, 150],
                                                     gamma=0.1,
                                                     last_epoch=-1)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Evaluation
    model.eval()
    eval_loss, eval_accuracy = evaluate_model(model=model,
                                              test_loader=test_loader,
                                              device=device,
                                              criterion=criterion)
    print("Epoch: {:03d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
        0, eval_loss, eval_accuracy))

    for epoch in range(num_epochs):
        TrainingBar = Bar('Training', max=len(train_loader))
        # Training
        model.train()

        running_loss = 0
        running_corrects = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            acc = 100. * running_corrects / total
            show_loss = running_loss / (batch_idx + 1)

            TrainingBar.suffix = f'({batch_idx + 1}/{len(train_loader)}) | ETA: {TrainingBar.eta_td} | Loss: {show_loss} | top1: {acc}'
            TrainingBar.next()

        train_loss = torch.true_divide(running_loss, len(train_loader.dataset))
        train_accuracy = torch.true_divide(running_corrects, len(train_loader.dataset))

        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(model=model,
                                                  test_loader=test_loader,
                                                  device=device,
                                                  criterion=criterion)

        # Save Best model
        if eval_accuracy > best_eval_acc:
            torch.save(model, 'checkpoints/{}_acc{:.3f}.pth'.format(model_name, best_eval_acc))
            best_eval_acc = eval_accuracy
        # Set learning rate scheduler
        scheduler.step()
        TrainingBar.finish()

        print(
            "Epoch: {:03d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}"
            .format(epoch + 1, train_loss, train_accuracy, eval_loss,
                    eval_accuracy))

    return model

def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', action='store', default='cuda:0',
                        help='device cpu or cuda')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='the intial learning rate')
    parser.add_argument('--wd', action='store', default=1e-5,
                        help='moment')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='the path to the resume model')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--eval_batch_size', type=int, default=48)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train_start')
    # W/A — bits
    parser.add_argument('--w_bits', type=int, default=8)
    parser.add_argument('--a_bits', type=int, default=8)
    # bn融合标志位
    parser.add_argument('--bn_fuse', action='store_true',
                        help='batch-normalization fuse')
    # bn融合校准标志位
    parser.add_argument('--bn_fuse_calib', action='store_true',
                        help='batch-normalization fuse calibration')
    # 量化方法选择
    parser.add_argument('--q_type', type=int, default=0,
                        help='quant_type:0-symmetric, 1-asymmetric')
    # 量化级别选择
    parser.add_argument('--q_level', type=int, default=0,
                        help='quant_level:0-per_channel, 1-per_layer')
    # weight_observer选择
    parser.add_argument('--weight_observer', type=int, default=0,
                        help='quant_weight_observer:0-MinMaxObserver, 1-MovingAverageMinMaxObserver')
    # qaft标志位
    parser.add_argument('--qaft', action='store_true',
                        help='quantization-aware-finetune')
    # ptq_percentile
    parser.add_argument('--percentile', type=float, default=0.999999,
                        help='the percentile of ptq')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='model name to be quantized.')
    # pretrained_model标志位
    parser.add_argument('--pretrained_model', action='store_true',
                        help='pretrained model.') 
    # cifar10 or imagenet
    parser.add_argument('--data_type', type=str, default='cifar10',
                        help='dataset type, cifar10 or imagenet')
    parser.add_argument('--data_path', type=str, default='../dataset/cifar10',
                        help='dataset path.')                                    
    args = parser.parse_args()

    return args


TestModelsDict = {'resnet18': resnet18}
TestModelsPretrainedPath = {}

def main():
    """
    Quantization Aware Training.
    """
    set_random_seed(0)

    args = parse_agrs()
    print('==> Options:', args)
    
    print('==> Prepare dataset ..')
    train_loader, test_loader = load_dataset(args.data_type, data_path=args.data_path,
                                             batch_size=args.batch_size, nb_workers=args.num_workers)
    
    print('==> load model ..')
    if args.data_type == 'cifar10':
        num_classes = 10
    elif args.data_type == 'imagenet':
        num_classes = 1000
    model = TestModelsDict[args.model](pretrained=args.pretrained_model, num_classes=num_classes)
    # initialize parameters
    if not args.pretrained_model:
        print('Initialize model parameters.')
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)

    # get all layers
    keys = model.state_dict().keys()
    layers = [k[:k.rfind(".")] for k in keys if 'bias' not in k]
    # remove BN layers
    layers = [layer for layer in layers if not isinstance(attrgetter(layer)(model), nn.BatchNorm2d)]
    # remove fc
    # for i in range(3):
    #    layers.pop()

    # if 'cuda' in args.device:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print('layers --> ', layers)
    print('\n original model --> ', model)
    print('==> Start insert fake quantizer ...')
    quantization.prepare(model, 
                        layers, 
                        inplace=True, 
                        a_bits=args.a_bits,
                        w_bits=args.w_bits,
                        q_type=args.q_type,
                        q_level=args.q_level,
                        weight_observer=args.weight_observer,
                        bn_fuse=args.bn_fuse,
                        pretrained_model=args.pretrained_model,
                        qaft=False,
                        ptq=False,
                        percentile=args.percentile)

    print('After quantize, quant model --> ', model)

    print('==> Start training ...')
    train_model(model, args.model, train_loader, test_loader, device=args.device, learning_rate=args.lr, num_epochs=args.epochs) 

if __name__ == "__main__":
    main()