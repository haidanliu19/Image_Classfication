# -*- coding: utf-8 -*-
# python train.py --config ./configs/model_train.yaml --device 0,1
# python train.py --config ./configs/model_train_AlexNet.yaml --device 0,1
# python train.py --config ./configs/model_train_VGGNetA.yaml --device 0,1
# python train.py --config ./configs/model_train_VGGNetA-LRN.yaml --device 0,1
# tensorboard 관련 내용 https://tutorials.pytorch.kr/intermediate/tensorboard_tutorial.html
import argparse
import logging
import os
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm
import torch.nn as nn

from utils.config_utils import load_config
from utils.general import (set_logging, init_seeds, torch_distributed_zero_first, increment_dir, plot_classes_preds,
                          matplotlib_imshow)
from utils.torch_utils import (select_device)
from utils import dataset
from models.LeNet5 import BaseModel
import importlib

import torch
from torch.utils.tensorboard import SummaryWriter

import torchvision

logger = logging.getLogger(__name__)
import test

def _get_model(opt):
    module_name = opt.base_model_name + '.' + opt.model_name
    cls = getattr(importlib.import_module(module_name), opt.class_name)
    model = cls()
    logging.info(model)
    return model
        
# https://tutorials.pytorch.kr/recipes/recipes/amp_recipe.html
def trainer(model, arg, opt, start_epoch, train_loader, test_loader, tb_writer, classes, criterion, optimizer, log_dir, rank, device):
    # 학습
    train_loss, train_correct = 0.0, 0
    results = (0.0, 0, 0.0, 0) 
    best_accuracy = -1
    running_loss=0
    for n_epoch, epoch in enumerate(range(start_epoch, opt.num_epoch)):
        start=time.time()        
        model.train()        
        
        loss_log = f'[epoch : {n_epoch + 1} - {epoch + 1}/{opt.num_epoch} ]'
        logger.info(loss_log)

        nb = len(train_loader)  
        pbar = enumerate(train_loader)
        if rank in (-1, 0):
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        
        total = len(train_loader.dataset)
        for n_iter, data_point in pbar:
      
            inputs, targets = data_point
            inputs, targets = inputs.to(device), targets.to(device)
           
            output = model(inputs)
            train_correct += (output.argmax(1) == targets).type(torch.float).sum().item()
            loss = criterion(output, targets)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip) #gradient clipping with 5
            optimizer.step()
            
            running_loss += loss.item()
     
                        
        train_loss = running_loss/  nb
        train_correct /= total
        
        # 모델 평가하기
        training = True
        test_loss, test_correct = test.test(test_loader, model, classes, criterion, training, rank, tb_writer)
        
        
        #keep the best
        if test_correct > best_accuracy:
            best_accuracy = test_correct
            torch.save(model.module.state_dict(), f'{log_dir}/weights/best.pth')
        
        
        results = (train_loss, train_correct, test_loss, test_correct)
        # Tensorboard
        if tb_writer:
            tags = ['Train Loss/Epochs', 'Train Accuracy/Epochs', 'Validation Loss/Epochs', 'Validation Accuracy/Epochs']  # params
            for x, tag in zip(list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)
            tb_writer.flush()       
        results = list(map(str, results))
        with open(f'{log_dir}/log_train.txt', 'a') as log:
            log.write("          ".join(results)+'\n')
    
    tb_writer.close()
    
def train(arg, opt, device, tb_writer, log_dir):
    logger.info("train start")
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(log_dir) 
    
    
    cuda = device.type != 'cpu'
    rank = arg.global_rank
    init_seeds(2 + rank)
    
    wdir = log_dir / 'weights'
    os.makedirs(wdir, exist_ok=True)
    # 데이터 
    if opt.dataset == 'cifar10':
        train_loader, test_loader, classes = dataset.cifar10_datast(opt, opt.num_workers, opt.batch_size)
    
    # dataset 보여주기 
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    img_grid = torchvision.utils.make_grid(images)
    tb_writer.add_image('Train ' +opt.dataset + ' Image', img_grid)
    tb_writer.flush()
    
    # 모델 
    model = _get_model(opt)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue
    
        
    model = model.to(device)
    if cuda and rank == -1 and torch.cuda.device_count() > 0:
        logger.info("DP mode")
        print("device_ids", list(range(torch.cuda.device_count())))
        device_ids = list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    # DDP mode
    if cuda and rank != -1:
        logger.info("DDP mode")
        logger.info(arg.local_rank)
        print("device_ids", list(range(torch.cuda.device_count())))
        model = DDP(model, device_ids=[arg.local_rank], output_device=arg.local_rank, find_unused_parameters=True)
    logger.info(model)
    
    # filter that only require gradient descent
    filtered_parameters = []
    params_num = []    
    for p in filter(lambda p : p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    logger.info('Tranable params : ')
    logger.info(sum(params_num))
        
    criterion = torch.nn.CrossEntropyLoss()
    if opt.optim == 'Adam':
        optimizer = torch.optim.Adam(filtered_parameters)
  
    start_epoch  = 0 
    # load pretrained model
    '''
    if opt.saved_model != '':
        base_path = './runs'
        print(f'looking for pretrained model from {os.path.join(base_path, opt.saved_model)}')
        try :
            model.load_state_dict(torch.load(os.path.join(base_path, opt.saved_model)))
            print('loading complete ')    
        except Exception as e:
            print(e)
            print('coud not load model')        
    '''
    trainer(model, arg, opt, start_epoch, train_loader, test_loader, tb_writer, classes, criterion, optimizer, log_dir, rank, device)   
    

def main(arg):    

    opt = load_config(arg.config)
    
    
    # Set DDP variables
    arg.total_batch_size = opt.batch_size
    # word_size : 총 프로세스 수, 분산 학습에 사용되는 디바이스의 총 수(gpu)
    arg.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    # RANK : ddp에서 가동되는 process id(작업 단위)
    arg.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    # Global Rank : 전체 노드(컴퓨터)에 가동되는 process ID
    set_logging(arg.global_rank)
    
    if not opt.resume:  # resume an interrupted run
        log_dir = increment_dir(Path(arg.logdir) / opt.saved_model_name, '')  # runs/exp/1
    
    # yolov5 소스 에서 device 설정하기 
    device = select_device(arg.device, batch_size = opt.batch_size)
    
    # DDP mode
    if arg.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(arg.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend = 'nccl', 
                                init_method = 'env://', 
                                #world_size = opt.world_size, 
                                #rank = opt.global_rank
                               )  # distributed backend
        assert opt.batch_size % arg.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = arg.total_batch_size // arg.world_size  
        
    logger.info("arg")
    logger.info(arg)
    logger.info("model_config")
    logger.info(opt)
    
    # tensorboard관련 내용 
    tb_writer = None
    if arg.global_rank in [-1, 0]:
        logger.info('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % opt.logdir)
        tb_writer = SummaryWriter(log_dir=log_dir)  # runs/exp0
    
    train(arg, opt, device, tb_writer, log_dir)
    
def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='path to config file')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')  
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    arg = parser.parse_args()
    return arg    

if __name__ == '__main__':
    arg = parse_arg()
    main(arg)