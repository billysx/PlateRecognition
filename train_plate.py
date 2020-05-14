import os
import sys
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import numpy as np
import PIL
import time
import logging
import argparse
import torch.nn.functional as F



from PIL import Image
from tqdm import tqdm
from models import *
from mydataset import *
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
best_iou = 0.0

def get_args():
    parser = argparse.ArgumentParser("Plate recognition")
    parser.add_argument("--data_path", type=str,default="../data/Plate_dataset")
    parser.add_argument("--lr", type=float, dest="lr", default=2e-4, help="Base Learning Rate")
    parser.add_argument("--batchsize", type=int, dest="batchsize",default=16, help="optimizing batch")
    parser.add_argument("--epoch", type=int, dest="epoch", default=1000, help="Number of epochs")

    parser.add_argument("--gpu", type=int, dest="gpunum", default=1, help="gpu number")
    parser.add_argument("--ft", type=int, dest="ft", default=0, help="whether it is a finetune process")
    parser.add_argument('--save', type=str, default='/mnt/hdd/yushixing/pydm/plate_r/resnet34_1', help='path for saving trained models')
    parser.add_argument('--val_interval', type=int, default=1, help='validation interval')
    parser.add_argument('--save_interval', type=int, default=10, help='model saving interval')
    parser.add_argument('--auto_continue',type=bool,default = 0)
    parser.add_argument("--num_classes",type=int, default = 4)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args)

    # Log
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    fh = logging.FileHandler(os.path.join('log/train-{}{:02}{}'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    args.use_gpu = False
    if torch.cuda.is_available():
        args.use_gpu = True
    torch.backends.cudnn.enabled = True

    # dataset splitting
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])
    trainset = PlateDataset(args.data_path, istrain=True, transform = train_transform)
    valset   = PlateDataset(args.data_path, istrain=False, transform = val_transform)
    print(len(trainset),len(valset))

    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    valLoader   = torch.utils.data.DataLoader(valset, batch_size=args.batchsize, shuffle=False)
    print('load data successfully')

    model = resnet34()
    init_weights(model)
    criterion = GiouLoss()

    optimizer = torch.optim.Adamax(model.parameters(),lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                    lambda step : (1.0-step/args.epoch) if step <= args.epoch else 0, last_epoch=-1)


    if args.use_gpu:
        #model = nn.DataParallel(model).cuda()
        model     = model.cuda()
        criterion = criterion.cuda()
        device    = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    all_iters = 0
    if args.auto_continue:
        lastest_model, iters = get_lastest_model()
        if lastest_model is not None:
            all_iters = iters
            checkpoint = torch.load(lastest_model, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('load from checkpoint')
            for i in range(iters):
                scheduler.step()

    args.optimizer = optimizer
    args.loss_function = criterion
    args.scheduler = scheduler

    args.trainLoader = trainLoader
    args.valLoader = valLoader

    for epoch in range(args.epoch):
        all_iters = train(model, device, args, epoch=epoch, all_iters=all_iters)
        if (epoch+1) % args.val_interval == 0:
            validate(model, device, args, all_iters=all_iters, is_save = (epoch+1)%args.save_interval==0, epoch = epoch)
        scheduler.step()



def train(model, device, args, epoch, all_iters=None):

    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler

    iou_sum,  giou_sum  = 0.0, 0.0
    iou_intv, giou_intv = 0.0, 0.0
    model.train()
    optimizer.zero_grad()
    printinterval = 1
    print(f"---------------  [EPOCH {epoch}]  ---------------")

    for i, (data, target) in tqdm(enumerate(args.trainLoader,0),ncols = 100):
        t1 = time.time()
        all_iters += 1
        data, target = data.to(device), target.to(device)
        b, c, h, w = data.shape
        output = torch.sigmoid(model(data))
        pred   = torch.zeros_like(output)
        pred[:, 2:4] = output[:, 0:2] + output[:, 2:4]
        pred[:, 0:2] = output[:, 0:2]
        pred = pred.clamp(0.0, 1.0)


        # compute loss
        if args.use_gpu:
            scale = torch.Tensor([w,h,w,h]).cuda()
        else:
            scale = torch.Tensor([w,h,w,h])
        pos = pred*scale

        iou, giou = args.loss_function(args, pos, target)

        loss = (1 - giou).mean()

        # print("forward time", time.time() - t1)
        loss.backward()

        # print("back ward time", time.time()-t1)
        for p in model.parameters():
            if p.grad is not None and p.grad.sum() == 0:
                p.grad = None

        optimizer.step()
        optimizer.zero_grad()

        iou_sum   += iou.mean().item()
        iou_intv  += iou.mean().item()
        giou_sum  += giou.mean().item()
        giou_intv += giou.mean().item()

        if (i+1)%printinterval == 0:
            printInfo = 'EPOCH {} ITER {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch,all_iters,scheduler.get_lr()[0], loss.item()) + \
                        'IoU = {:.6f},\t'.format(iou_intv / printinterval) + \
                        'GIoU = {:.6f},\t'.format(giou_intv / printinterval)
            logging.info(printInfo)
            iou_intv, giou_intv = 0.0, 0.0
        # exit()
    printInfo = 'EPOCH {}: \tloss = {:.6f},\t'.format(scheduler.get_lr()[0], loss.item()) + \
                    'IoU = {:.6f},\t'.format(iou_sum /len(args.trainLoader)) + \
                    'GIoU = {:.6f},\t'.format(giou_sum / len(args.trainLoader))
    logging.info(printInfo)
    t1 = time.time()
    iou_sum, giou_sum = 0.0, 0.0

    return all_iters


def validate(model, device, args, all_iters, is_save, epoch):
    print("validation started")
    iou_sum, giou_sum = 0.0, 0.0

    loss_function = args.loss_function

    model.eval()
    t1  = time.time()
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(args.valLoader,0),ncols = 100):
            t1 = time.time()
            all_iters += 1
            data, target = data.to(device), target.to(device)
            b, c, h, w = data.shape
            output = torch.sigmoid(model(data))
            pred   = torch.zeros_like(output)
            pred[:, 2:4] = output[:, 0:2] + output[:, 2:4]
            pred[:, 0:2] = output[:, 0:2]
            pred = pred.clamp(0.0, 1.0)


            # compute loss
            if args.use_gpu:
                scale = torch.Tensor([w,h,w,h]).cuda()
            else:
                scale = torch.Tensor([w,h,w,h])
            pos = pred*scale

            iou, giou = args.loss_function(args, pos, target)

            iou_sum   += iou.mean().item()
            giou_sum  += giou.mean().item()


        printInfo = 'ITER {}:\t'.format(all_iters) + \
                    'IoU = {:.6f},\t'.format(iou_sum / len(args.valLoader)) + \
                    'GIoU = {:.6f},\t'.format(giou_sum / len(args.valLoader))
        logging.info(printInfo)

    # exit()
    global best_iou
    iou_avg = iou_sum / len(args.valLoader)
    if is_save and iou_avg>best_iou:
        best_iou = iou_avg
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        filename = os.path.join(
            args.save, f"checkpoint{epoch:04}-iou{iou_sum / len(args.valLoader):4f}.pth.tar")
        print(filename)
        torch.save({'state_dict': model.state_dict(),}, filename)


if __name__ == "__main__":
    main()

