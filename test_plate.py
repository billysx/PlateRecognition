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
    parser.add_argument("--batchsize", type=int, dest="batchsize",default=1, help="optimizing batch")
    parser.add_argument("--epoch", type=int, dest="epoch", default=1000, help="Number of epochs")

    parser.add_argument("--gpu", type=int, dest="gpunum", default=1, help="gpu number")
    parser.add_argument("--ft", type=int, dest="ft", default=0, help="whether it is a finetune process")
    parser.add_argument('--save', type=str, default='/mnt/hdd/yushixing/pydm/plate_r/resnet34_1', help='path for saving trained models')
    parser.add_argument('--val_interval', type=int, default=1, help='validation interval')
    parser.add_argument('--save_interval', type=int, default=10, help='model saving interval')
    parser.add_argument('--model_path',type=str,default = "/home/yusx/data/loc_model.pth.tar")
    parser.add_argument("--num_classes",type=int, default = 4)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args)

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
    criterion = GiouLoss()

    if args.use_gpu:
        #model = nn.DataParallel(model).cuda()
        model     = model.cuda()
        criterion = criterion.cuda()
        device    = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    checkpoint = torch.load(args.model_path, map_location=None if args.use_gpu else 'cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print('load from checkpoint')

    args.loss_function = criterion

    args.trainLoader = trainLoader
    args.valLoader = valLoader

    validate(model, device, args, all_iters=0, is_save = 0, epoch = 0)



def validate(model, device, args, all_iters, is_save, epoch):
    print("validation started")
    iou_sum, giou_sum = 0.0, 0.0

    loss_function = args.loss_function

    model.eval()
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(args.valLoader,0),ncols = 100):
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
        print(printInfo)




if __name__ == "__main__":
    main()

