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



from PIL import Image
from tqdm import tqdm
from models import *
from mydataset import *
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def get_args():
    parser = argparse.ArgumentParser("character classification")
    parser.add_argument("--data_path", type=str,default="../data/Chars_data")
    parser.add_argument("--lr", type=float, dest="lr", default=2e-4, help="Base Learning Rate")
    parser.add_argument("--batchsize", type=int, dest="batchsize",default=32, help="optimizing batch")
    parser.add_argument("--epoch", type=int, dest="epoch", default=10, help="Number of epochs")

    parser.add_argument("--gpu", type=int, dest="gpunum", default=1, help="gpu number")
    parser.add_argument("--ft", type=int, dest="ft", default=0, help="whether it is a finetune process")
    parser.add_argument('--save', type=str, default='/mnt/hdd/yushixing/pydm/char_c/resnet34_1', help='path for saving trained models')
    parser.add_argument('--val_interval', type=int, default=1, help='validation interval')
    parser.add_argument('--save_interval', type=int, default=1, help='model saving interval')
    parser.add_argument('--auto_continue',type=bool,default = 0)
    parser.add_argument("--num_classes",type=int, default = 36)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args)

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
    torch.backends.cudnn.enabled = True

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
        ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
        ])


    model = resnet34(num_classes = args.num_classes, inchannels=1)
    lastest_model = "/mnt/hdd/yushixing/pydm/char_c/resnet34_1/checkpoint000350-acc1.000000.pth.tar"
    checkpoint = torch.load(lastest_model)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    mylist = [("P/gt_386_2.jpg",23), ("P/debug_char_auxRoi_1260.jpg",23), ("E/gt_631_1.jpg",14),
                ("E/gt_568_1.jpg",14),("P/debug_char_auxRoi_649.jpg",23), ("L/116-2.jpg",20)]

    if use_gpu:
        model = model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    for obj in mylist:
        im = Image.open(os.path.join("../data/Chars_data/", obj[0]))
        im = val_transform(im).unsqueeze(0).cuda()
        print(im.sum())
        target = torch.Tensor(obj[1])
        output = model(im)
        print(output)
        print(obj[0])
        _, pred = output.topk(1,1,True,True)
        print(pred.t())



if __name__ == "__main__":
    with torch.no_grad():
        main()

