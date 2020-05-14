import os
import sys
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import numpy as np
import time
import logging
import argparse
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import imageio


from models import *
from mydataset import *
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def get_args():
    parser = argparse.ArgumentParser("Plate recognition")
    parser.add_argument("--model_path",type=str, default="/mnt/hdd/yushixing/pydm/plate_r/resnet34_2/checkpoint0749-iou0.796886.pth.tar")
    parser.add_argument("--data_path", type=str,default="../data/Plate_dataset")
    parser.add_argument("--batchsize", type=int, dest="batchsize",default=1, help="optimizing batch")

    parser.add_argument("--gpu", type=int, dest="gpunum", default=1, help="gpu number")
    parser.add_argument("--use_gt", type=int, default=1, help="whether use ground truth label")
    parser.add_argument("--ft", type=int, dest="ft", default=0, help="whether it is a finetune process")
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
    trainset = PlateDataset(args.data_path, istrain=True, transform = train_transform, istest = True)
    valset   = PlateDataset(args.data_path, istrain=False, transform = val_transform, istest = True)
    print(len(trainset),len(valset))

    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    valLoader   = torch.utils.data.DataLoader(valset, batch_size=args.batchsize, shuffle=False)
    print('load data successfully')

    model = resnet34()
    model.load_state_dict(torch.load(args.model_path)["state_dict"])

    if args.use_gpu:
        model     = model.cuda()
        device    = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    args.trainLoader = trainLoader
    args.valLoader = valLoader

    validate(model, device, args, all_iters=0, is_save = 0, epoch = 0)


def segmentation(img_thre, path):
    white = []  # 记录每一列的白色像素总和
    black = []  # ..........黑色.......
    height = img_thre.shape[0]
    width = img_thre.shape[1]
    white_max = 0
    black_max = 0

    white_by_row    = np.sum(img_thre, axis=1)/255
    black_by_row    = height - white_by_row
    white_max       = white_by_row.max()
    black_max       = black_by_row.max()
    starth, endh = 0, height-1
    for i in range(0, int(height/3)):
        if white_by_row[i] >= 60:
            starth = i
            break

    for i in range(0, int(height/3)):
        if white_by_row[height -1 -i] >= 60:
            endh = height -1 -i
            break
    img_thre = img_thre[starth:endh, :]


    white_by_column = np.sum(img_thre, axis=0)/255
    black_by_column = width  - white_by_column
    startw, endw = 0, width-1

    for i in range(0, 5):
        if white_by_column[i] <= 10:
            startw = i


    img_thre = img_thre[:, startw:endw]

    cv2.imwrite(f"binarize/{path}", img_thre)

    # arg = False  # False表示白底黑字；True表示黑底白字

    # # 分割图像
    # def find_end(start_):
    #     end_ = start_+1
    #     for m in range(start_+1, width-1):
    #         if (white_by_column[m]) > (0.95 * white_max):  # 0.95这个参数请多调整，对应下面的0.05
    #             end_ = m
    #             break
    #     return end_

    # n = 1
    # start = 1
    # end = 2
    # while n < width-2:
    #     n += 1
    #     if (white_by_column[n] if arg else black_by_column[n]) > (0.05 * black_max):
    #         # 上面这些判断用来辨别是白底黑字还是黑底白字
    #         # 0.05这个参数请多调整，对应上面的0.95
    #         start = n
    #         end = find_end(start)
    #         n = end
    #         if end-start > 5:
    #             cj = img_thre[1:height, start:end]
    #             cv2.imwrite(f"binarize/{n}.jpg",cj)


def validate(model, device, args, all_iters, is_save, epoch):
    print("validation started")
    iou_sum, giou_sum = 0.0, 0.0

    model.eval()
    t1  = time.time()
    if args.use_gt:
        save_path = os.path.join(args.data_path,"AC","test", "plate_gt")
    else:
        save_path = os.path.join(args.data_path,"AC","test", "plate")
    read_path = os.path.join(args.data_path,"AC","test", "jpeg")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with torch.no_grad():
        for i, (data, target, tmp, imgpath) in tqdm(enumerate(args.valLoader,0),ncols = 100):
            all_iters += 1
            data, target = data.to(device), target.to(device)
            b, c, h, w = data.shape
            output = torch.sigmoid(model(data))

            # Get the corrdinate of the car plate
            pred   = torch.zeros_like(output)
            pred[:, 2:4] = output[:, 0:2] + output[:, 2:4]
            pred[:, 0:2] = output[:, 0:2]
            pred = pred.clamp(0.0, 1.0)
            if args.use_gpu:
                scale = torch.Tensor([w,h,w,h]).cuda()
            else:
                scale = torch.Tensor([w,h,w,h])
            pos = pred*scale
            if args.use_gt:
                rec1 = target.squeeze()
            else:
                rec1 = pos.squeeze()

            x1, y1, x2, y2 = int(rec1[0].item()), int(rec1[1].item()), int(rec1[2].item()), int(rec1[3].item())

            data = imageio.imread(os.path.join(read_path, imgpath[0]))
            out  = data.squeeze()[:,tmp:tmp+320, :][y1:y2, x1:x2, :]
            imageio.imsave(os.path.join(save_path, imgpath[0]), out)

            # segmentation
            img = cv2.imread(os.path.join(save_path, imgpath[0]))
            # img = cv2.medianBlur(img, 3)
            h, w, _  = img.shape
            img = img[7:h-3, :, :]
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_thre = img_gray
            img_thre = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)
            # cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV, img_thre)
            # cv2.imwrite(os.path.join(save_path, imgpath[0]), img_thre)
            segmentation(img_thre, imgpath[0])
            # exit()





if __name__ == "__main__":
    main()

