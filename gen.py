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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# no "I" and "O"
label_dic = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"]
num_dict = {}


def get_args():
    parser = argparse.ArgumentParser("Plate recognition")
    parser.add_argument("--model_path",type=str,
        default="/mnt/hdd/yushixing/pydm/plate_r/resnet34_2/checkpoint0749-iou0.796886.pth.tar")
    parser.add_argument("--classifier_path", type=str,
        default="/mnt/hdd/yushixing/pydm/char_c/resnet34_1/checkpoint002100-acc0.992908.pth.tar")
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
    # if torch.cuda.is_available():
    #     args.use_gpu = True
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

    # model = resnet34()
    # model.load_state_dict(torch.load(args.model_path)["state_dict"])
    classifier = resnet34(num_classes = 36, inchannels=1)
    classifier.load_state_dict(torch.load(args.classifier_path, map_location=lambda storage, loc: storage)["state_dict"])
    # model.eval()
    classifier.eval()

    if args.use_gpu:
        classifier = classifier.cuda()
        device     = torch.device("cuda")
    else:
        device     = torch.device("cpu")
    classifier     = classifier.to(device)

    args.classifier  = classifier
    args.trainLoader = trainLoader
    args.valLoader   = valLoader

    validate(None, device, args, all_iters=0, is_save = 0, epoch = 0)


def segmentation(img_gray,img_thre, path, args, label):

    height = img_thre.shape[0]
    width = img_thre.shape[1]

    white_by_column = np.sum(img_thre, axis=0)/255
    height, width = img_thre.shape

    edge = [0]
    for i in range(5):
        center = int((i+1)/6*width)
        tmp = center
        max_white = 0
        for idx in range(center-2, center+3):
            if white_by_column[idx] > white_by_column[tmp]:
                tmp = idx
                max_white = white_by_column[idx]
        edge.append(tmp)
    edge.append(width)

    val_transform = transforms.Compose([
        transforms.Resize((20,20)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
    mylist = ["3/gt_376_5.jpg", "P/gt_1239_3.jpg","E/gt_144_4.jpg",
                "6/gt_604_4.jpg","0/debug_char_auxRoi_2050.jpg", "L/116-2.jpg"]
    ans = ""
    for i in range(6):
        im = img_gray[:,max(0,edge[i]-1):min(edge[i+1]+1,width)]
        # im = cv2.GaussianBlur(im,(3,3),0)
        # equalized = cv2.equalizeHist(new)
        # _, img_thre = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # img_thre = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 5)
        # res = 255 - img_thre
        # kernel = np.ones((3, 3), np.uint8)
        # kernel[0,0] = 0
        # kernel[2,2] = 0
        # kernel[0,2] = 0
        # kernel[2,0] = 0
        # res = cv2.dilate(res, kernel)
        char_dir = f"../data/mychar_data/{label[0][i]}"
        if not os.path.exists(char_dir):
            os.makedirs(char_dir)
        num_dict[label[0][i]] = num_dict.get(label[0][i],0) + 1
        cv2.imwrite(os.path.join(char_dir,f"{num_dict[label[0][i]]}.jpg"), im)



def validate(model, device, args, all_iters, is_save, epoch):
    print("validation started")
    iou_sum, giou_sum = 0.0, 0.0

    # model.eval()
    t1  = time.time()
    if args.use_gt:
        save_path = os.path.join(args.data_path,"AC","train", "plate_gt")
    else:
        save_path = os.path.join(args.data_path,"AC","train", "plate")
    read_path = os.path.join(args.data_path,"AC","train", "jpeg")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with torch.no_grad():
        for i, (data, target, tmp, imgpath, platelabel) in tqdm(enumerate(args.trainLoader,0),ncols = 100):
            all_iters += 1
            data, target = data.to(device), target.to(device)
            b, c, h, w = data.shape

            rec1 = target.squeeze()

            x1, y1, x2, y2 = int(rec1[0].item()), int(rec1[1].item()), int(rec1[2].item()), int(rec1[3].item())

            data = imageio.imread(os.path.join(read_path, imgpath[0]))
            out  = data.squeeze()[:,tmp:tmp+320, :][y1:y2, x1:x2, :]
            imageio.imsave(os.path.join(save_path, imgpath[0]), out)

            # segmentation
            img = cv2.imread(os.path.join(save_path, imgpath[0]))
            # img = cv2.medianBlur(img, 3)
            h, w, _  = img.shape
            img = img[7:-3, :, :]
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_thre = img_gray

            _, img_thre = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            segmentation(img_gray,img_thre, imgpath[0], args, platelabel)
            # exit()





if __name__ == "__main__":
    main()

