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


import char_net
from mydataset import *
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# no "I" and "O"
label_dic = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"]
plate_cnt = 0
char_cnt = 0
def get_args():
    parser = argparse.ArgumentParser("Plate recognition")

    parser.add_argument("--classifier_path", type=str, default="/home/yusx/data/char_model.pth.tar")
    parser.add_argument("--data_path", type=str,default="../data/Plate_dataset")
    parser.add_argument("--batchsize", type=int, dest="batchsize",default=1, help="optimizing batch")

    parser.add_argument("--gpu", type=int, dest="gpunum", default=1, help="gpu number")
    parser.add_argument("--use_gt", type=int, default=1, help="whether use ground truth label")
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
    trainset = PlateDataset(args.data_path, istrain=True, transform = train_transform, istest = True)
    valset   = PlateDataset(args.data_path, istrain=False, transform = val_transform, istest = True)
    print(len(trainset),len(valset))

    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    valLoader   = torch.utils.data.DataLoader(valset, batch_size=args.batchsize, shuffle=False)
    print('load data successfully')

    classifier = char_net.resnet34(num_classes = 34, inchannels=1)
    classifier.load_state_dict(torch.load(args.classifier_path, map_location=lambda storage, loc: storage)["state_dict"])
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

    validate(device, args)


def segmentation(img_gray,img_thre, path, args, label):
    if not os.path.exists(f"../seg/{path[:-4]}/"):
        os.makedirs(f"../seg/{path[:-4]}/")
    writefile = open(f"../seg/{path[:-4]}/pred.txt","w")
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

        cv2.imwrite(f"../seg/{path[:-4]}/{i+1}.jpg", im)
        im = Image.open(f"../seg/{path[:-4]}/{i+1}.jpg")


        im = val_transform(im).unsqueeze(0)
        if args.use_gpu:
            im = im.cuda()
        output = args.classifier(im)
        _, pred = output.topk(5, 1, True, True)
        pred = pred.t()
        if label_dic[pred[0,0]] == label[0][i]:
            global char_cnt
            char_cnt+=1
        ans += label_dic[pred[0,0]]

    if(ans == label[0]):
        global plate_cnt
        plate_cnt += 1

    print(ans,file=writefile,end="")
    print("",file=writefile)
    print(label[0],file = writefile)
    writefile.close()


def validate(device, args):
    print("validation started")
    iou_sum, giou_sum = 0.0, 0.0

    # model.eval()
    t1  = time.time()
    if args.use_gt:
        save_path = os.path.join(args.data_path,"AC","test", "plate_gt")
    else:
        save_path = os.path.join(args.data_path,"AC","test", "plate")
    read_path = os.path.join(args.data_path,"AC","test", "jpeg")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with torch.no_grad():
        for i, (data, target, tmp, imgpath, platelabel) in tqdm(enumerate(args.valLoader,0),ncols = 100):

            data, target = data.to(device), target.to(device)
            b, c, h, w = data.shape

            rec1 = target.squeeze()
            if not args.use_gt:
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
                rec1 = pos.squeeze()

            x1, y1, x2, y2 = int(rec1[0].item()), int(rec1[1].item()), int(rec1[2].item()), int(rec1[3].item())

            data = imageio.imread(os.path.join(read_path, imgpath[0]))
            out  = data.squeeze()[:,tmp:tmp+320, :][y1:y2, x1:x2, :]
            imageio.imsave(os.path.join(save_path, imgpath[0]), out)

            # Segmentation
            img = cv2.imread(os.path.join(save_path, imgpath[0]))
            h, w, _  = img.shape
            img = img[7:-3, 2:-2, :]
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_thre = img_gray

            _, img_thre = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            segmentation(img_gray,img_thre, imgpath[0], args, platelabel)

    global char_cnt
    print(f"Charater recognition accuracy: {100*char_cnt / 600:.2f}%")
    global plate_cnt
    print(f"Plate recognition accuracy: {100*plate_cnt / 100:.2f}%")




if __name__ == "__main__":
    main()

