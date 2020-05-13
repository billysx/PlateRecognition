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
    parser.add_argument('--save', type=str, default='/mnt/hdd/yushixing/char_c/resnet34', help='path for saving trained models')
    parser.add_argument('--val_interval', type=int, default=1, help='validation interval')
    parser.add_argument('--save_interval', type=int, default=1, help='model saving interval')
    parser.add_argument('--auto_continue',type=bool,default = 0)
    parser.add_argument("--num_classes",type=int, default = 36)
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

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
    torch.backends.cudnn.enabled = True

    # dataset
    train_csv_path = os.path.join(args.data_path, "train.csv")
    test_csv_path = os.path.join(args.data_path, "test.csv")
    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        charDataset_txt_gen(args.data_path)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
        ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
        ])
    trainset    = CharDataset(args.data_path, istrain=True, transform = train_transform)
    valset      = CharDataset(args.data_path, istrain=False, transform = val_transform)
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    valLoader   = torch.utils.data.DataLoader(valset, batch_size=args.batchsize, shuffle=False)

    print('load data successfully')

    model = resnet34(num_classes = args.num_classes, inchannels=1)
    init_weights(model)
    criterion_smooth = CrossEntropyLabelSmooth(args.num_classes, 0.1)


    optimizer = torch.optim.Adamax(model.parameters(),lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                    lambda step : (1.0-step/args.epoch) if step <= args.epoch else 0, last_epoch=-1)


    if use_gpu:
        model = model.cuda()
        loss_function = criterion_smooth.cuda()
        device = torch.device("cuda")
    else:
        loss_function = criterion_smooth
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
    args.loss_function = loss_function
    args.scheduler = scheduler

    args.trainLoader = trainLoader
    args.valLoader = valLoader

    for epoch in range(args.epoch):
        all_iters = train(model, device, args, epoch=epoch, all_iters=all_iters)
        if (epoch+1) % args.val_interval == 0:
            validate(model, device, args, all_iters=all_iters, is_save = (epoch+1)%args.save_interval==0)
        scheduler.step()



def train(model, device, args, epoch, all_iters=None):

    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler


    t1 = time.time()
    Top1_err, Top5_err = 0.0, 0.0
    Top1_intv,Top5_intv = 0.0, 0.0
    model.train()
    optimizer.zero_grad()
    printinterval = 100
    print(f"---------------  [EPOCH {epoch}]  ---------------")
    for i, (data, target) in tqdm(enumerate(args.trainLoader,0),ncols = 100):
        all_iters += 1
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)

        output = model(data).squeeze()

        loss = loss_function(output, target)
        loss.backward()

        for p in model.parameters():
            if p.grad is not None and p.grad.sum() == 0:
                p.grad = None

        optimizer.step()
        optimizer.zero_grad()
        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        Top1_err += 1 - prec1.item() / 100
        Top5_err += 1 - prec5.item() / 100
        Top1_intv += 1 - prec1.item() / 100
        Top5_intv += 1 - prec5.item() / 100

        if (i+1)%printinterval == 0:
            printInfo = 'EPOCH {} ITER {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch,all_iters,scheduler.get_lr()[0], loss.item()) + \
                        'Top-1 err = {:.6f},\t'.format(Top1_intv / printinterval) + \
                        'Top-5 err = {:.6f},\t'.format(Top5_intv / printinterval)
            logging.info(printInfo)
            Top1_intv, Top5_intv = 0.0, 0.0
    t1 = time.time()
    Top1_err, Top5_err = 0.0, 0.0

    return all_iters


def validate(model, device, args, all_iters, is_save):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    loss_function = args.loss_function

    model.eval()
    max_val_iters = 250
    t1  = time.time()
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(args.valLoader,0)):

            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)

            output = model(data).squeeze()
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            # print(i,loss.item(),prec1.item(),prec5.item())
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)


    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(all_iters, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(1 - top1.avg / 100) + \
              'Top-5 err = {:.6f},\t'.format(1 - top5.avg / 100) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    logging.info(logInfo)
    # exit()

    if is_save:
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        filename = os.path.join(
            args.save, f"checkpoint{all_iters:06}-acc{top1.avg / 100:4f}.pth.tar")
        print(filename)
        torch.save({'state_dict': model.state_dict(),}, filename)


if __name__ == "__main__":
    main()

