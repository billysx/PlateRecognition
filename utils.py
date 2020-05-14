import os
import pandas as pd
import numpy as np
import torch.nn.init as init
import torch.nn as nn
import torch
import time

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        # print("inside CrossEntropy",inputs.shape, targets)

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * \
            targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    # box [xmin,ymin,xmax,ymax]
    xA = max(boxA[0].item(), boxB[0].item())
    yA = max(boxA[1].item(), boxB[1].item())
    xB = min(boxA[2].item(), boxB[2].item())
    yB = min(boxA[3].item(), boxB[3].item())

    # compute the area of intersection rectangle

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def single_giou(rec1,rec2):
    #分别是第一个矩形左右上下的坐标
    x1,y1,x2,y2 = rec1
    x3,y3,x4,y4 = rec2
    iou = bb_intersection_over_union(rec1,rec2)
    area_C = (max(x1,x2,x3,x4)-min(x1,x2,x3,x4))*(max(y1,y2,y3,y4)-min(y1,y2,y3,y4))
    print(f"area_c: {area_C}")
    area_1 = (x2-x1)*(y2-y1)
    area_2 = (x4-x3)*(y4-y3)
    sum_area = area_1 + area_2
    print(f"sum_area: {sum_area}")
    w1 = x2 - x1   #第一个矩形的宽
    w2 = x4 - x3   #第二个矩形的宽
    h1 = y2 - y1
    h2 = y4 - y3
    W = min(x1,x2,x3,x4)+w1+w2-max(x1,x2,x3,x4)    #交叉部分的宽
    H = min(y1,y2,y3,y4)+h1+h2-max(y1,y2,y3,y4)    #交叉部分的高
    Area = W*H    #交叉的面积
    add_area = sum_area - Area    #两矩形并集的面积
    print(f"add_area: {add_area}")
    end_area = (area_C - add_area)/area_C    #闭包区域中不属于两个框的区域占闭包区域的比重
    giou = iou - end_area
    return giou


def charDataset_txt_gen(datapath, validation_split=0.15):
    datalist = []
    labellist = []
    labelname = []
    cnt = -1
    for i,dir in enumerate(sorted(os.listdir(datapath)),0):
        if len(dir)!=1:
            continue
        cnt += 1
        for file in os.listdir(os.path.join(datapath,dir)):
            datalist.append(os.path.join(dir,file))
            labellist.append(i)
            labelname.append(dir)

    data_frame = pd.DataFrame({"image":datalist, "label":labellist,"name":labelname},
        columns = ["image","label","name"])

    dataset_size = len(data_frame)
    print(dataset_size)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    trainset = data_frame.loc[train_indices]
    valset   = data_frame.loc[val_indices]
    print(len(trainset),len(valset))
    trainset.to_csv(os.path.join(datapath,"train.csv"),index=False,sep=',')
    valset.to_csv(os.path.join(datapath,"val.csv"),index=False,sep=',')


class GiouLoss(nn.Module):

    def __init__(self):
        super(GiouLoss, self).__init__()

    def forward(self, args, pred, target):
        return self.batch_giou(args, pred, target)

    def batch_iou(self, args, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        # box [xmin,ymin,xmax,ymax]
        xA = torch.max(boxA[:,0], boxB[:,0])
        yA = torch.max(boxA[:,1], boxB[:,1])
        xB = torch.min(boxA[:,2], boxB[:,2])
        yB = torch.min(boxA[:,3], boxB[:,3])

        # compute the area of intersection rectangle
        if args.use_gpu:
            my_zero = torch.Tensor([0]).cuda()
        else:
            my_zero = torch.Tensor([0])
        interArea = torch.max(my_zero, xB - xA + 1) * torch.max(my_zero, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[:,2] - boxA[:,0] + 1) * (boxA[:,3] - boxA[:,1] + 1)
        boxBArea = (boxB[:,2] - boxB[:,0] + 1) * (boxB[:,3] - boxB[:,1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / (boxAArea + boxBArea - interArea).float()

        # return the intersection over union value
        return iou, interArea

    def batch_giou(self, args, rec1, rec2):
        x1, y1, x2, y2 = rec1[:,0:1], rec1[:,1:2], rec1[:,2:3], rec1[:,3:]
        x3, y3, x4, y4 = rec2[:,0:1], rec2[:,1:2], rec2[:,2:3], rec2[:,3:]

        xcat, ycat = torch.cat([x1, x2, x3, x4], dim=1), torch.cat([y1, y2, y3, y4], dim=1)

        xmin, ymin = torch.min(xcat, dim=1)[0].unsqueeze(1), torch.min(ycat, dim=1)[0].unsqueeze(1)
        xmax, ymax = torch.max(xcat, dim=1)[0].unsqueeze(1), torch.max(ycat, dim=1)[0].unsqueeze(1)
        iou, interArea = self.batch_iou(args, rec1, rec2)
        area_C = (xmax - xmin) * (ymax - ymin)
        area_1 = (x2 - x1) * (y2 - y1)
        area_2 = (x4 - x3) * (y4 - y3)
        sum_area = area_1 + area_2



        add_area = sum_area - interArea.unsqueeze(1)    # add area of the two box
        end_area = (area_C - add_area)/area_C    # The area that is out of the box but in the bibao
        # print(f"sum_area: {sum_area}")
        # print(f"whole area: {area_C}")
        # print(f"intersection: {interArea}")
        # print(f"iou: {iou}")
        giou = iou.unsqueeze(1) - end_area

        return iou, giou