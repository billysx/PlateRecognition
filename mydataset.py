import torch.utils.data as data
import os
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as ET
from torchvision import datasets
import pandas as pd
import random
import torch

class PlateDataset(data.Dataset):
    def __init__(self, datapath, transform = None, istrain = True, istest = False):
        super(PlateDataset, self).__init__()
        self.transform = transform
        self.istrain   = istrain
        self.istest    = istest
        if istrain:
            self.datapath = os.path.join(datapath,"AC","train")
        else:
            self.datapath = os.path.join(datapath,"AC","test")
        self.img_list   = sorted(os.listdir(os.path.join(self.datapath,"jpeg")))
        self.label_list = sorted(os.listdir(os.path.join(self.datapath, "xml")))

    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):

        im_path    = os.path.join(self.datapath, "jpeg", self.img_list[idx])
        label_path = os.path.join(self.datapath, "xml", self.label_list[idx])

        # Image reading
        img   = Image.open(im_path)
        img   = self.transform(img)
        c,h,w = img.shape

        anno  = ET.ElementTree(file=label_path)
        label = anno.find('object').find('platetext').text
        xmin  = anno.find('object').find('bndbox').find('xmin').text
        ymin  = anno.find('object').find('bndbox').find('ymin').text
        xmax  = anno.find('object').find('bndbox').find('xmax').text
        ymax  = anno.find('object').find('bndbox').find('ymax').text
        # x,y,w,h
        # bbox  = [xmin,ymin,xmax-xmin,ymax-ymin]
        # x1,y1,x2,y2
        bbox  = [xmin,ymin,xmax,ymax]
        bbox  = torch.Tensor([int(b)  for b in bbox])
        tmp = 0
        if w != 320:
            edge = ( w - 320 )
            tmp = random.randint( max(0, int(bbox[2])-320), min(bbox[0] ,edge) )
            img = img[:,:,tmp:tmp+320]
            bbox[0] -= tmp
            bbox[2] -= tmp
            w = 320


        # target = self.encoder(bbox / torch.Tensor([w,h,w,h])) # 7x7x5
        target = bbox
        if self.istest:
             return img, target, tmp, self.img_list[idx]
        return img, target

    def encoder(self, boxes):
        '''
        boxes (tensor) [x1,y1,x2,y2]
        return grid_num x grid_num x 5
        '''
        grid_num = 14
        target = torch.zeros((grid_num, grid_num, 5))
        cell_size = 1./grid_num
        wh = boxes[2:]-boxes[:2]

        cxcy = (boxes[2:]+boxes[:2])/2  # Center

        ij = (cxcy/cell_size).ceil()-1 #
        target[int(ij[1]),int(ij[0]),4] = 1

        xy = ij*cell_size #匹配到的网格的左上角相对坐标
        delta_xy = (cxcy - xy)/cell_size
        target[int(ij[1]),int(ij[0]),2:4] = wh
        target[int(ij[1]),int(ij[0]),:2]  = delta_xy



class CharDataset(data.Dataset):
    def __init__(self, datapath, transform = None, istrain = True):
        super(CharDataset, self).__init__()
        self.transform = transform
        self.istrain   = istrain
        self.datapath  = datapath

        if istrain:
            self.datalist = pd.read_csv(os.path.join(datapath,"train.csv"))
        else:
            self.datalist = pd.read_csv(os.path.join(datapath,"val.csv"))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        imagepath = os.path.join(self.datapath, self.datalist.iloc[idx].image)
        label = self.datalist.iloc[idx].label
        img = Image.open(imagepath)
        img = self.transform(img)
        if not self.istrain:
            return img, label, self.datalist.iloc[idx].image
        return img, label





