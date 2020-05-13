import torch.utils.data as data
import os
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as ET
from torchvision import datasets
import pandas as pd

class PlateDataset(data.Dataset):
	def __init__(self, datapath, transform = None, istrain = True):
		super(PlateDataset, self).__init__()
		self.transform = transform
		self.istrain   = istrain
		if istrain:
			self.datapath = os.path.join(datapath,"AC","train")
		else:
			self.datapath = os.path.join(datapath,"AC","test")
		self.img_list   = sorted(os.listdir(os.path.join(self.datapath,"jpeg")))
		self.label_list = sorted(os.listdir(os.path.join(self.datapath, "xml")))

	def __len__(self):
		return len(self.img_list)


	def __getitem__(self, idx):

		im_path    = self.img_list[idx]
		label_path = self.label_list[idx]

		# Image reading
		img = Image.open(f_img)
		img = self.transform(img)

		anno = ET.ElementTree(file=f_xml)
		label = anno.find('object').find('platetext').text
		xmin = anno.find('object').find('bndbox').find('xmin').text
		ymin = anno.find('object').find('bndbox').find('ymin').text
		xmax  = anno.find('object').find('bndbox').find('xmax').text
		ymax = anno.find('object').find('bndbox').find('ymax').text
		bbox = [xmin,ymin,xmax,ymax]
		bbox = [int(b)  for b in bbox]

		return img, bbox


class CharDataset(data.Dataset):
	def __init__(self, datapath, transform = None, istrain = True):
		super(CharDataset, self).__init__()
		self.transform = transform
		self.istrain   = istrain

		if istrain:
			self.datalist = pd.read_csv(os.path.join(datapath,"train.csv"))
		else:
			self.datalist = pd.read_csv(os.path.join(datapath,"val.csv"))

	def __len__(self):
		return len(self.img_list)

	def __getitem__(self, idx):
        imagepath = os.path.join(self.datapath, self.datalist.iloc[idx].img)
        label = self.datalist.iloc[idx].label
        img = Image.open(imagepath)
        img = self.transform(img)

		# Image reading
		img = Image.open()
		img = self.transform(img)





