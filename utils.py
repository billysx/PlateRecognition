import os
import pandas
import numpy as np

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    # box [xmin,ymin,xmax,ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

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

def charDataset_txt_gen(datapath):
    datalist = []
    labellist = []
    labelname = []
    for i,dir in enumerate(os.listdir(datapath),0):
        if len(dir)!=1:
            continue
        for file in os.listdir(os.path.join(datapath,dir)):
            datalist.append(os.path.join(dir,file))
            labellist.append(i)
            labelname.append(dir)

    data_frame = pd.DataFrame({"image":datalist, "label":labellist,"name":labelname},
        columns = ["image","label","name"])

    validation_split = 0.15
    dataset_size = len(data_frame)
    print(dataset_size)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    trainset = data_frame.loc[train_indices]
    valset   = data_frame.loc[val_indices]
    print(len(trainset),len(valset))
    trainset.to_csv("train.csv",index=False,sep=',')
    valset.to_csv("val.csv",index=False,sep=',')
