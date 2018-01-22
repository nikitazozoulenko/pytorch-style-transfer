import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from network import VGG, Loss
from utils import show_image, graph_losses
from PIL import Image, ImageOps

import torchvision.datasets as datasets
import torchvision.transforms as transforms

coco_data = datasets.CocoDetection(root = "/hdd/Data/MSCOCO2017/images/train2017",
                                  annFile = "/hdd/Data/MSCOCO2017/annotations/instances_train2017.json")

def process_example(example):
    image, objects = example
    width, height = image.size
    num_objects = len(objects)
    
    gt_bboxes = []
    gt_cats = []
    for obj in objects:
        cat = obj["category_id"]
        gt_cats += [cat]
        
        bbox = np.copy(obj["bbox"])
        bbox[2] = (bbox[2] + bbox[0]) / width
        bbox[3] = (bbox[3] + bbox[1]) / height
        bbox[0] = bbox[0] / width
        bbox[1] = bbox[1] / height
        gt_bboxes += [bbox]

    gt_bboxes = np.array(gt_bboxes).astype(np.float32)
    gt_cats = np.array(gt_cats)

    #flip horizontally
    random = np.random.randint(0,2)
    random = 0
    if(random == 0):
        image = ImageOps.mirror(image)
        #xmax = 1-xmin
        xmax_temp = np.copy(gt_bboxes[:, 2:3])
        gt_bboxes[:, 2:3] = 1 - gt_bboxes[:, 0:1]
        #xmin = 1-xmax
        gt_bboxes[:, 0:1] = 1 - xmax_temp
        
    image_array = np.asarray(image)
    return [image_array, gt_bboxes, gt_cats, num_objects]

for i in range(5):
    print(i)
    print(process_example(coco_data[1]))
    
