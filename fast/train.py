import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from network import ImageTransformerNetwork
from loss import Loss, VGG
from utils import show_image, graph_losses
from data_feeder import DataFeeder

style_root = "/hdd/Data/painters"
coco_path = "/hdd/Data/MSCOCO2017/images"
annFile = "/hdd/Data/MSCOCO2017/annotations"

train_data_feeder = DataFeeder(coco_path+"/train2017/",
                               annFile+"/captions_train2017.json",
                               style_root+"/test/",
                               preprocess_workers=4, cuda_workers=1,
                               numpy_size=20, cuda_size=2, batch_size=4)
train_data_feeder.start_queue_threads()



image_transformer_network = ImageTransformerNetwork().cuda()
vgg = VGG().cuda()
loss = Loss().cuda()
for param in vgg.parameters():
    param.requires_grad = False

#learning_rate = 0.001
learning_rate = 0.0001
optimizer = optim.SGD(image_transformer_network.parameters(), lr=learning_rate,
                      momentum=0.9, weight_decay=0.0001)

def train_batch(i, data_feeder):
    batch = data_feeder.get_batch()
    style, content = batch
    style = style/255
    content = content/255
    image = image_transformer_network(content, style)

    input_features = vgg(image)
    content_features = vgg(content)
    style_features = vgg(style)
    
    total_loss = loss(input_features, content_features, style_features)
    return total_loss

losses = []
x_indices = []
num_iterations = 5000
for i in range(num_iterations):
    # training loss
    optimizer.zero_grad()
    total_loss = train_batch(i, train_data_feeder)
    total_loss.backward()
    optimizer.step()

    losses += [total_loss.data.cpu().numpy()]
    x_indices += [i]

    if i in [346363]:
        learning_rate *= 10
        print("updated learning rate: current lr:", learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    # print progress
    if i % 100 == 0:
        print(i)

batch = train_data_feeder.get_batch()
style, content = batch
style = style/255
content = content/255
image = image_transformer_network(content, style)
show_image(content)
show_image(style)
show_image(image)
graph_losses(losses, x_indices)
train_data_feeder.kill_queue_threads()
