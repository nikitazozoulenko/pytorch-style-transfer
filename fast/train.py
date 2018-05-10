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
from PIL import Image

coco_path = "/hdd/Data/MSCOCO2017/images"
annFile = "/hdd/Data/MSCOCO2017/annotations"

train_data_feeder = DataFeeder(coco_path+"/train2017/",
                               annFile+"/captions_train2017.json",
                               preprocess_workers=1, cuda_workers=1,
                               numpy_size=20, cuda_size=2, batch_size=1)

train_data_feeder.start_queue_threads()

image_transformer_network = ImageTransformerNetwork().cuda()
#image_transformer_network = torch.load("savedir/model_2_acidcrop_it90k.pt")
vgg = VGG().cuda()
vgg.eval()
loss = Loss().cuda()
for param in vgg.parameters():
    param.requires_grad = False


learning_rate = 0.0001
optimizer = optim.Adam(image_transformer_network.parameters(), lr=learning_rate)

style = Variable(torch.from_numpy(np.asarray(Image.open("/hdd/Images/mosaic.jpg").convert("RGB").resize((640,480)))).float().cuda().permute(2,0,1).unsqueeze(0)/255)
style_features = vgg(style, ["1_1", "2_1", "3_1", "4_1", "5_1"])


def train_batch(i, data_feeder):
    batch = data_feeder.get_batch()
    content = batch
    content = content/255
    image = image_transformer_network(content)
    content_3_2 = vgg(content, ["3_2"])[0]
    input_features = vgg(image, ["1_1", "2_1", "3_1", "4_1", "5_1", "3_2"])
    input_3_2 = input_features[-1]
    input_features = input_features[:-1]
    
    total_loss = loss(input_features, input_3_2, content_3_2, style_features)
    return total_loss

losses = []
x_indices = []
val_x_indices = []
val_losses = []
num_iterations = 1000000
image_transformer_network.train()
for i in range(num_iterations):
    # training loss
    optimizer.zero_grad()
    total_loss = train_batch(i, train_data_feeder)
    total_loss.backward()
    optimizer.step()

    losses += [total_loss.data.cpu().numpy()]
    x_indices += [i]

    if i % 10000 == 0 and i != 0:
        torch.save(image_transformer_network, "savedir/model_mosaic_it"+str(i//1000)+"k.pt")

    # print progress
    if i % 100 == 0:
        print(i)

graph_losses(losses, x_indices, val_losses, val_x_indices)

image_transformer_network.eval()
batch = train_data_feeder.get_batch()
content = batch
content = content/255
image = image_transformer_network(content)
show_image(content)
show_image(style)
show_image(image)
train_data_feeder.kill_queue_threads()
