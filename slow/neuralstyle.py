import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from network import VGG, Loss
from utils import show_image, graph_losses
from PIL import Image

import argparse

parser = argparse.ArgumentParser(description='PyTorch Neural Style Implementation')
parser.add_argument("--style", type=str, metavar="PATH", help="path to style image")
parser.add_argument("--content", type=str, metavar="PATH", help="path to content image")
parser.add_argument("--iterations", default=100, type=int, metavar="N", help="how many training iterations to optimize the loss function")
args = parser.parse_args()

height, width = 960, 540
#IMAGES
#input_image = Variable(torch.randn(1,3,width, height).cuda(), requires_grad = True)

style_image = Image.open("/hdd/Images/polyphoenix.jpg").resize((height,width))
style_image = (np.array(style_image))
style_image = Variable(torch.from_numpy(style_image).permute(2,0,1).unsqueeze(0)).float().cuda()

content_image = Image.open("/hdd/Images/graffiti.jpg").resize((height,width))
content_image = (np.array(content_image))
content_image = Variable(torch.from_numpy(content_image).permute(2,0,1).unsqueeze(0)).float().cuda()

input_image = Variable(content_image.data.clone(), requires_grad = True)

show_image(style_image)
show_image(content_image)
show_image(input_image)
#MODEL
vgg = VGG().cuda()
loss = Loss().cuda()
for param in vgg.parameters():
    param.requires_grad = False

#OPTIMIZER
total_losses = []
learning_rate = 10
optimizer = optim.Adam([input_image], lr=learning_rate)
num_iterations = args.iterations
for i in range(num_iterations):
    optimizer.zero_grad()
    input_features = vgg(input_image)
    content_features = vgg(content_image)
    style_features = vgg(style_image)
    
    total_loss = loss(input_features, content_features, style_features)
    
    total_loss.backward()
    optimizer.step()    
    input_image.data.clamp_(0, 255)
    
    total_losses += [total_loss.data.cpu().numpy()[0]]
    if i % 10 == 0:
        print(i/num_iterations* 100, "%")
show_image(input_image)
        
graph_losses(total_losses)









   
