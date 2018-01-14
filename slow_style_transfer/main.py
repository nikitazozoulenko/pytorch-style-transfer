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

size = 128
#IMAGES
input_image = Variable(torch.rand(1,3,size,size).cuda()*2-1, requires_grad = True)

style_image = Image.open("/hdd/Images/polyfoxstyle.png").resize((size,size))
style_image = (np.array(style_image)-127.5) / 127.5
style_image = Variable(torch.from_numpy(style_image).permute(2,0,1).unsqueeze(0)).float().cuda()

content_image = Image.open("/hdd/Images/lion.png").resize((size,size))
content_image = (np.array(content_image)-127.5) / 127.5
content_image = Variable(torch.from_numpy(content_image).permute(2,0,1).unsqueeze(0)).float().cuda()

#MODEL
vgg = VGG().cuda()
loss = Loss().cuda()
for param in vgg.parameters():
    param.requires_grad = False

#OPTIMIZER
total_losses = []
learning_rate = 0.01
optimizer = optim.Adam([input_image], lr=learning_rate)
num_iterations = 501
for i in range(num_iterations):
    optimizer.zero_grad()
    input_features = vgg(input_image)
    content_features = vgg(content_image)
    style_features = vgg(style_image)
    
    total_loss = loss(input_features, content_features, style_features)
    
    total_loss.backward()
    optimizer.step()    
    input_image.data.clamp_(-1, 1)
    
    total_losses += [total_loss.data.cpu().numpy()[0]]
    if i % 10 == 0:
        print(i)
    if i % 100 == 0:
        show_image(input_image)
    if i == 100000 or i == 1000000 or i == 15000000 or i == 500000:
            learning_rate /= (np.sqrt(10)**2)
            print("updated learning rate: current lr:", learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate


# optimizer = optim.LBFGS([input_image])

# def closure():
#     # correct the values of updated input image
#     optimizer.zero_grad()
#     input_features = vgg(input_image)
#     content_features = vgg(content_image)
#     style_features = vgg(style_image)
    
#     total_loss = loss(input_features, content_features, style_features)
#     total_loss.backward()
#     #total_losses += [total_loss.data.cpu().numpy()[0]]

#     input_image.data.clamp_(0, 1)
            
#     return total_loss

# #TRAINING
# show_image(style_image)
# show_image(content_image)
# num_iterations = 500
# for i in range(num_iterations):
#     optimizer.step(closure)
#     if i % 10 == 0:
#         print(i)
#     if i % 100 == 0:
#         show_image(input_image)
        
graph_losses(total_losses)









   
