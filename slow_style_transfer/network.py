import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn import Parameter
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg = models.vgg16(pretrained=True)
        module_list = list(vgg.features.modules())
        
        # vgg16_bn
        # self.relu_1_2 = nn.Sequential(*module_list[1:7])
        # self.relu_2_2 = nn.Sequential(nn.AvgPool2d(3, 2,1), *module_list[8:14])
        # self.relu_3_3 = nn.Sequential(nn.AvgPool2d(3, 2,1), *module_list[15:24])
        # self.relu_4_3 = nn.Sequential(nn.AvgPool2d(3, 2,1), *module_list[25:34])
        # self.relu_5_3 = nn.Sequential(nn.AvgPool2d(3, 2,1), *module_list[35:44])
        
        self.relu_1_2 = nn.Sequential(*module_list[1:5])
        self.relu_2_2 = nn.Sequential(nn.AvgPool2d(3, 2,1), *module_list[6:10])
        self.relu_3_3 = nn.Sequential(nn.AvgPool2d(3, 2,1), *module_list[11:17])
        self.relu_4_3 = nn.Sequential(nn.AvgPool2d(3, 2,1), *module_list[18:24])
        self.relu_5_3 = nn.Sequential(nn.AvgPool2d(3, 2,1), *module_list[25:31])
        
    def forward(self, x):
        relu_1_2 = self.relu_1_2(x)
        relu_2_2 = self.relu_2_2(relu_1_2)
        relu_3_3 = self.relu_3_3(relu_2_2)
        relu_4_3 = self.relu_4_3(relu_3_3)
        relu_5_3 = self.relu_5_3(relu_4_3)

        return relu_1_2, relu_2_2, relu_3_3, relu_4_3, relu_5_3

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.L2_loss = nn.MSELoss(size_average=True, reduce=True)
        
    def forward(self, x, y):
        #take the L2 loss of both tensors
        content_loss = self.L2_loss(x, y)
        return content_loss

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.L2_loss = nn.MSELoss(size_average=True, reduce=True)
        
    def forward(self, x, y):
        R, C, H, W = x.size()
        
        #transpose and calculate the gram matrices
        x = x.view(C, H*W)
        gram_x = torch.matmul(x, x.t())

        y = y.view(C, H*W)
        gram_y = torch.matmul(y, y.t())

        #take the L2 loss of the gram matrices
        style_loss = self.L2_loss(gram_x, gram_y) / (H*H*W*W*4)
        return style_loss

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.style_loss = StyleLoss()
        self.content_loss = ContentLoss()
        
    def forward(self, input_features, content_features, style_features):
        style = 0
        content = 0
        alpha = 1
        beta = 1000

        #style loss in relu_1_2, relu_2_2, relu_3_3, relu_4_3, relu_5_3
        #content loss in relu_3_3
        for input_feat, style_feat in zip(input_features, style_features):
            style += self.style_loss(input_feat, style_feat) / 5
        content += self.content_loss(input_features[2], content_features[2])

        return alpha*content + beta*style
