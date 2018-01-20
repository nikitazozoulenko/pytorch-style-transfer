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

        self.relu_1_2 = nn.Sequential(*module_list[1:5])
        self.relu_2_2 = nn.Sequential(nn.AvgPool2d(3, 2,1), *module_list[6:10])
        self.relu_3_3 = nn.Sequential(nn.AvgPool2d(3, 2,1), *module_list[11:17])
        self.relu_4_3 = nn.Sequential(nn.AvgPool2d(3, 2,1), *module_list[18:24])

    def forward(self, x, features = "style"):
        relu_1_2 = self.relu_1_2(x)
        relu_2_2 = self.relu_2_2(relu_1_2)
        relu_3_3 = self.relu_3_3(relu_2_2)
        if features == "content":
            return relu3_3

        relu_4_3 = self.relu_4_3(relu_3_3)
        return relu_1_2, relu_2_2, relu_3_3, relu_4_3

class Bottleneck(nn.Module):
    def __init__(self, in_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels/4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels/4)
        self.conv2 = nn.Conv2d(in_channels/4, in_channels/4, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels/4)
        self.conv3 = nn.Conv2d(in_channels/4, in_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += res
        out = self.relu(out)

        return out

class ImageTransformerNetwork(nn.Module):
    def __init__(self):
        super(ImageTransformerNetwork, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.bn0 = nn.BatchNorm2d(3)
        self.conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 512, kernel_size=1)

        self.resBN = nn.BatchNorm2d(512)
        res = []
        for i in range(5):
            res.append(Bottleneck(512))
        self.res = nn.Sequential(*res)

        self.knn = nn.Upsample(scale_factor=2, mode="nearest")

        self.bn3 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 128, kernel_size=3, padding =1)

        self.bn4 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding =1)

        self.bn5 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 3, kernel_size=3, padding = 1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv0(x)
        x = self.relu(x)

        x = self.bn1(x)
        x = self.conv1(x)
        x = self.relu(x)

        x = self.bn2(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = self.resBN(x)
        x = self.res(x)

        x = self.bn3(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.knn(x)

        x = self.bn4(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.knn(x)

        x = self.bn5(x)
        x = self.conv5(x)
        x = (self.tanh(x)+1)*255

        return x

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
