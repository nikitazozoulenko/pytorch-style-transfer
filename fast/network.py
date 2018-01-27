import torch
import torch.nn as nn
import numpy as np

class ImageTransformerNetwork(nn.Module):
    def __init__(self):
        super(ImageTransformerNetwork, self).__init__()

        self.downsample = nn.Sequential(nn.BatchNorm2d(6),
                                        nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm2d(64),
                                        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(inplace=True))
        self.in_channels = 128

        residual = []
        for i in range(4):
            residual.append(self._make_residual_layer(128, stride = 1))
        self.residual = nn.Sequential(*res)
        
        self.residual_upsample = nn.Sequential(self._make_residual_layer(128, stride = 2, spatial_scaling = "upsample"),
                                               self._make_residual_layer(128, stride = 2, spatial_scaling = "upsample"),
                                               nn.Conv2d(self.in_channels, 3, kernel_size=1, stride=1))
        self.tanh = nn.Tanh()

    def forward(self, content, style):
        x = torch.cat((content, style), dim = 1)
        x = self.BN_input(x)
        x = self.downsample(x)
        x = self.residual(x)
        x = self.residual_upsample(x)
        x = (self.tanh(x)+1)*255
        return x

    
    def _make_residual_layer(self, channels, stride = 1, spatial_scaling = "same"):
        expansion = 4
        if self.in_channels != (channels*expansion):
            channel_scaling = nn.Sequential(nn.Conv2d(self.in_channels, channels*expansion, kernel_size=1),
                                            nn.BatchNorm2d(channels*expansion))
        bottleneck = Bottleneck(self.in_channels, channels, expansion, stride, channel_scaling, spatial_scaling)
        self.in_channels = channels*expansion
        return bottleneck

class Bottleneck(nn.Module):
    def __init__(self, in_channels, channels, expansion, stride, channel_scaling, spatial_scaling):
        super(Bottleneck, self).__init__()

        self.convbn0 = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(channels))
        convbn1_layers = []
        self.scaling = None
        if stride != 1 or spatial_scaling == "upsample":
            if spatial_scaling == "upsample":
                self.scaling = self.knn_upsample
                stride = 1
                convbn1_layers += [self.knn_upsample]
            else: #if spatial_scaling == "downsample":
                self.scaling = self.knn_downsample
        convbn1_layers += [nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1,  bias=False)]
        convbn1_layers += [nn.BatchNorm2d(channels)]
        self.convbn1 = nn.Sequential(*convbn1_layers)
        
        self.convbn2 = nn.Sequential(nn.Conv2d(channels, channels*expansion, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(channels*expansion))

    def forward(self, x):
        res = x

        out = self.convbn0(x)
        out = self.convbn1(x)
        out = self.convbn2(x)

        if self.scaling != None:
            res = self.scaling(res)

        out = self.relu(out+res)
        return out
