import torch
import torch.nn as nn
import numpy as np

class ImageTransformerNetwork(nn.Module):
    def __init__(self):
        super(ImageTransformerNetwork, self).__init__()
        self.expansion = 4
        
        self.in_channels = 6
        self.downsample = nn.Sequential(nn.BatchNorm2d(6),
                                        self._make_residual_layer(24, scaling = "downsample"),
                                        self._make_residual_layer(24, scaling = "same"),
                                        self._make_residual_layer(48, scaling = "downsample"))

        self.residual = nn.Sequential(self._make_residual_layer(48, scaling = "same"),
                                      self._make_residual_layer(48, scaling = "same"),
                                      self._make_residual_layer(48, scaling = "same"),
                                      self._make_residual_layer(48, scaling = "same"))
        
        self.upsample = nn.Sequential(self._make_residual_layer(24, scaling = "upsample"),
                                        self._make_residual_layer(24, scaling = "same"),
                                        self._make_residual_layer(12, scaling = "upsample"),
                                        self._make_residual_layer(12, scaling = "same"),
                                        nn.Conv2d(self.in_channels, 3, kernel_size=3, padding=1))
                                        
        self.tanh = nn.Tanh()

        
    def forward(self, content, style):
        x = torch.cat((content, style), dim = 1)
        x = self.downsample(x)
        x = self.residual(x)
        x = self.upsample(x)
        x = (self.tanh(x)+1)/2
        return x

    
    def _make_residual_layer(self, channels, scaling = "same"):
        bottleneck = Bottleneck(self.in_channels, channels, scaling, self.expansion)
        self.in_channels = channels*self.expansion
        return bottleneck

    
class Bottleneck(nn.Module):
    def __init__(self, in_channels, channels, scaling, expansion):
        super(Bottleneck, self).__init__()
        self.residual = False
        if in_channels == (channels*expansion) and scaling == "same":
            self.residual = True
        
        self.convbn0 = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(channels))

        if scaling == "downsample":
            self.convbn1 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride = 2, padding=1),
                                     nn.BatchNorm2d(channels))
        elif scaling == "upsample":
            self.convbn1 = nn.Sequential(nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
                                     nn.BatchNorm2d(channels))
        else: #scaling == "same"
            self.convbn1 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(channels))
        
        self.convbn2 = nn.Sequential(nn.Conv2d(channels, channels*expansion, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(channels*expansion))
        self.relu = nn.ReLU(inplace = True)

        
    def forward(self, x):
        res = x

        out = self.convbn0(x)
        out = self.convbn1(out)
        out = self.convbn2(out)

        if self.residual:
            out += res

        out = self.relu(out)
        return out
