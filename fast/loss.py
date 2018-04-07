import torch
import torch.nn as nn
from torchvision import models

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg = models.vgg16(pretrained=True)
        module_list = list(vgg.features.modules())

        pooling = nn.AvgPool2d(3, 2,1)
        #pooling = nn.MaxPool2d(3, 2,1)
        
        self.relu_1_1 = nn.Sequential(*module_list[1:3])
        self.relu_2_1 = nn.Sequential(*module_list[3:8])
        self.relu_3_1 = nn.Sequential(*module_list[8:13])
        self.relu_3_2 = nn.Sequential(*module_list[13:15])
        self.relu_4_1 = nn.Sequential(*module_list[15:20])
        self.relu_5_1 = nn.Sequential(*module_list[20:27])

       
    def forward(self, x, keys):
        out = {}
        out["1_1"] = self.relu_1_1(x)
        out["2_1"] = self.relu_2_1(out["1_1"])
        out["3_1"] = self.relu_3_1(out["2_1"])
        out["3_2"] = self.relu_3_2(out["3_1"])
        out["4_1"] = self.relu_4_1(out["3_2"])
        out["5_1"] = self.relu_5_1(out["4_1"])

        return [out[key] for key in keys]

    
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
        gram_x = torch.matmul(x, x.t()) / (C*H*W) * 1000 * 1.2
        y = y.view(C, H*W)
        gram_y = torch.matmul(y, y.t()) / (C*H*W) * 1000 * 1.2

        #take the L2 loss of the gram matrices
        style_loss = self.L2_loss(gram_x, gram_y)
        return style_loss

    
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.style_loss = StyleLoss()
        self.content_loss = ContentLoss()

        
    def forward(self, input_features, input_3_2, content_3_2, style_features):
        style = 0
        content = 0
        
        #style loss in relu_1_1, relu_2_1, relu_3_1, relu_4_1, relu_5_1
        #content loss in relu_3_3
        for input_feat, style_feat in zip(input_features, style_features):
            style += self.style_loss(input_feat, style_feat)
        content += self.content_loss(input_3_2, content_3_2)

        return content + style
