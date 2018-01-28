import torch
import torch.nn as nn
from torchvision import models

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg = models.vgg16(pretrained=True)
        module_list = list(vgg.features.modules())

        #relu
        self.relu_1_2 = nn.Sequential(*module_list[1:5])
        self.relu_2_2 = nn.Sequential(nn.AvgPool2d(3, 2,1), *module_list[6:10])
        self.relu_3_3 = nn.Sequential(nn.AvgPool2d(3, 2,1), *module_list[11:17])
        self.relu_4_3 = nn.Sequential(nn.AvgPool2d(3, 2,1), *module_list[18:24])
        
    def forward(self, x):
        relu_1_2 = self.relu_1_2(x)
        relu_2_2 = self.relu_2_2(relu_1_2)
        relu_3_3 = self.relu_3_3(relu_2_2)
        relu_4_3 = self.relu_4_3(relu_3_3)

        return relu_1_2, relu_2_2, relu_3_3, relu_4_3

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
        C, H, W = x.size()
        
        #transpose and calculate the gram matrices
        x = x.view(C, H*W)
        gram_x = torch.matmul(x, x.t()) / (H*W*C) * 1000

        y = y.view(C, H*W)
        gram_y = torch.matmul(y, y.t()) / (H*W*C) * 1000

        #take the L2 loss of the gram matrices
        style_loss = self.L2_loss(gram_x, gram_y)
        return style_loss

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.style_loss = StyleLoss()
        self.content_loss = ContentLoss()

    def forward(self, batch_input_features, batch_content_features, batch_style_features):
        R = batch_input_features[0].size(0)
        
        style_loss = 0
        content_loss = 0
        
        #style loss in relu_1_2, relu_2_2, relu_3_3, relu_4_3
        for input_feat, style_feat in zip(batch_input_features, batch_style_features):
            for inp, style in zip(input_feat, style_feat):
                style_loss += self.style_loss(inp, style)
        #content loss in relu_3_3
        for inp, content in zip(batch_input_features[2], batch_content_features[2]):
            content_loss += self.content_loss(inp, content)

        return (content_loss + style_loss) / R
