import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np

from network import VGG, Loss
from utils import show_image
from PIL import Image

import argparse

def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch Neural Style Implementation')
    parser.add_argument("--content", type=str, metavar="PATH", help="path to content image", required = True)
    parser.add_argument("--style", type=str, metavar="PATH", help="path to style image", required = True)
    parser.add_argument("--output", type=str, metavar="PATH", default = "output.png", help="path to output file")
    parser.add_argument("--width", type=int, metavar="INT", default = -1, help="width of the output image")
    parser.add_argument("--height", type=int, metavar="INT", default = -1, help="height of the output image")
    parser.add_argument("--iter", type=int, metavar="INT", default = 500, help="number of iterations of the neural style algorithm")
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    content_image, style_image, input_image = read_images(args, use_gpu)

    #MODEL
    vgg = VGG().cuda()
    loss = Loss().cuda()
    if use_gpu:
        vgg = VGG().cuda()
        loss = Loss().cuda()
    for param in vgg.parameters():
        param.requires_grad = False

    #OPTIMIZER
    learning_rate = 1
    optimizer = optim.Adam([input_image], lr=learning_rate)
    num_iterations = 600
    for i in range(num_iterations):
        optimizer.zero_grad()
        input_features = vgg(input_image)
        content_features = vgg(content_image)
        style_features = vgg(style_image)
    
        total_loss = loss(input_features, content_features, style_features)
    
        total_loss.backward()
        optimizer.step()    
        input_image.data.clamp_(0, 255)
        if i % 30 == 0:
            print(i/num_iterations* 100, "%")
    print("100.0 %")

    show_image(input_image)
    output = Image.fromarray(input_image.data.squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8))
    output.save(args.output)

def read_images(args, use_gpu):
    #IMAGES
    content_image = Image.open(args.content).convert("RGB")
    width, height = content_image.size
    if args.width > 1:
        width = args.width
    if args.height > 1:
        height = args.height
    content_image = content_image.resize((width, height))
    content_image = np.array(content_image)
    content_image = Variable(torch.from_numpy(content_image).permute(2,0,1).unsqueeze(0)).float()

    style_image = Image.open(args.style).convert("RGB").resize((width, height))
    style_image = (np.array(style_image))
    style_image = Variable(torch.from_numpy(style_image).permute(2,0,1).unsqueeze(0)).float()
        
    if use_gpu:
        content_image = content_image.cuda()
        style_image = style_image.cuda()

    input_image = Variable(content_image.data.clone(), requires_grad = True)

    return content_image, style_image, input_image

if __name__ == "__main__":
    main()




   
