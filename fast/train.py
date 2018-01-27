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
from PIL import Image, ImageOps

image_transformer_network = ImageTransformerNetwork().cuda()
vgg = VGG()
loss = Loss()
