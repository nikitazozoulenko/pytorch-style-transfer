import os

from PIL import Image

style_root = "/hdd/Data/painters/train/train/"

for i, filename in enumerate(os.listdir(style_root)):
    im = Image.open(style_root+filename)
    print(i, " ", filename)
