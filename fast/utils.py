import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def show_image(var_image):
    numpy_image = var_image.data[0].permute(1,2,0).cpu().numpy()

    im = Image.fromarray(numpy_image.astype(np.uint8))
    im.show()

def graph_losses(losses):
    plt.plot(losses, "b", label="Total Loss")
    plt.legend(loc=1)
    plt.show()







