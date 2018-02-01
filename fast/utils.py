import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def show_image(var_image):
    numpy_image = var_image.data[0].permute(1,2,0).cpu().numpy()*255

    im = Image.fromarray(numpy_image.astype(np.uint8))
    im.show()

def graph_losses(losses, x_indices, val_losses, val_x_indices):
    plt.figure(1)
    plt.plot(x_indices, losses, "r", label="Training Loss")
    plt.legend(loc=1)

    plt.figure(2)
    plt.plot(val_x_indices, val_losses, "g", label="Validation Loss")
    plt.legend(loc=1)

    plt.figure(3)
    plt.plot(x_indices, losses, "r", label="Training Loss")
    plt.plot(val_x_indices, val_losses, "g", label="Validation Loss")
    plt.legend(loc=1)
    plt.show()


