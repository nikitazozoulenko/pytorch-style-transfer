import numpy as np
from PIL import Image

def show_image(var_image):
    numpy_image = var_image.data[0].permute(1,2,0).cpu().numpy()*255

    im = Image.fromarray(numpy_image.astype(np.uint8))
    im.show()

def graph_losses(losses):
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(losses, "r", label="Loss")
    plt.legend(loc=1)
    plt.show()
