import numpy as np
from PIL import Image

def show_image(var_image):
    numpy_image = var_image.data[0].permute(1,2,0).cpu().numpy()

    im = Image.fromarray(numpy_image.astype(np.uint8))
    im.show()
