import numpy as np
from PIL import Image

np.set_printoptions(threshold=np.inf)

im = Image.open("/home/yangxf/my_projects/multimodal_perception/data/flir_aligned/Annotations/FLIR_10206_mask.jpg")
a = np.array(im)
print(a)
