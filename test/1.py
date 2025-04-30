from PIL import Image
from multimodal_perception.utils import data_parse

rgb_train, thremal_train, mask_train, rgb_val, thremal_val, mask_val = data_parse(
    "/home/yangxf/my_projects/multimodal_perception/data/flir_aligned"
)

rgb_im = rgb_val[1000]
im = Image.fromarray(rgb_im)
im.show()

thremal_im = thremal_val[1000]
im = Image.fromarray(thremal_im)
im.show()

mask = mask_val[1000]
im = Image.fromarray(mask)
im.show()
