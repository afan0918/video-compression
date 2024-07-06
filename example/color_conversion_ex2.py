from PIL import Image
import numpy as np

from color_conversion import rgb2yuv709, subsample_cb_cr

image_paths = ["foreman_qcif_0_rgb.bmp", "foreman_qcif_1_rgb.bmp", "foreman_qcif_2_rgb.bmp"]
ycbcr_images = []

for path in image_paths:
    rgb_img = Image.open("img/" + path)
    ycbcr_array = rgb2yuv709(rgb_img)
    y = ycbcr_array[:, :, 0]
    cb = ycbcr_array[:, :, 1]
    cr = ycbcr_array[:, :, 2]
    cb_subsampled, cr_subsampled = subsample_cb_cr(cb, cr)
    ycbcr_images.append((y, cb_subsampled, cr_subsampled))

# 保存到.yuv文件
with open("ans/2.yuv", "wb") as f:
    for ycbcr_frame in ycbcr_images:
        y, cb, cr = ycbcr_frame
        # 將Y分量寫入文件
        y_data = np.array(y)
        # print(len(y_data.tobytes()))
        f.write(y_data.tobytes())
        f.write(b'\n')
        # 將Cb分量寫入文件
        cb_data = np.array(cb)
        f.write(cb_data.tobytes())
        f.write(b'\n')
        # 將Cr分量寫入文件
        cr_data = np.array(cr)
        f.write(cr_data.tobytes())
        f.write(b'\n')
