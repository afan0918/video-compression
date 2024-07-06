from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from color_conversion import rgb2yuv709, subsample_cb_cr, yuv2rgb709

# 載入圖像
image_path = "img/foreman_qcif_0_rgb.bmp"
rgb_img = Image.open(image_path)
rgb_array = np.array(rgb_img)

# 轉換為YCbCr色彩空間
ycbcr_array = rgb2yuv709(rgb_array)
y = ycbcr_array[:, :, 0]
cb = ycbcr_array[:, :, 1]
cr = ycbcr_array[:, :, 2]

# 對Cb和Cr分量進行取樣
cb_sampled, cr_sampled = subsample_cb_cr(cb, cr)

# 將取樣後的Cb和Cr分量重新組合
ycbcr_img = yuv2rgb709(y, cb_sampled, cr_sampled)

# Display original RGB image
plt.figure(figsize=(10, 6))
plt.subplot(2, 3, 1)
plt.title("Original RGB Image")
plt.imshow(rgb_img)
plt.axis('off')

ycbcr_img = np.array(ycbcr_img)
# Display Y component
plt.subplot(2, 3, 2)
plt.title("Y Component")
plt.imshow(y, cmap='gray')
plt.axis('off')

# Display Cb component
plt.subplot(2, 3, 3)
plt.title("Cb Component")
plt.imshow(cb_sampled, cmap='gray')
plt.axis('off')

# Display Cr component
plt.subplot(2, 3, 4)
plt.title("Cr Component")
plt.imshow(cr_sampled, cmap='gray')
plt.axis('off')

# Display subsampled RGB image
plt.subplot(2, 3, 5)
plt.title("Subsampled RGB Image (4:2:0)")
plt.imshow(ycbcr_img)
plt.axis('off')

plt.tight_layout()
plt.savefig('ans/1.png')
plt.show()
