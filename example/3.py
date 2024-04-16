import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from color_conversion import rgb2yuv709, subsample_cb_cr, yuv2rgb709
from huffman import huffman_encoding

image_paths = ["foreman_qcif_0_rgb.bmp", "foreman_qcif_1_rgb.bmp", "foreman_qcif_2_rgb.bmp"]
ycbcr_images = []

# 處理每個圖像
for path in image_paths:
    rgb_img = Image.open("img/" + path)
    ycbcr_array = rgb2yuv709(rgb_img)
    ycbcr_array.astype(np.uint8)
    y = ycbcr_array[:, :, 0]
    cb = ycbcr_array[:, :, 1]
    cr = ycbcr_array[:, :, 2]
    cb_subsampled, cr_subsampled = subsample_cb_cr(cb, cr, 1, 1)
    ycbcr_images.append((y, cb_subsampled, cr_subsampled))

total_pixel = []
for ycbcr_frame in ycbcr_images:
    y, cb, cr = ycbcr_frame
    ycbcr = np.concatenate((y.flatten(), cb.flatten(), cr.flatten()))
    quantized_pixels = np.floor_divide(ycbcr, 16)
    total_pixel.append(quantized_pixels)
    # print(quantized_pixels)

total_pixel = np.concatenate((total_pixel[0], total_pixel[1], total_pixel[2])).astype(np.uint8)
print(total_pixel)
_, huffman_dict = huffman_encoding(total_pixel)
print(huffman_dict)

reverse_huffman_dict = dict()
for key, value in huffman_dict.items():
    reverse_huffman_dict[value] = key
print(reverse_huffman_dict)

binary_string = ""
h = 144
w = 176
for ycbcr_frame in ycbcr_images:
    y, cb, cr = ycbcr_frame
    y //= 16
    cb //= 16
    cr //= 16

    for i in range(h):
        for j in range(w):
            binary_string += huffman_dict[y[i][j]]

    for i in range(h // 2):
        for j in range(w // 2):
            binary_string += huffman_dict[cb[i][j]]

    for i in range(h // 2):
        for j in range(w // 2):
            binary_string += huffman_dict[cr[i][j]]

# 寫檔
with open('ans/3.yuv', 'wb') as f:
    byte_value = int(binary_string, 2).to_bytes((len(binary_string) + 7) // 8, 'big')
    f.write(byte_value)

# 讀檔
with open('ans/3.yuv', 'rb') as f:
    byte_value = f.read()
    binary_string = ''.join(format(byte, '08b') for byte in byte_value)

# print(binary_string)

tmp = ""
for _ in range(3):  # 三張圖片，我菜雞哇哇
    w = 176
    h = 144

    y = np.zeros((h, w), dtype=np.uint8)
    u = np.zeros((h // 2, w // 2), dtype=np.uint8)
    v = np.zeros((h // 2, w // 2), dtype=np.uint8)

    # y
    for i in range(h):
        for j in range(w):
            if len(tmp) < 1024:
                # readbytes = f.read(4)
                readbytes = binary_string[:1024]
                if len(binary_string) > 1024:
                    binary_string = binary_string[1024:]
                tmp += readbytes
                # print(tmp)
            for idx in range(2, 12):
                if reverse_huffman_dict.get(tmp[:idx], -1) == -1:
                    continue
                y[i][j] = reverse_huffman_dict.get(tmp[:idx], -1)
                tmp = tmp[idx:]
                break

    y *= 16
    y += 8  # 取中間值

    # u
    for i in range(h // 2):
        for j in range(w // 2):
            if len(tmp) < 1024:
                # readbytes = f.read(4)
                readbytes = binary_string[:1024]
                if len(binary_string) > 1024:
                    binary_string = binary_string[1024:]
                tmp += readbytes
                # print(tmp)
            for idx in range(2, 12):
                if reverse_huffman_dict.get(tmp[:idx], -1) == -1:
                    continue
                u[i][j] = reverse_huffman_dict.get(tmp[:idx], -1)
                tmp = tmp[idx:]
                break

    u *= 16
    u += 8  # 取中間值

    # v
    for i in range(h // 2):
        for j in range(w // 2):
            if len(tmp) < 1024:
                # readbytes = f.read(4)
                readbytes = binary_string[:1024]
                if len(binary_string) > 1024:
                    binary_string = binary_string[1024:]
                tmp += readbytes
                # print(tmp)
            for idx in range(2, 12):
                if reverse_huffman_dict.get(tmp[:idx], -1) == -1:
                    continue
                v[i][j] = reverse_huffman_dict.get(tmp[:idx], -1)
                tmp = tmp[idx:]
                break
        # print(i)

    v *= 16
    v += 8  # 取中間值

    ycbcr_img = yuv2rgb709(y, u, v)
    ycbcr_img = np.array(ycbcr_img)
    plt.subplot(1, 3, _ + 1)
    plt.imshow(ycbcr_img)
    plt.axis('off')

plt.tight_layout()
plt.show()


def binary_string_to_bytes(binary_string):
    """
    將二進位字串轉換為位元組（bytes）
    """
    integer_value = int(binary_string, 2)
    byte_value = integer_value.to_bytes((integer_value.bit_length() + 7) // 8, 'big')
    return byte_value


def bytes_to_binary_string(byte_value):
    """
    將位元組轉換為二進位字串
    """
    integer_value = int.from_bytes(byte_value, 'big')
    binary_string = bin(integer_value)[2:]
    return binary_string

# tmp = ""
# with open("3.yuv", "wb") as f:
#     for ycbcr_frame in ycbcr_images:
#         y, cb, cr = ycbcr_frame
#         ycbcr = np.concatenate((y.flatten(), cb.flatten(), cr.flatten()))
#         quantized_pixels = np.floor_divide(ycbcr, 16)
#         for quantized_pixel in quantized_pixels:
#             tmp += huffman_dict[quantized_pixel]
#             while len(tmp) >= 8:
#                 f.write(binary_string_to_bytes(tmp[:8]))
#                 tmp = tmp[8:]
#     f.write(binary_string_to_bytes(tmp[:8]))  # 把剩下的寫檔
#
# plt.figure(figsize=(10, 6))
#
# with open("3.yuv", "rb") as f:
#     tmp = ""
#     for _ in range(3):  # 三張圖片，我菜雞哇哇
#         w = 176
#         h = 144
#
#         y = np.zeros((h, w), dtype=np.uint8)
#         u = np.zeros((h // 2, w // 2), dtype=np.uint8)
#         v = np.zeros((h // 2, w // 2), dtype=np.uint8)
#
#         # y
#         for i in range(h):
#             for j in range(w):
#                 if len(tmp) < 1024:
#                     readbytes = f.read(4)
#                     tmp += bin(bytes_to_long(readbytes))[2:]
#                     print(tmp)
#                 for idx in range(2, 12):
#                     if reverse_huffman_dict.get(tmp[:idx], -1) == -1:
#                         continue
#                     y[i][j] = reverse_huffman_dict.get(tmp[:idx], -1)
#                     tmp = tmp[idx:]
#                     break
#
#         y *= 16
#         y += 8  # 取中間值
#
#         # u
#         for i in range(h // 2):
#             for j in range(w // 2):
#                 if len(tmp) < 1024:
#                     readbytes = f.read(4)
#                     tmp += bin(bytes_to_long(readbytes))[2:]
#                     print(tmp)
#                 for idx in range(2, 12):
#                     if reverse_huffman_dict.get(tmp[:idx], -1) == -1:
#                         continue
#                     u[i][j] = reverse_huffman_dict.get(tmp[:idx], -1)
#                     tmp = tmp[idx:]
#                     break
#
#         u *= 16
#         u += 8  # 取中間值
#
#         # v
#         for i in range(h // 2):
#             for j in range(w // 2):
#                 if len(tmp) < 1024:
#                     readbytes = f.read(4)
#                     tmp += bin(bytes_to_long(readbytes))[2:]
#                     print(tmp)
#                 for idx in range(2, 12):
#                     if reverse_huffman_dict.get(tmp[:idx], -1) == -1:
#                         continue
#                     v[i][j] = reverse_huffman_dict.get(tmp[:idx], -1)
#                     tmp = tmp[idx:]
#                     break
#             print(i)
#
#         v *= 16
#         v += 8  # 取中間值
#
#         ycbcr_img = ycbcr_to_rgb(y, u, v)
#         ycbcr_img = np.array(ycbcr_img)
#         plt.subplot(1, 3, _ + 1)
#         plt.imshow(ycbcr_img)
#         plt.axis('off')
#
# plt.tight_layout()
# plt.show()
