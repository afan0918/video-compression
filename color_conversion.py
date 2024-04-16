import numpy as np
from PIL import Image


def rgb2yuv601(rgb_array):
    # RGB轉換矩陣（ITU-R BT.601）
    rgb_to_ycbcr_matrix = np.array([[0.299, 0.587, 0.114],
                                    [-0.168736, -0.331264, 0.5],
                                    [0.5, -0.418688, -0.081312]])

    # 對每個像素應用轉換矩陣
    ycbcr_array = np.dot(rgb_array, rgb_to_ycbcr_matrix.T)

    ycbcr_array[:, :, 1:] += 128.0

    # 將範圍限制在0到255之間
    ycbcr_array = np.clip(ycbcr_array, 0, 255)

    # 將結果轉換為無符號8位整數
    ycbcr_array = ycbcr_array.astype(np.uint8)

    return ycbcr_array


def rgb2yuv709(rgb_array):
    # RGB轉換矩陣（ITU-R BT.709）
    rgb_to_ycbcr_matrix = np.array([[0.2126, 0.7152, 0.0722],
                                    [-0.115, -0.386, 0.5],
                                    [0.5, -0.454, -0.046]])

    # 對每個像素應用轉換矩陣
    ycbcr_array = np.dot(rgb_array, rgb_to_ycbcr_matrix.T)

    ycbcr_array[:, :, 0] += 16.0
    ycbcr_array[:, :, 1:] += 128.0

    # 將範圍限制在0到255之間
    ycbcr_array = np.clip(ycbcr_array, 0, 255)

    return ycbcr_array


def yuv2rgb601(y, cb_sampled, cr_sampled):
    # RGB轉換矩陣（ITU-R BT.709）
    ycbcr_to_rgb_matrix = np.array([[1.0, 0.0, 1.402],
                                    [1.0, -0.344136, -0.714136],
                                    [1.0, 1.772, 0.0]])

    cb_sampled -= 128.0
    cr_sampled -= 128.0

    # 將 Cb 和 Cr 調整回原始大小
    cb_resized = np.kron(cb_sampled, np.ones((2, 2)))
    cr_resized = np.kron(cr_sampled, np.ones((2, 2)))

    # 組合 Y、Cr、Cb 向量
    ycbcr_array = np.stack((y, cb_resized, cr_resized), axis=-1)

    # 對每個像素做逆矩陣處理
    rgb_array = np.dot(ycbcr_array, ycbcr_to_rgb_matrix.T)

    # 將範圍限制在 0 到 255 之間
    rgb_array = np.clip(rgb_array, 0, 255)

    # 將結果轉換為無符號8位整數
    rgb_array = rgb_array.astype(np.uint8)

    return rgb_array


def yuv2rgb709(y, cb_sampled, cr_sampled):
    # RGB轉換矩陣（ITU-R BT.709）
    ycbcr_to_rgb_matrix = np.array([[1.0, 0.0, 1.5748],
                                    [1.0, -0.1873, -0.4681],
                                    [1.0, 1.8556, 0.0]])

    y = y.astype(np.float64)
    cb_sampled = cb_sampled.astype(np.float64)
    cr_sampled = cr_sampled.astype(np.float64)

    y -= 16.0
    cb_sampled -= 128.0
    cr_sampled -= 128.0

    # 將 Cb 和 Cr 調整回原始大小
    cb_resized = np.kron(cb_sampled, np.ones((2, 2)))
    cr_resized = np.kron(cr_sampled, np.ones((2, 2)))

    # 組合 Y、Cr、Cb 向量
    ycbcr_array = np.stack((y, cb_resized, cr_resized), axis=-1)

    # 對每個像素做逆矩陣處理
    rgb_array = np.dot(ycbcr_array, ycbcr_to_rgb_matrix.T)

    # 將範圍限制在 0 到 255 之間
    rgb_array = np.clip(rgb_array, 0, 255)

    # 將結果轉換為無符號8位整數
    rgb_array = rgb_array.astype(np.uint8)

    return rgb_array


def subsample_cb_cr(cb, cr, k_cb=1, k_cr=1):
    """
    對 Cb 和 Cr 陣列進行 4:2:0 採樣，並根據 k_cb 和 k_cr 進行亮度採樣
    """
    cb_sampled = cb[::2, ::2]
    cr_sampled = cr[::2, ::2]

    if k_cb != 1:
        cb_sampled = np.array(cb_sampled) * k_cb
    if k_cr != 1:
        cr_sampled = np.array(cr_sampled) * k_cr

    return cb_sampled, cr_sampled
