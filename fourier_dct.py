import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(image_path):
    return Image.open(image_path).convert('YCbCr')


def extract_luma(image):
    y, _, _ = image.split()
    return np.array(y)


def dft2d(image):
    """2D Discrete Fourier Transform."""
    h, w = image.shape
    f_transform = np.zeros((h, w), dtype=complex)
    for u in range(h):
        for v in range(w):
            sum_value = 0.0 + 0.0j
            for x in range(h):
                for y in range(w):
                    e = np.exp(-2j * np.pi * ((u * x / h) + (v * y / w)))
                    sum_value += image[x, y] * e
            f_transform[u, v] = sum_value
    return f_transform


def idft2d(f_transform):
    """2D Inverse Discrete Fourier Transform."""
    h, w = f_transform.shape
    image = np.zeros((h, w), dtype=complex)
    for x in range(h):
        for y in range(w):
            sum_value = 0.0 + 0.0j
            for u in range(h):
                for v in range(w):
                    e = np.exp(2j * np.pi * ((u * x / h) + (v * y / w)))
                    sum_value += f_transform[u, v] * e
            image[x, y] = sum_value / (h * w)
    return image


def apply_fourier_transform(luma):
    f_transform = dft2d(luma)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
    return magnitude_spectrum


def dct1d(vector):
    N = len(vector)
    result = np.zeros(N)
    factor = np.pi / (2 * N)
    for k in range(N):
        sum_value = 0.0
        for n in range(N):
            sum_value += vector[n] * np.cos((2 * n + 1) * k * factor)
        result[k] = sum_value * (np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N))
    return result


def idct1d(vector):
    N = len(vector)
    result = np.zeros(N)
    factor = np.pi / (2 * N)
    for n in range(N):
        sum_value = vector[0] * np.sqrt(1 / N)
        for k in range(1, N):
            sum_value += vector[k] * np.sqrt(2 / N) * np.cos((2 * n + 1) * k * factor)
        result[n] = sum_value
    return result


def dct2d(block):
    return np.array([dct1d(row) for row in block.T]).T


def idct2d(block):
    return np.array([idct1d(row) for row in block.T]).T


def block_process(image, block_size, process, q_matrix):
    h, w = image.shape
    processed_image = np.zeros_like(image, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i + block_size, j:j + block_size]
            processed_image[i:i + block_size, j:j + block_size] = process(block, q_matrix)
    return processed_image


def quantize(block, q_matrix):
    return np.round(block / q_matrix)


def dequantize(block, q_matrix):
    return block * q_matrix


def dct_quantization(block, q_matrix):
    dct_block = dct2d(block)
    quantized_block = quantize(dct_block, q_matrix)
    return quantized_block


def idct_dequantization(block, q_matrix):
    dequantized_block = dequantize(block, q_matrix)
    idct_block = idct2d(dequantized_block)
    return idct_block


def display_image(image, title='Image'):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()