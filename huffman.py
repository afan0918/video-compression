import heapq

import numpy as np


def calculate_frequencies(image_pixels):
    frequencies = np.bincount(image_pixels.flatten(), minlength=16)
    return frequencies


def build_huffman_tree(frequencies):
    heap = [[weight, [symbol, ""]] for symbol, weight in enumerate(frequencies)]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return heap[0][1:]


def huffman_encoding(image_pixels, huffman_dict=None):
    if huffman_dict == None:
        frequencies = calculate_frequencies(image_pixels)
        print(frequencies)
        huffman_tree = build_huffman_tree(frequencies)
        huffman_dict = {symbol: huffman_code for symbol, huffman_code in huffman_tree}
    encoded_image = ''.join(huffman_dict[pixel] for pixel in image_pixels)
    return encoded_image, huffman_dict


def huffman_decoding(encoded_image, huffman_dict):
    decoded_image = ""
    code = ""
    for bit in encoded_image:
        code += bit
        if code in huffman_dict.values():
            symbol = list(huffman_dict.keys())[list(huffman_dict.values()).index(code)]
            decoded_image += str(symbol) + " "
            code = ""
    return decoded_image
