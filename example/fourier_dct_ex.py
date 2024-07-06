from fourier_dct import *


def main():
    q_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    image_path = 'img/foreman_qcif_0_rgb.bmp'
    image = load_image(image_path)
    luma = extract_luma(image)

    print("1. Fourier Transform")
    magnitude_spectrum = apply_fourier_transform(luma)
    display_image(magnitude_spectrum, title='Fourier Transform')

    print("2. DCT")
    dct_quantized = block_process(luma, 8, dct_quantization, q_matrix)

    # IDCT
    decoded_luma = block_process(dct_quantized, 8, idct_dequantization, q_matrix)
    display_image(decoded_luma, title='Decoded Frame')


if __name__ == "__main__":
    main()