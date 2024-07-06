import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(image_path):
    return Image.open(image_path).convert('YCbCr')


def extract_luma(image):
    y, _, _ = image.split()
    return np.array(y)


def calculate_sad(block1, block2):
    return np.sum(np.abs(block1 - block2))


def get_spiral_coords(radius):
    coords = []
    x, y = 0, 0
    dx, dy = 0, -1
    for _ in range((2 * radius + 1) ** 2):
        if (-radius < x <= radius) and (-radius < y <= radius):
            coords.append((x, y))
        if (x == y) or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
            dx, dy = -dy, dx
        x, y = x + dx, y + dy
    return coords


def full_search(reference_frame, current_frame, mb_size=16, search_range=16):
    height, width = current_frame.shape
    mv = np.zeros((height // mb_size, width // mb_size, 2), dtype=int)

    for i in range(0, height, mb_size):
        for j in range(0, width, mb_size):
            current_block = current_frame[i:i + mb_size, j:j + mb_size]
            min_sad = float('inf')
            best_mv = (0, 0)

            for x in range(-search_range, search_range + 1):
                for y in range(-search_range, search_range + 1):
                    ref_x, ref_y = i + x, j + y
                    if 0 <= ref_x <= height - mb_size and 0 <= ref_y <= width - mb_size:
                        ref_block = reference_frame[ref_x:ref_x + mb_size, ref_y:ref_y + mb_size]
                        sad = calculate_sad(current_block, ref_block)
                        if sad < min_sad:
                            min_sad = sad
                            best_mv = (x, y)

            mv[i // mb_size, j // mb_size] = best_mv

    return mv


def full_search_spiral(reference_frame, current_frame, mb_size=16, search_range=16):
    height, width = current_frame.shape
    mv = np.zeros((height // mb_size, width // mb_size, 2), dtype=int)
    spiral_coords = get_spiral_coords(search_range)

    for i in range(0, height, mb_size):
        for j in range(0, width, mb_size):
            min_sad = float('inf')
            best_mv = (0, 0)
            for dx, dy in spiral_coords:
                ref_x = i + dx
                ref_y = j + dy
                if ref_x < 0 or ref_y < 0 or ref_x + mb_size > height or ref_y + mb_size > width:
                    continue
                sad = calculate_sad(current_frame[i:i + mb_size, j:j + mb_size],
                                    reference_frame[ref_x:ref_x + mb_size, ref_y:ref_y + mb_size])
                if sad < min_sad:
                    min_sad = sad
                    best_mv = (dx, dy)
            mv[i // mb_size, j // mb_size] = best_mv
    return mv


def diamond_search(reference_frame, current_frame, mb_size=16, search_range=16):
    height, width = current_frame.shape
    mv = np.zeros((height // mb_size, width // mb_size, 2), dtype=int)

    LDSP = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    SDSP = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(0, height, mb_size):
        for j in range(0, width, mb_size):
            min_sad = float('inf')
            best_mv = (0, 0)
            center = (0, 0)

            while True:
                best_local_mv = None
                for dx, dy in LDSP:
                    ref_x, ref_y = i + center[0] + dx, j + center[1] + dy
                    if ref_x < 0 or ref_y < 0 or ref_x + mb_size > height or ref_y + mb_size > width:
                        continue
                    sad = calculate_sad(current_frame[i:i + mb_size, j:j + mb_size],
                                        reference_frame[ref_x:ref_x + mb_size, ref_y:ref_y + mb_size])
                    if sad < min_sad:
                        min_sad = sad
                        best_local_mv = (dx, dy)
                        best_mv = (center[0] + dx, center[1] + dy)

                if best_local_mv is None or best_local_mv == (0, 0):
                    break
                center = best_mv

            for dx, dy in SDSP:
                ref_x, ref_y = i + center[0] + dx, j + center[1] + dy
                if ref_x < 0 or ref_y < 0 or ref_x + mb_size > height or ref_y + mb_size > width:
                    continue
                sad = calculate_sad(current_frame[i:i + mb_size, j:j + mb_size],
                                    reference_frame[ref_x:ref_x + mb_size, ref_y:ref_y + mb_size])
                if sad < min_sad:
                    min_sad = sad
                    best_mv = (center[0] + dx, center[1] + dy)

            mv[i // mb_size, j // mb_size] = best_mv

    return mv


def intra_prediction(luma_frame, mb_size=16):
    height, width = luma_frame.shape
    modes = np.zeros((height // mb_size, width // mb_size), dtype=int)
    predicted_frame = np.zeros_like(luma_frame)

    for i in range(mb_size, height, mb_size):
        for j in range(mb_size, width, mb_size):
            block = luma_frame[i:i + mb_size, j:j + mb_size]
            predictions = {}

            if j >= mb_size:
                left_block = luma_frame[i:i + mb_size, j - mb_size:j]
                predictions[0] = np.tile(left_block[:, -1:], (1, mb_size))  # Vertical prediction

            if i >= mb_size:
                top_block = luma_frame[i - mb_size:i, j:j + mb_size]
                predictions[1] = np.tile(top_block[-1:, :], (mb_size, 1))  # Horizontal prediction

            if j >= mb_size and i >= mb_size:
                top_left_block = luma_frame[i - mb_size:i, j - mb_size:j]
                predictions[2] = np.tile(top_left_block[-1, -1], (mb_size, mb_size))  # DC prediction

            if j >= mb_size and i >= mb_size:
                predictions[4] = np.tile(luma_frame[i - mb_size:i, j - mb_size:j].mean(),
                                         (mb_size, mb_size))  # Plane prediction

            if predictions:
                best_mode = min(predictions, key=lambda mode: calculate_sad(block, predictions[mode]))
                modes[i // mb_size, j // mb_size] = best_mode
                predicted_frame[i:i + mb_size, j:j + mb_size] = predictions[best_mode]
            else:
                predicted_frame[i:i + mb_size, j:j + mb_size] = block

    return predicted_frame, modes


def visualize_motion_vectors(mv, mb_size=16):
    plt.figure()
    for i in range(mv.shape[0]):
        for j in range(mv.shape[1]):
            plt.arrow(j * mb_size + mb_size // 2, i * mb_size + mb_size // 2,
                      mv[i, j, 1], mv[i, j, 0], color='red', head_width=1)
    plt.gca().invert_yaxis()
    plt.show()


def visualize_modes(modes):
    plt.figure()
    plt.imshow(modes, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.show()
