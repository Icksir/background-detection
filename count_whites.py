import cv2
import numpy as np
from background_substraction import sorted_alphanumeric
from glob import glob

# Lectura de archivos blancos
def count_whites(origin_path):

    image_paths = sorted(glob(f"{origin_path}/*.jpg"))
    image_paths = sorted_alphanumeric(image_paths)

    white_pixels = np.arange(len(image_paths))

    idx = 0
    for image in image_paths:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        n_white_pix = np.sum(img == 255)
        white_pixels[idx] = n_white_pix
        idx += 1

    print(f"Media de píxeles contados: {white_pixels.mean()}")

    return white_pixels.mean()

if __name__ == '__main__':
    origin_path = '/media/icksir/Ricky/video_dataset/mask_gmg/camion_movimiento_1_mask'
    count_whites(origin_path)