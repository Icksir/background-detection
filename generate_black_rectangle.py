import cv2
import numpy as np
from tqdm import tqdm
import re
from glob import glob
import os
from background_substraction import sorted_alphanumeric

state = 'detenido'

def generate_black_rectangle(data_path, output_directory):

    for i in tqdm(range(1, 13), desc='carpeta'):

        data_path = '/media/icksir/Ricky/video_dataset/camion_'+ state + '_'+ str(i) +'_frames'
        image_paths = sorted(glob(f"{data_path}/*.jpg"))
        image_paths = sorted_alphanumeric(image_paths)

        output_directory = '/media/icksir/Ricky/video_dataset/censored_driver/camion_' + state + '_'+ str(i) +'_censored'
        os.makedirs(output_directory, exist_ok=True)

        # Set the dimensions and position of the rectangle
        rect_x, rect_y = 300, 210
        rect_width, rect_height = 370, 730

        # Set the color (black in BGR)
        black = (0, 0, 0)

        for j in tqdm(range(len(image_paths)), desc="censoring"):

            image = cv2.imread(image_paths[j])

            cv2.rectangle(image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), black, thickness=cv2.FILLED)

            # Save the modified image
            cv2.imwrite(output_directory + "/frame" + str(j) +".jpg", image)

if __name__ == '__main__':

    data_path = '/media/icksir/Ricky/video_dataset/camion_movimiento_1_frames'
    output_directory = '/media/icksir/Ricky/video_dataset/censored_driver/camion_movimiento_1_censored'
    generate_black_rectangle()
        