import os
from glob import glob
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

# MOG2 es utilizado para cuando hay ambientes con varias sombras
def background_substractor_mog2():

    state = 'detenido'

    with open('train_set_' + state +'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["path", "state"]

        writer.writerow(field)

        for i in tqdm(range(1, 13), desc='carpeta'):

            data_path = '/media/icksir/Ricky/video_dataset/censored_driver/camion_'+ state + '_'+ str(i) +'_censored'
            image_paths = sorted(glob(f"{data_path}/*.jpg"))
            image_paths = sorted_alphanumeric(image_paths)

            # Directorio de salida para las imágenes JPEG
            output_directory = '/media/icksir/Ricky/video_dataset/mask_mog2/camion_' + state + '_'+ str(i) +'_mask'
            os.makedirs(output_directory, exist_ok=True)

            for j in tqdm(range(10, len(image_paths)), desc="mask"): 

                sub_type = 'MOG2' # 'KNN' # 'MOG2'
                if sub_type == "MOG2":
                    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=1, detectShadows=True)
                    backSub.setShadowThreshold(0.5)
                else:
                    backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=5, detectShadows=False)

                for img_path in image_paths[i-10:i]:
                    image = cv2.imread(img_path)

                    fg_mask = backSub.apply(image)

                path_saved = output_directory + '/frame_' + str(j) +'.jpg'
                cv2.imwrite(path_saved, fg_mask)

                writer.writerow([path_saved, state])

# GMG es utilizado en ambientes con luminosidad variable
def background_substractor_gmg(origin_path, output_directory):

    '''
    background_substractor_gmg()
        origin_path: path a carpeta donde
    '''

    image_paths = sorted(glob(f"{origin_path}/*.jpg"))
    image_paths = sorted_alphanumeric(image_paths)

    # Directorio de salida para las imágenes .jpg
    os.makedirs(output_directory, exist_ok=True)

    backSub = cv2.bgsegm.createBackgroundSubtractorGMG(decisionThreshold=0.62, initializationFrames=1)

    counter = 0
    for img_path in tqdm(image_paths, desc="mask"): 

        image = cv2.imread(img_path)

        fg_mask = backSub.apply(image)

        path_saved = output_directory + '/frame_' + str(counter) +'.jpg'
        counter += 1
        
        cv2.imwrite(path_saved, fg_mask)
                

# Solo para probar
def testing_changing_parameters():

    state = ['movimiento', 'detenido']

    for i in state:

        data_path = '/media/icksir/Ricky/video_dataset/censored_driver/camion_'+ i + '_1_censored'
        image_paths = sorted(glob(f"{data_path}/*.jpg"))
        image_paths = sorted_alphanumeric(image_paths)

        # Directorio de salida para las imágenes JPEG
        output_directory = '/media/icksir/Ricky/video_dataset/mask_testing/gmg/' + i
        os.makedirs(output_directory, exist_ok=True)

        sub_type = 'GMG' # 'KNN' # 'MOG2'
        if sub_type == "MOG2":
            backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=1, detectShadows=True)
            backSub.setShadowThreshold(0.5)

        elif sub_type == 'KNN':
            backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=5, detectShadows=False)

        elif sub_type == 'GMG':
            backSub = cv2.bgsegm.createBackgroundSubtractorGMG(decisionThreshold=0.62, initializationFrames=1)

        counter = 0
        for img_path in tqdm(image_paths, desc="mask"): 

            image = cv2.imread(img_path)

            fg_mask = backSub.apply(image)

            path_saved = output_directory + '/frame_' + str(counter) +'.jpg'
            counter += 1
            
            cv2.imwrite(path_saved, fg_mask)

if __name__ == '__main__':
    origin_path = '/media/icksir/Ricky/video_dataset/censored_driver/camion_movimiento_1_censored'
    output_directory = '/media/icksir/Ricky/video_dataset/mask_gmg/camion_movimiento_1_mask'
    background_substractor_gmg(origin_path, output_directory)