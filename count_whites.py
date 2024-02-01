import cv2
import numpy as np
from background_substraction import sorted_alphanumeric
from glob import glob

def count_whites_total(origin_path: str):

    """
    count_whites_total()
    ---
    Identifica píxeles blancos de cada frame y retorna la media de los píxeles blancos de todos
    los frames ubicados en `origin_path`. 
    
    Params:
    * `origin_path:` path de la carpeta que contiene los frames-máscaras jpg

    Output:
    * Media de píxeles blancos del video
    """

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

def count_whites_ratio(origin_path: str, threshold:int = 85000):

    """
    count_whites_ratio()
    ---
    Identifica píxeles blancos de cada frame, y retorna el porcentaje de frames en movimiento y de frames detenidos
    dado un threshold
    
    Params:
    * `origin_path`: path de la carpeta que contiene los frames-máscaras jpg
    * `threshold`: valor int

    Output:
    * `(mvm_perc, stp_perc)`: porcentaje en movimiento respecto a frames totales, porcentaje detenido respecto a frames totales
    """

    image_paths = sorted(glob(f"{origin_path}/*.jpg"))
    image_paths = sorted_alphanumeric(image_paths)

    mvm = 0
    stp = 0

    for image in image_paths:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        n_white_pix = np.sum(img == 255)

        if n_white_pix > threshold:
            mvm += 1
        else:
            stp += 1

    mvm_perc = mvm / len(image_paths) * 100
    stp_perc = stp / len(image_paths) * 100

    print(f"La imagen consiste en {int(mvm_perc)}% de frames probablemente en movimiento y {int(stp_perc)}% de frames probablemente detenido")

    return mvm_perc, stp_perc

if __name__ == '__main__':
    origin_path = '/media/icksir/Ricky/video_dataset/mask_gmg/camion_movimiento_8_mask'
    count_whites_ratio(origin_path, threshold=30000)