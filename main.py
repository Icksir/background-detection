from background_substraction import background_substractor_gmg
from count_whites import count_whites

def main():

    original_path = f'/media/icksir/Ricky/video_dataset/censored_driver/camion_movimiento_1_censored'
    gmg_output_directory = f'/media/icksir/Ricky/video_dataset/mask_gmg/camion_movimiento_1_mask'

    background_substractor_gmg(original_path, gmg_output_directory)
    mean = count_whites(gmg_output_directory)

    if mean > 85000:
        print('Movimiento')
    else:
        print('Detenido')

    print("-" * 10)    


if __name__ == '__main__':
    main()