from background_substraction import background_substractor_gmg
from count_whites import count_whites_ratio, count_whites_total 

def main():

    # Change paths to desire
    original_path = f'/media/icksir/Ricky/video_dataset/censored_driver/camion_movimiento_1_censored'
    gmg_output_directory = f'/media/icksir/Ricky/video_dataset/mask_gmg/camion_movimiento_1_mask'

    background_substractor_gmg(original_path, gmg_output_directory)
    mvm_perc, stp_perc = count_whites_ratio(gmg_output_directory)

    if mvm_perc > stp_perc:
        print("Se encuentra probablemente en movimiento")
    else:
        print("Se encuentra probablemente detenido")

    print("-" * 10)   


def testing_with_folders():

    '''
    testing_with_folders()
    ---
    Uso por deployment con carpetas locales, ignorar
    '''

    chances = ["movimiento", "detenido"]

    for chance in chances:
        for i in range(1,13):
            gmg_output_directory = f'/media/icksir/Ricky/video_dataset/mask_gmg/camion_{chance}_{i}_mask'

            # background_substractor_gmg(original_path, gmg_output_directory)
            mvm_perc, stp_perc = count_whites_ratio(gmg_output_directory, threshold=20000)

            print(f"Carpeta {chance} {i}")

            if mvm_perc > stp_perc:
                print("Se encuentra probablemente en movimiento")
            else:
                print("Se encuentra probablemente detenido")

            print("-" * 10)  


if __name__ == '__main__':
    main()