
import cv2
import numpy as np
import argparse
from os.path import join as path_join

# Custom imports
from code.misc import file_exists, folder_guard, folder_is_empty, parse_file_path, pick_triplets_1D, pick_triplets_images, is_file_type
from code.io import load_config, load_sim_log, load_images, save_pickled_data, load_pickled_data
from code.plots import plot_images, plot_distribution, plot_model_history
from code.prepare import prepare_data
from code.model import train_model, save_model, load_model

FOLDER_DATA = './data'
FOLDER_MODELS = './models'

INFO_PREFIX = 'INFO_MAIN: '
WARNING_PREFIX = 'WARNING_MAIN: '
ERROR_PREFIX = 'ERROR_MAIN: '

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description = 'Behavioral Cloning')

    # Show


    parser.add_argument(
        '--show_images',
        type = str,
        nargs = '?',
        default = '',
        help = 'Show a set of images from the .csv file or a pickled (.p) file.',
    )

    parser.add_argument(
        '--n_triplets',
        type = int,
        default = 3,
        help = 'The number of triplets (left cam, center cam, right cam) in the image plot.'
    )


    # Config

    parser.add_argument(
        '--model_config',
        type = str,
        nargs = '?',    
        default = None,
        help = 'Path to a .json file containing project config.',
    )

    parser.add_argument(
        '--augment',
        action = 'store_true',
        help = 'If enabled, data will be augmented (flipped, translated and brightness correction).'
    )

    parser.add_argument(
        '--preview',
        action = 'store_true',
        help = 'If enabled, the user can preview the steering angle distribution without processing the images.'
    )

    # Misc

    parser.add_argument(
        '--force_save',
        action = 'store_true',
        help = 'If enabled, permits overwriting existing data.'
    )

    args = parser.parse_args()

    # Init paths

    file_path_config = args.model_config

    file_path_show_images = args.show_images

    # Init config 

    if (file_path_config is not None) and not file_exists(file_path_config):
        print(ERROR_PREFIX + 'The model config located at: ' + file_path_config + ' does not exist!')
        exit()

    model_config = load_config(file_path_config)


    # Init values

    # Init flags

    flag_show_csv = is_file_type(file_path_show_images, '.csv')
    flag_show_pickled = is_file_type(file_path_show_images, '.p')

    flag_data_preview = args.preview
    flag_data_augment = args.augment

    flag_force_save = args.force_save

    # Folder setup

    folder_guard(FOLDER_DATA)
    folder_guard(FOLDER_MODELS)

    
    if flag_show_csv or flag_show_pickled:

        print(INFO_PREFIX + 'Showing images from: ' + file_path_show_images)

        n_triplets = args.n_triplets

        if flag_show_csv:

            file_paths = load_sim_log(file_path_show_images)[1]
            file_paths = pick_triplets_1D(file_paths, n_triplets, dtype = 'U128')

            images, _ = load_images(file_paths)

            file_paths = None

        else:
            _, images = load_pickled_data(file_path_show_images)
            images = pick_triplets_images(images, n_triplets)


        titles = n_triplets * ['Left', 'Center', 'Right']
        
        plot_images(images, titles, title_fig_window = file_path_show_images)


    if model_config is not None:

        file_path_driving_log = model_config["driving_log"]

        if not file_exists(file_path_driving_log):
            print(ERROR_PREFIX + 'The driving log located at: ' + file_path_driving_log + ' does not exist!')
            exit()

        
        file_path_data_prepared = model_config["data_prepared"]



        print(INFO_PREFIX + 'Using config from: ' + file_path_config)

        angle_correction = model_config["angle_correction"]
        angle_flatten = model_config["angle_flatten"]

        if not file_exists(file_path_data_prepared):

            print(INFO_PREFIX + 'Preparing data!')

            angles, images = prepare_data(file_path_driving_log, angle_correction, angle_flatten, 
                                          augment = flag_data_augment, preview = flag_data_preview)
            
            if images is not None:

                print(INFO_PREFIX + 'Saving preparing data at: ' + file_path_data_prepared)

                save_pickled_data(file_path_data_prepared, angles, images)

        else:
            
            print(INFO_PREFIX + 'Loading prepared data located at: ' + file_path_data_prepared)

            angles, images = load_pickled_data(file_path_data_prepared)

        if flag_data_preview:
            
            print(INFO_PREFIX + 'Previewing data!')

            plot_distribution(angles, 
                              'Steering angle distribution', 'steering_angle_dist',
                              angle_correction, angle_flatten)

            print(INFO_PREFIX + 'Preview flag (--preview) set, exiting program!')
            exit()

        lrn_rate = model_config["lrn_rate"]
        batch_size = model_config["batch_size"]
        n_max_epochs = model_config["n_max_epochs"]

        file_path_model = model_config["model_path"]

        if not file_exists(file_path_model):

            print(INFO_PREFIX + 'Training model: ' + file_path_model)
            model, history = train_model(images, angles, lrn_rate, batch_size, n_max_epochs)

            print(INFO_PREFIX + 'Saving model: ' + file_path_model)
            save_model(file_path_model, model)

            model_name = parse_file_path(file_path_model)[1]
            model_name = model_name[:len(model_name) - len('.h5')]
        
            file_path_save = path_join(FOLDER_MODELS, model_name + '.png')
            plot_model_history(history, model_name, lrn_rate, batch_size, n_max_epochs, file_path_save)

        else:
            print(INFO_PREFIX + 'Loading model: ' + file_path_model)
            model = load_model(file_path_model)










