
import cv2
import numpy as np
import argparse
from os.path import join as path_join

# Custom imports
from code.misc import file_exists, folder_guard, folder_is_empty, parse_file_path, pick_triplets, is_file_type
from code.io import load_config, load_sim_log, load_images, save_pickled_data
from code.plots import plot_images, plot_distribution
from code.prepare import prepare_data

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

        if flag_show_csv:

            n_triplets = 3

            file_paths = load_sim_log(file_path_show_images)[1]
            file_paths = pick_triplets(file_paths, n_triplets, dtype = 'U128')

            images, file_names = load_images(file_paths)

            file_paths = None
        else:
            pass

        titles = n_triplets * ['Left', 'Center', 'Right']
        title_fig_window = 'images_driving_log'

        plot_images(images, titles, title_fig_window)


    if model_config is not None:

        model_config = load_config(file_path_config)

        print(INFO_PREFIX + 'Using config from: ' + file_path_config)

        file_path_driving_log = model_config["driving_log"]

        if not file_exists(file_path_driving_log):
            print(ERROR_PREFIX + 'The driving log located at: ' + file_path_driving_log + ' does not exist!')
            exit()

        file_path_model = model_config["model_path"]
        file_path_data_prepared = model_config["data_prepared"]

        if file_exists(file_path_model) and file_exists(file_path_data_prepared) and (not flag_force_save):
            print(WARNING_PREFIX + 'The model: ' + file_path_model + ' and dataset ' + file_path_data_prepared + ' already exists!')
            print(WARNING_PREFIX + 'Use --force_save to overwrite them!')
            exit()

        angle_correction = model_config["angle_correction"]
        angle_flatten = model_config["angle_flatten"]

        print(INFO_PREFIX + 'Previewing data!')

        angles, images = prepare_data(file_path_driving_log, angle_correction, angle_flatten, 
                                      augment = flag_data_augment, preview = flag_data_preview)

        if flag_data_preview:
            plot_distribution(angles, 
                              'Steering angle distribution', 'steering_angle_dist',
                              angle_correction, angle_flatten)

            # Exit when the preview flag is set, since 'images = None'
            print(INFO_PREFIX + 'Preview flag (--preview) set, exiting program!')
            exit()

        save_pickled_data(file_path_data_prepared, angles, images)










