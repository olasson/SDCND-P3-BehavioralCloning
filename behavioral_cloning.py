
import cv2
import numpy as np
import argparse
from os.path import join as path_join

# Custom imports
from code.misc import file_exists, folder_guard, folder_is_empty, parse_file_path, pick_triplets, is_file_type
from code.io import load_config, load_sim_log, load_images
from code.plots import plot_images
from code.prepare import prepare_data

FOLDER_DATA = './data'

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

    model_config = load_config(file_path_config)

    # Init values

    # Init flags

    flag_show_csv = is_file_type(file_path_show_images, '.csv')
    flag_show_pickled = is_file_type(file_path_show_images, '.p')

    flag_force_save = args.force_save

    # Folder setup

    folder_guard(FOLDER_DATA)

    
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
        print(INFO_PREFIX + 'Using config from: ' + file_path_config)

        file_path_driving_log = model_config["driving_log"]








