
import cv2
import numpy as np
import argparse
from os.path import join as path_join

# Custom imports
from code.misc import file_exists, folder_guard, folder_is_empty, parse_file_path
from code.io import load_config, glob_images
from code.plots import plot_images
from code.prepare import prepare_data

FOLDER_DATA = './data'

INFO_PREFIX = 'INFO_MAIN: '
WARNING_PREFIX = 'WARNING_MAIN: '
ERROR_PREFIX = 'ERROR_MAIN: '

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description = 'Behavioral Cloning')

    # Data

    parser.add_argument(
        '--driving_log',
        type = str,
        nargs = '?',
        default = '',
        help = 'File path to a .csv file containing the driving log from the simulator.',
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

    # Init config

    model_config = load_config(file_path_config)

    # Init values

    # Init flags

    flag_force_save = args.force_save

    # Folder setup

    folder_guard(FOLDER_DATA)

    if model_config is not None:
        print(INFO_PREFIX + 'Using config from: ' + file_path_config)








