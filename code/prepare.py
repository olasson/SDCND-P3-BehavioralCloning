"""
This file contains functions for preparing the simulator data for use by a model.
"""

import cv2
import numpy as np

from code.io import load_sim_log, load_image
from code.augment import translate_image, adjust_brightness

from code.constants import N_CAMS, N_ROWS_PREPARED, N_COLS_PREPARED, N_CHANNELS_PREPARED

# Original image size
N_ROWS = 160 
N_COLS = 320 
N_CHANNELS = 3

CROP_ROWS_TOP = 60 
CROP_ROWS_BOTTOM = 25

CROP_COLS_LEFT = 15 
CROP_COLS_RIGHT = 15

# Conversion factor for translating angles
RAD_PER_PIXEL = 0.01


def _find_indices_to_delete(data_1D, flatten_factor):
    """
    Flatted a data distribution by findining indicies to delete.
    
    Inputs
    ----------
    data_1D: numpy.ndarray
        Numpy array containing a 1D data set.
    flatten_factor: float
        Scalar value determining how many indicies should be dropped. 
       
    Outputs
    -------
    indices: numpy.ndarray
        Numpy array containing a set of indicies that should be removed from 'data_1D' to make it more uniform.   
    """

    values, bins = np.histogram(data_1D, bins = 'auto')

    n_max_samples_in_bin = (np.mean(values) * (1.0 / flatten_factor)).astype('uint32')

    n_bins = len(bins)

    indices = []

    # bins = [bin_left1, bin_right1, bin_left2, bin_right2...]
    for i in range(n_bins - 1):

        bin_left = bins[i]
        bin_right = bins[i + 1]

        if i == (n_bins - 2):
            # Last bin is closed
            samples_in_bin = np.where((data_1D >= bin_left) & (data_1D <= bin_right))[0]
        else:
            # All bins except the last is half open (ref numpy docs)
            samples_in_bin = np.where((data_1D >= bin_left) & (data_1D < bin_right))[0]
        
        n_samples_in_bin = len(samples_in_bin)

        if n_samples_in_bin > n_max_samples_in_bin:
            n_indices_to_remove = n_samples_in_bin - n_max_samples_in_bin

            remove = np.random.choice(samples_in_bin, size = n_indices_to_remove, replace = False)

            indices.extend(remove)

    return indices


def prepare_sim_log(angles, file_paths, angle_correction, angle_flatten):
    """
    Prepare the sim log data for further use
    
    Inputs
    ----------
    angles: numpy.ndarray
        Numpy array containing a set of angles
    file_paths: numpy.ndarray
        Numpy array containing a set of file paths
    angle_correction: float
        Angle correction which will be applied to left and right angles
    angle_flatten: int
        Scalar determining how many samples from 'angles' and 'file_paths' to be removed
       
    Outputs
    -------
    angles: numpy.ndarray
        Numpy array containing a set of angles 
    file_paths: numpy.ndarray
        Numpy array containing a set of file paths. len(angles) == len(file_paths)
    """

    if angle_correction != 0.0:

        n_iterations = len(angles) // N_CAMS

        for i in range(n_iterations):

            # Left cam
            angles[1 + (N_CAMS * i)] = angles[1 + (N_CAMS * i)] + angle_correction

            # Right cam
            angles[2 + (N_CAMS * i)] = angles[2 + (N_CAMS * i)] - angle_correction

    if angle_flatten > 0.0:

        indices = _find_indices_to_delete(angles, angle_flatten)

        angles = np.delete(angles, indices, axis = 0)

        file_paths = np.delete(file_paths, indices, axis = 0)


    return angles, file_paths


def prepare_image(image, T_x = None, brightness_factor = None):
    """
    Prepare a single image for use by a model
    
    Inputs
    ----------
    image: numpy.ndarray
        Numpy array containing a single RGB image
    T_x: (None | int)
        Number of pixels the image will be translated in the x-dir (along a row)
    alpha: (None | float)
        Scalar value to darken (lower values) or brighten (higher values)
       
    Outputs
    -------
    image: numpy.ndarray
        Numpy array containing a single prepared YUV image.
    """

    if T_x is not None:
        image = translate_image(image, T_x)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    image = image[CROP_ROWS_TOP:N_ROWS - CROP_ROWS_BOTTOM, CROP_COLS_LEFT:N_COLS - CROP_COLS_RIGHT, :]

    if brightness_factor is not None:
        image = adjust_brightness(image, brightness_factor)

    image = cv2.resize(image, (N_COLS_PREPARED, N_ROWS_PREPARED), interpolation = cv2.INTER_AREA)

    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image

def prepare_data(file_path, angle_correction, angle_flatten, augment = False, preview = False):
    """
    Prepare a dataset (images, angles) for use by a model.
    
    Inputs
    ----------
    path: str
        Path to the driving_log.csv from the simulator.
    angle_correction: float
        Angle correction which will be applied to left and right angles
    angle_flatten: float
        Scalar determining how many samples from 'angles' and 'file_names' to be removed
    preview: bool
        If True,  'prepare_data' will return angles, and images_out = None. Useful for previewing how an angle distribution will look.
    augment: bool
        If True, augmentation (flip, translate and adjust brightness) will be applied to the dataset. 
       
    Outputs
    -------
    angles_out: numpy.ndarray
        Numpy array containing a set of prepared angles.
    images_out: (None | numpy.ndarray)
        Numpy array containing a set of prepared images.
    """

    angles, file_paths = load_sim_log(file_path)

    angles, file_paths = prepare_sim_log(angles, file_paths, angle_correction, angle_flatten)

    n_samples = len(angles)

    if augment:
        # The multiplier allocates space for (Original + Flipped + Translated) 
        n_samples_total = 3 * n_samples
    else:
        n_samples_total = n_samples

    if preview:
        images_out = None
    else:
        images_out = np.zeros((n_samples_total, N_ROWS_PREPARED, N_COLS_PREPARED, N_CHANNELS_PREPARED), dtype = 'uint8')

    angles_out = np.zeros((n_samples_total), dtype = np.float32)

    for i in range(n_samples):

        # Original 
        angles_out[i] = angles[i]

        if augment:

            # Flipped
            angles_out[i + n_samples] = -angles_out[i]

            # Translated
            T_x = int(np.random.randint(-CROP_COLS_LEFT, CROP_COLS_RIGHT))
            angles_out[i + (2 * n_samples)] = angles_out[i] + (RAD_PER_PIXEL * T_x)

        if not preview:

            image = load_image(file_paths[i])

            # Original
            images_out[i] = prepare_image(image)

            if augment:

                # Flipped
                images_out[i + n_samples] = cv2.flip(images_out[i], 1)

                brightness_factor = np.random.uniform(0.8, 1.5)
                images_out[i + (2 * n_samples)] = prepare_image(image, T_x, brightness_factor)

    angles_out = np.array(np.clip(angles_out, a_min = -1.0, a_max = 1.0))

    return angles_out, images_out


