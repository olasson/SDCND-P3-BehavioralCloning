import numpy as np

from code.io import load_sim_log

N_CAMS = 3

# Prepared image size
N_ROWS_PREPARED = 64 
N_COLS_PREPARED = 128 
N_CHANNELS_PREPARED = 3


def _find_indices_to_delete(file_names, alpha):
    """
    Find a set of indices in a file_namesset to remove in order to make it more uniform 
    
    Inputs
    ----------
    file_names: numpy.ndarray
        Numpy array containing scalar values.
    alpha: float
        Scalar value determining how many indicies should be dropped. 
       
    Outputs
    -------
    indices: numpy.ndarray
        Numpy array containing a set of indicies that should be removed from 'file_names' to make it more uniform.
        
    """

    values, bins = np.histogram(file_names, bins = 'auto')

    n_max_samples_in_bin = (np.mean(values) * (1.0 / alpha)).astype('uint32')

    n_bins = len(bins)

    indices = []

    # bins = [bin_left1, bin_right1, bin_left2, bin_right2...]
    for i in range(n_bins - 1):

        bin_left = bins[i]
        bin_right = bins[i + 1]

        if i == (n_bins - 2):
            # Last bin is closed
            file_names_in_bin = np.where((file_names >= bin_left) & (file_names <= bin_right))[0]
        else:
            # All bins except the last is half open (ref numpy docs)
            file_names_in_bin = np.where((file_names >= bin_left) & (file_names < bin_right))[0]
        
        n_samples_in_bin = len(file_names_in_bin)

        if n_samples_in_bin > n_max_samples_in_bin:
            n_indices_to_remove = n_samples_in_bin - n_max_samples_in_bin

            remove = np.random.choice(file_names_in_bin, size = n_indices_to_remove, replace = False)

            indices.extend(remove)

    return indices


def prepare_sim_log(angles, file_names, angle_correction, angle_flatten):
    """
    Prepare the sim log data for further use
    
    Inputs
    ----------
    angles: numpy.ndarray
        Numpy array containing a set of angles
    file_names: numpy.ndarray
        Numpy array containing a set of file names
    angle_correction: float
        Angle correction which will be applied to left and right angles
    angle_flatten: int
        Scalar determining how many samples from 'angles' and 'file_names' to be removed
       
    Outputs
    -------
    angles: numpy.ndarray
        Numpy array containing a set of angles 
    file_names: numpy.ndarray
        Numpy array containing a set of file names. len(angles) == len(file_names)
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

        file_names = np.delete(file_names, indices, axis = 0)


    return angles, file_names

def prepare_data(file_path, angle_correction, angle_flatten, augment = True, preview = False):

    angles, file_names = load_sim_log(file_path)

    angles, file_names = prepare_sim_log(angles, file_names, angle_correction, angle_flatten)

    if augment:
        n_samples = 3 * len(angles)
    else:
        n_samples = len(angles)

    if preview:
        images_out = None
    else:
        images_out = np.zeros((n_samples, N_ROWS_PREPARED, N_COLS_PREPARED, N_CHANNELS_PREPARED), dtype = 'uint8')

    return angles, images_out


