import numpy as np

from code.io import load_sim_log

N_CAMS = 3

# Prepared image size
N_ROWS_PREPARED = 64 
N_COLS_PREPARED = 128 
N_CHANNELS_PREPARED = 3


def _find_indices_to_delete(file_paths, flatten_factor):
    """
    Find a set of indices from 'file_paths' to remove in order to only load images of interest.
    
    Inputs
    ----------
    file_paths: numpy.ndarray
        Numpy array containing a set of file_paths.
    flatten_factor: float
        Scalar value determining how many indicies should be dropped. 
       
    Outputs
    -------
    indices: numpy.ndarray
        Numpy array containing a set of indicies that should be removed from 'file_paths'.   
    """

    values, bins = np.histogram(file_paths, bins = 'auto')

    n_max_samples_in_bin = (np.mean(values) * (1.0 / flatten_factor)).astype('uint32')

    n_bins = len(bins)

    indices = []

    # bins = [bin_left1, bin_right1, bin_left2, bin_right2...]
    for i in range(n_bins - 1):

        bin_left = bins[i]
        bin_right = bins[i + 1]

        if i == (n_bins - 2):
            # Last bin is closed
            file_paths_in_bin = np.where((file_paths >= bin_left) & (file_paths <= bin_right))[0]
        else:
            # All bins except the last is half open (ref numpy docs)
            file_paths_in_bin = np.where((file_paths >= bin_left) & (file_paths < bin_right))[0]
        
        n_samples_in_bin = len(file_paths_in_bin)

        if n_samples_in_bin > n_max_samples_in_bin:
            n_indices_to_remove = n_samples_in_bin - n_max_samples_in_bin

            remove = np.random.choice(file_paths_in_bin, size = n_indices_to_remove, replace = False)

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

def prepare_data(file_path, angle_correction, angle_flatten, augment = False, preview = False):

    angles, file_paths = load_sim_log(file_path)

    angles, file_paths = prepare_sim_log(angles, file_paths, angle_correction, angle_flatten)

    n_samples = len(angles)

    if augment:
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

    return angles_out, images_out


