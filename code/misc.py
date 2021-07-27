"""
This file contains some miscellaneous helper functions and os wrappers.
"""

import os
import numpy as np

from code.constants import N_CAMS

def file_exists(file_path):
    """
    Check if a file exists.
    
    Inputs
    ----------
    path: str
        Path to file.
       
    Outputs
    -------
    bool
        True if file exists, false otherwise.
        
    """

    if file_path is None:
        return False

    if not os.path.isfile(file_path):
        return False

    return True

def folder_guard(folder_path):
    """
    Checks if a folder exists and creates it if it does not.
    
    Inputs
    ----------
    folder_path: str
        Path to folder.
       
    Outputs
    -------
        N/A
        
    """
    if not os.path.isdir(folder_path):
        print('INFO:folder_guard(): Creating folder: ' + folder_path + '...')
        os.mkdir(folder_path)

def folder_is_empty(folder_path):
    """
    Check if a folder is empty. If the folder does not exist, it counts as being empty. 
    
    Inputs
    ----------
    folder_path: str
        Path to folder.
       
    Outputs
    -------
    bool
        True if folder exists and contains elements, false otherwise.
        
    """

    if os.path.isdir(folder_path):
        return (len(os.listdir(folder_path)) == 0)
    
    return True

def parse_file_path(file_path):

    """
    Parse out the folder path and file path from a full path.
    
    Inputs
    ----------
    file_path: string
        Path to a file - './path/to/myfile.jpg'
        
    Outputs
    -------
    folder_path: string
        The folder path contained in 'file_path' - './path/to/'
    file_name: string
        The file_name contained in 'file_path' - 'myfile.jpg'
    """

    file_name = os.path.basename(file_path)

    cutoff = len(file_path) - len(file_name)

    folder_path = file_path[:cutoff]

    return folder_path, file_name


def pick_triplets(data, n_triplets, dtype = np.float32):
    """
    Pick out sets of 'cam triplets'  (left, center, right). 
    
    Inputs
    ----------
    data: numpy.ndarray
        Numpy array containing scalar values.
    n_triplets: int
        The number of triplets. 
       
    Outputs
    -------
    data_out: numpy.ndarray
        Numpy array containing a set of 'n_triplets'.
        
    """

    n_samples = n_triplets * N_CAMS

    data_out = np.zeros((n_samples), dtype = dtype)

    indices = range(0, len(data), N_CAMS)

    for i in range(0, n_samples, N_CAMS):
        k = np.random.choice(indices)
        data_out[i] = data[k + 1] # Left
        data_out[i + 1] = data[k] # Center
        data_out[i + 2] = data[k + 2] # Right

    return data_out


def is_file_type(file_path, file_type):
    """
    Check if a file has the expected file type or not.
    
    Inputs
    ----------
    file_path : str
        Path to a file.
    file_type : str
        File type extension.
       
    Outputs
    -------
    bool
        True if file exists and is of type 'file_type', false otherwise.
        
    """

    if not file_exists(file_path):
        return False

    if file_path.endswith(file_type):
        return True

    return False


