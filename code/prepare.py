import numpy as np

from code.io import load_sim_log

def prepare_data(file_path, angle_correction, angle_flatten):

    angles, file_names = load_sim_log(file_path)
