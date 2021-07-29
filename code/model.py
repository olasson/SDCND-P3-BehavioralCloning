"""
This file contains the model implementation used in the project.
"""


# Suppress some of the "standard" tensorflow output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dropout, Lambda, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Prevent tensorflow from using too much GPU memory
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)
session.close()

from code.constants import N_ROWS_PREPARED, N_COLS_PREPARED, N_CHANNELS_PREPARED

ACTIVATION_FUNCTION = 'elu'

# Based on https://arxiv.org/pdf/1604.07316v1.pdf
def nVidia_model():
    
    model = Sequential()

    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (N_ROWS_PREPARED, N_COLS_PREPARED, N_CHANNELS_PREPARED)))

    model.add(Conv2D(24, (5, 5), activation = ACTIVATION_FUNCTION, strides = (2, 2)))

    model.add(Conv2D(36, (5, 5), activation = ACTIVATION_FUNCTION, strides = (2, 2)))

    model.add(Conv2D(48, (5, 5), activation = ACTIVATION_FUNCTION, strides = (2, 2)))

    model.add(Conv2D(64, (3, 3), activation = ACTIVATION_FUNCTION))

    model.add(Conv2D(64, (3, 3), activation = ACTIVATION_FUNCTION))

    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(100, activation = ACTIVATION_FUNCTION))
    model.add(Dense(50, activation = ACTIVATION_FUNCTION))
    model.add(Dense(10, activation = ACTIVATION_FUNCTION))
    model.add(Dense(1))

    return model


def save_model(file_path, model):

    # Wrapper

    model.save(file_path)

def train_model(images, angles, lrn_rate, batch_size, n_max_epochs):
    """
    Train a model with callback and save it. 
    
    Inputs
    ----------
    images : numpy.ndarray
        Numpy array containing a set of images.
    angles : numpy.ndarray
        Numpy array containing a set of angles.
    lrn_rate : float
        Model learning rate.
    batch_size : int
        Model batch_size.
    n_max_epochs : int
        The maximum number of epochs the model can train for.
       
    Outputs
    -------
    model: Keras model object.
        Keras sequential model.
    history: numpy.ndarray
        Keras model history object.
        
    """

    model = nVidia_model()

    model.compile(optimizer = Adam(lr = lrn_rate), loss = 'mse', metrics=['mean_squared_error'])

    # The callback will stop training if the model does not see a certain improvement in 'val_loss' over 3 epochs. 
    early_stopping = EarlyStopping(monitor = 'val_loss', 
                                   patience = 3,
                                   verbose = 0, 
                                   mode = 'min')

    history = model.fit(images, angles, batch_size = batch_size, epochs = n_max_epochs, 
                        validation_split = 0.2, callbacks = [early_stopping])

    return model, history


