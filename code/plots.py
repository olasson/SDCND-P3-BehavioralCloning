"""
This file contains function(s) for visualizing data.
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

N_IMAGES_MAX = 50

def plot_images(images, 
                titles_top = None, 
                titles_bottom = None, 
                title_fig_window = None, 
                fig_size = (15, 15), 
                font_size = 10, 
                cmap = None, 
                n_max_cols = 3, 
                titles_bottom_h_align = 'center', 
                titles_bottom_v_align = 'top', 
                titles_bottom_pos = (16, 32)):
    """
    Show a set of images
    
    Inputs
    ----------
    images : numpy.ndarray
        A set of images, RGB or grayscale
    titles_top: (None | list)
        A set of image titles to be displayed on top of an image
    titles_bottom: (None | list)
        A set of image titles to be displayed at the bottom of an image
    title_fig_window: (None | string)
        Title for the figure window
    figsize: (int, int)
        Tuple specifying figure width and height in inches
    fontsize: int
        Fontsize of 'titles_top' and 'titles_bottom'
    cmap: (None | string)
        RGB or grayscale
    n_max_cols: int
        Maximum number of columns allowed in figure
    titles_bottom_h_align: string
        Horizontal alignment of 'titles_bottom'
    titles_bottom_v_align: string
        Vertical alignment of 'titles_bottom'
    titles_bottom_pos: (int, int)
        Tuple containing the position of 'titles_bottom'
    titles_bottom_transform: string
        Coordinate system used by matplotlib for 'titles_bottom'
    Outputs
    -------
    plt.figure
        Figure showing 'images' in an (n_rows x n_cols) layout
    
    """

    n_images = len(images)

    # Stop myself from accidentally trying to show 30,000 images
    if n_images > N_IMAGES_MAX:
        print("ERROR: code.plot.plot_images(): You're trying to show", n_images, "images. Max number of allowed images:", N_IMAGES_MAX)
        return

    n_cols = int(min(n_images, n_max_cols))
    n_rows = int(np.ceil(n_images / n_cols))

    fig = plt.figure(title_fig_window, figsize = fig_size)

    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i + 1)

        # Expect that images are loaded with openCV - BGR representation
        #image = cv2.cvtColor(images[i].astype('uint8'), cv2.COLOR_BGR2RGB)

        image = images[i].astype('uint8')

        plt.imshow(image, cmap = cmap)

        plt.xticks([])
        plt.yticks([])

        if titles_top is not None:
            plt.title(titles_top[i], fontsize = font_size)

        if titles_bottom is not None:
            plt.text(titles_bottom_pos[0], titles_bottom_pos[1], 
                     titles_bottom[i],
                     verticalalignment = titles_bottom_v_align, 
                     horizontalalignment = titles_bottom_h_align,
                     fontsize = font_size - 3)

    #plt.tight_layout()
    plt.show()


def plot_distribution(angles, title = None, title_fig_window = None, angle_correction = 0.0, angle_flatten = 0.0,
                      bins = 'auto', fig_size = (15, 6), font_size = 12):

    """
    Plot a distribution of angles.
    
    Inputs
    ----------
    angles : numpy.ndarray
        Numpy array containing a set of angles.
    title: (None | str)
        A title for the plot.
    title_fig_window: (None | string)
        Title for the figure window
    angle_correction: float
        Angle correction used on 'angles'
    angle_flatten: float
        Flattening used on 'angles'
    bins: (str | int)
        Number of bins to use.
    fig_size: (int, int)
        Tuple specifying figure width and height in inches
    font_size: int
        Fontsize of 'titles'
    Outputs
    -------
    plt.figure
        Figure angle distribution
    
    """

    fig = plt.figure(title_fig_window, figsize = fig_size)

    plt.hist(angles, bins = bins)

    if title is not None:
        plt.title(title)

    mean_value = np.mean(angles, axis = 0)
    plt.axvline(mean_value, color = 'k', linestyle = 'dashed', linewidth = 1)

    info_str = 'Number of samples: ' + str(len(angles)) 
    info_str += '\n Angle correction: ' + str(angle_correction)
    info_str += '\n Flatten factor: ' + str(angle_flatten)
    info_str += '\n Mean value: ' + str(round(mean_value, 2))

    fig.text(0.9, 0.9, info_str,
            verticalalignment = 'top', 
            horizontalalignment = 'center',
            color = 'black', fontsize = font_size)

    plt.tight_layout()

    plt.show()

def plot_model_history(history, model_name = None, lrn_rate = None, batch_size = None, n_max_epochs = None, file_path_save = None):
    """
    Plot model history and metadata
    
    Inputs
    ----------
    model_name: string
        Name of the model
    history: Keras History Object
        Model history (output from .fit)
    path_save: (None | string)
        Path to where the plot will be saved. 
    lrn_rate: (None | float)
        Model learning rate
    batch_size: (None | int)
        Model batch size
    n_max_epochs: (None | int)
        Model max epochs 
        
    Outputs
    -------
    plt.figure
        Figure showing model history and metadata, either shown directly or saved in location 'path_save'
    
    """

    if model_name is None:
        model_name = 'model'

    train_log = history.history['loss']
    valid_log = history.history['val_loss']
    
    train_loss = train_log[-1]
    valid_loss = valid_log[-1]
    
    text = "Training/Validation Loss: " + str(round(train_loss, 3)) + '/' + str(round(valid_loss, 3))   
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    c1 = colors[0]
    c2 = colors[1]
    
    fig, ax1 = plt.subplots(figsize = (9, 6))
    
    ax1.set_xlabel('Epochs')    
    ax1.set_ylabel('Loss')

    x = np.arange(1, len(train_log) + 1)
    
    ax1.plot(x, train_log, label = 'Train Loss', color = c1)
    ax1.plot(x, valid_log, label = 'Validation Loss', color = c2)


    stopping_epoch = len(history.history['loss'])

    # ---------- Construct a title for the plot ---------- # 

    model_name_title = 'Model Name: '+ model_name + ' | '

    if lrn_rate is not None:
        lrn_rate_title = 'Lrn rate: ' + str(lrn_rate) + ' | '
    else:
        lrn_rate_title = ''

    if batch_size is not None:
        batch_size_title = 'Batch size: ' + str(batch_size) + ' | '
    else:
        batch_size_title = ''

    if n_max_epochs is not None:
        epochs_title = 'Stopp/Max (Epoch): ' + str(stopping_epoch) + '/' + str(n_max_epochs)
    else:
        epochs_title = 'Stopp Epoch: ' + str(stopping_epoch)

    plt.title(model_name_title + lrn_rate_title + batch_size_title + epochs_title)

    # ---------- Misc ---------- #
    
    fig.text(0.5, 0, text,
                verticalalignment = 'top', 
                horizontalalignment = 'center',
                color = 'black', fontsize = 10)
    
    handles, labels = ax1.get_legend_handles_labels()
    
    fig.legend(handles, labels, loc = (0.7, 0.5))
    fig.tight_layout()

    # ---------- Show or save ---------- #
    
    # If the user has opted to save the model history, don't show the plot directly
    if file_path_save is not None:
        fig.savefig(file_path_save, bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()