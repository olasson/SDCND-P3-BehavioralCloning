# **Behavioral Cloning** 

*by olasson*

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

*This is a revised version of my Behavioral cloning project.*

## Project overview

The majority of the project code is located in the folder `code`:

* [`augment.py`](https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/code/augment.py)
* [`constants.py`](https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/code/constants.py)
* [`misc.py`](https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/code/misc.py)
* [`io.py`](https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/code/io.py)
* [`model.py`](https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/code/model.py)
* [`prepare.py`](https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/code/prepare.py)
* [`plots.py`](https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/code/plots.py)
* [`drive.py`](https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/code/drive.py)
* [`video.py`](https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/code/video.py)

The main project script is called [`behavioral_cloning.py`](https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/behavioral_cloning.py). It contains the implementation of a very simple command line tool.

The images shown in this readme are found in 

* `images/`

The trained model used in this project readme are found in

* `models/`

The videos of the model driving are found in

* `videos/`

## Command line arguments

The following command line arguments are defined:

#### Show

* `--show_images:` File path to a simulator driving log (.csv) or a pickled file (.p) file containing a set of prepared images.
* `--n_triplets:` The number of triplets (one image from left cam, center cam and right cam each) in the image plot.

#### Config

* `--model_config:` Path to a .json file containing model config.

#### Data

* `--preview:` If enabled, the user can preview the steering angle distribution without processing the images.

#### Simulator

* `--drive:` If enabled, the program will attempt to drive the simulator car with the model provided by --model_config.
* `--record:` If enabled, the simulator run will be recorded.
* `--speed:` The speed of the car in the simulator.
* `--track_name:` Name of the simulator track used in the recording, which in turn is used in the video output name.


## Config system

The model is kept the same for all experiments, but in order to simplify the process of experimenting with different parameters, I have implemented a very simple config system. To create new datasets and train a model, the user only needs to define a `.json` file in the folder `config/` on the form (example):

    {
        "model_path": "./models/nVidia_02.h5",
        "driving_log": "./data/driving_log.csv",
        "data_prepared": "./data/data_prepared_02.p",
        "augment": 1,
        "angle_correction": 0.2,
        "angle_flatten": 0.6,
        "lrn_rate": 0.001,
        "n_max_epochs": 50,
        "batch_size": 64
    }

After the config file is created run the command

`python behavioral_cloning.py --model_config './config/<config_file_name>.json'`

The [`behavioral_cloning.py`](https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/behavioral_cloning.py) script will automatically detect if the specified datasets and model exists or not, and create them if needed.

## Data Gathering and exploration

I collected data for this project by driving each track 1 time clockwise and 1 time counter-clockwise, resulting in 2 full laps for each trach. In addition, I repeated certain problem areas (tight turns, different groun textures etc.) to hopefully improve model robustness.

The simulator collects the following information for each time it takes a sample:

* `Center Image, Left Image, Right Image, Angle, Throttle, Break, Speed`

For the project, the following was used: `Center Image, Left Image, Right Image, Angle`. Throttle, Break and Speed was discarded, mainly due to the fact that driving properly using the mouse and keyboard was somewhat difficult, especially on the jungle track, and the quality of those datapoints might not be top notch. 

First, lets take a look at images from each camera.  

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/images/driving_log_raw.png">
</p>

*Observation 1:* I can use all three images by associating each image with the angle recorded. I can then apply and angle correction to the left and right image to further enrich the data. The data from the sim log is loaded like this by the function `load_sim_log()` found in `io.py`.

    ...
    sim_log = read_csv(path, names = ['center', 'left', 'right', 'angle', 'throttle', 'break', 'speed'])

    file_names = sim_log[['center', 'left', 'right']].to_numpy().flatten()

    angles = sim_log[['angle', 'angle', 'angle']].to_numpy().flatten()
    ....
    
Note that the angle values is inserted once for each image from the camera. The array `file_names` contains the absolute path to every image saved by the simulator. 

Below is the raw angle distribution, completely unprocessed from the `driving_log.csv`. 

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/images/steering_angle_dist_raw.png">
</p>

*Observation 2:* As expected, there is a majority of angles centered around `0.0` This will need to be handled in the preparation steps. If not, the model will be overfit towards driving straight ahead. 

*Observation 3:* There are roughly 28,000 samples, but I don't want that many images with the angle `0.0`. Therefore, it is unnecessary to actually load each and every image, instead I should use distribution plot to determine which images to load by flattening the distribution. 

## Data Preparation

The main function for data preparation is called `prepare_data()` and is found in `misc.py`. It accepts a boolean value called `preview`. If `preview = True`, the function will only return the angle distribution and skip all image processing. This is useful for experimenting with different values, as it is incredibly quick compared to also preparing all images every time a value changes. In addition, it also accepts a boolean value called `augment`. If `preview = True` and `augment = True`, the user can see how the augmented angle distribution will look like.

### Angle Preparation

Before thinking about the details of how to prepare the images, I wanted to make the distribution look more reasonable. 

The function `prepare_sim_log()` found in `prepare.py` is called by `prepare_data()` and does this in a two step process:
1. Apply a non-zero angle correction
2. Flatten the distribution. 

The second step is accomplished by calling `find_indices_to_delete()` found in `misc.py`. It uses a scalar `angle_flatten` to compute a numpy array of indicies which are then deleted from both `angles` and `file_names`. To illustrate the effects of flattening the distribution, lets take a look at an example:


<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/images/steering_angle_dist_nVidia_02.png">
</p>

As we can see the distribution looks more reasonable now, but we have effectively thrown away roughly 12,000 samples, compared to the raw distribution. We can use data augmentation to artificially create more samples. The function `prepare_data()` does this as follows: 
         
        ...
        # Original
        angles_out[i] = angles[i]

        # Augmented
        if augment:

            # Flipped
            angles_out[i + n_samples] = -angles_out[i]

            # Translated
            T_x = int(np.random.randint(-CROP_COLS1, CROP_COLS2))
            angles_out[i + (2 * n_samples)] = translate_angle(angles_out[i], T_x)
        ...

If `augment = True`, `prepare_data()` will create `2*len('angles output from prepare_sim_log')`'s worth of new samples, or in this example case rougly `2*16000 = 32000` new samples "for free". After some experimenting with different values, my final distribution looked like this:

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/images/steering_angle_dist_nVidia_02.png">
</p>

Which seems to be a decent spread of values. 

### Image Preparation

The "minimum" image preparation for each image is handled by `prepare_image()` found in `prepare.py`:
1. Convert to YUV color space
2. Crop away strips from the top, bottom and sides of the image.
3. Resize the image
4. Blur the image

After this process is applied the image dimensions goes from `(160, 320, 3) -> (64, 128, 3)`, in order lighten computational load on the model. These values are configured in the file `config.py`. 

In addition, the images are augmented in two separate "passes" in `prepare_data()`: 

    ...

    # Augmented
    if augment:
        # Flipped
        images_out[i + n_samples] = cv2.flip(images_out[i], 1)

        # Translated and adjusted brightness
        alpha = np.random.uniform(ALPHA_MIN, ALPHA_MAX)
        images_out[i + (2 * n_samples)] = prepare_image(image, T_x, alpha)
    ...

The first pass simply flips the images about the vertical axis. The second pass translates and adjusts the brightness of the images. The second pass is handled entirely as optional in the function `prepare_image()` found in `prepare.py`

The dataset was first prepared and saved as a pickled file for easy re-use:

Command: ` python behavioral_cloning.py --data_save 'general_aug.p' --data_prepare --data_augment --angle_correction 0.2 --angle_flatten 0.6`

The above command creates an augmented dataset based on `driving_log.csv` and stores it at `./data/general_aug.p`.

Here is a random subset of images:

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/images/data_prepared_02.png">
</p>

The above images is what will be fed to the model. 

## Model Architecture

The model is based on the following [`paper`](https://arxiv.org/pdf/1604.07316v1.pdf). An overview provided by `model.summary()` is provided below:

      _________________________________________________________________
      Layer (type)                 Output Shape              Param #   
      =================================================================
      lambda (Lambda)              (None, 64, 128, 3)        0         
      _________________________________________________________________
      conv2d (Conv2D)              (None, 30, 62, 24)        1824      
      _________________________________________________________________
      conv2d_1 (Conv2D)            (None, 13, 29, 36)        21636     
      _________________________________________________________________
      conv2d_2 (Conv2D)            (None, 5, 13, 48)         43248     
      _________________________________________________________________
      conv2d_3 (Conv2D)            (None, 3, 11, 64)         27712     
      _________________________________________________________________
      conv2d_4 (Conv2D)            (None, 1, 9, 64)          36928     
      _________________________________________________________________
      dropout (Dropout)            (None, 1, 9, 64)          0         
      _________________________________________________________________
      flatten (Flatten)            (None, 576)               0         
      _________________________________________________________________
      dense (Dense)                (None, 100)               57700     
      _________________________________________________________________
      dense_1 (Dense)              (None, 50)                5050      
      _________________________________________________________________
      dense_2 (Dense)              (None, 10)                510       
      _________________________________________________________________
      dense_3 (Dense)              (None, 1)                 11        
      =================================================================
      Total params: 194,619
      Trainable params: 194,619
      Non-trainable params: 0


The activation function used for this project is `ELU`, found mostly by trial and error. I also added one dropout layer with 0.3 probability to combat overfitting.

## Training

The function `train_and_save_model()` found in `model.py` sets up and executes the training, like so:
    
    ...
    model.compile(optimizer = Adam(lr = lrn_rate), loss = 'mse', metrics=['mean_squared_error'])

    # The callback will stop training if the model does not see a certain improvement in 'val_loss' over 3 epochs. 
    early_stopping = EarlyStopping(monitor = 'val_loss', 
                                   patience = 3,
                                   verbose = 0, 
                                   mode = 'min')

    history = model.fit(images, angles, batch_size = batch_size, epochs = n_max_epochs, 
                        validation_split = 0.2, callbacks = [early_stopping])

    model.save(path_model_save)
    ...

Instead of trying to "guess" at the apropriate number of epochs, I instead rely on the `early_stopping` callback to decide when to stop training. The hyperparameter `n_max_epochs` defaults to 50, and is intended more as an upper limit rather than an exact number of epochs to train for. The parameter  `lrn_rate` defaults to 0.0001  and `batch_size` defaults to 64. 

Below is the plot for the training history.

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/images/nVidia_02.png">
</p>

It trained for 42/50 epochs, and seemes to steadily converge.


## Results and Discussion

Below are a couple of snippets from the model driving the car. The full videos are found in the `videos` folder. The best way to validate the model is to downlad it and test it against the Udacity simulatr. 

<p align="center">
  <img width="50%" height="50%" src="https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/videos/lake_nVidia_02.gif">
  <img width="50%" height="50%" src="https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/videos/jungle_nVidia_02.gif">
</p>

The model works reasonably well, able to drive a full lap on both tracks. However, there are a couple of issues I noted:
1. The model swerving at certain parts when driving on the lake track. I am unsure what causes this behavior, but I suspect this is related to the `angle_correction` value. If set too high (0.5 for example), the model would "aggressively" try to correct its trajectory when driving too close to the edge. 
2. I was unable to drive the jungle track at full speed. Instead, the speed for this track was set to 20. Anything higher, and the car would eventually drive off the track. 

