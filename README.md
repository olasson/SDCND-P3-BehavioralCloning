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

#### Data

* `--data_meta:` File path to a .csv data file containing sign meta data.

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

The [`behavioral_cloning.py`](https://github.com/olasson/SDCND-P3-BehavioralCloning/blob/master/behavioral_cloning.py.py) script will automatically detect if the specified datasets and model exists or not, and create them if needed.


