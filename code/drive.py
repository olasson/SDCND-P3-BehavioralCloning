"""
This file contains a modified version of the drive.py script provided by Udacity. It is no longer a standalone script, but
integrated to work with the main script behavioral_cloning.py
"""

import base64
from datetime import datetime
import os

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from os.path import join as path_join

# Custom imports
from code.prepare import prepare_image


def drive_sim_car(model, folder_path_sim_record, set_speed):
    """
    Function wrapper to make script callable.
    
    Inputs
    ----------
    model: Keras sequential model
        Keras model object
    folder_path_sim_record: str
        Path specifying where recorded images should be saved.
    set_speed: int
        Initial speed value for the simulator.
       
    Outputs
    -------
        N/A
        
    """

    sio = socketio.Server()


    class SimplePIController:
        def __init__(self, Kp, Ki):
            self.Kp = Kp
            self.Ki = Ki
            self.set_point = 0.
            self.error = 0.
            self.integral = 0.

        def set_desired(self, desired):
            self.set_point = desired

        def update(self, measurement):
            # proportional error
            self.error = self.set_point - measurement

            # integral error
            self.integral += self.error

            return self.Kp * self.error + self.Ki * self.integral


    @sio.on('telemetry')
    def telemetry(sid, data):

        if data:
            # The current steering angle of the car
            steering_angle = data["steering_angle"]
            # The current throttle of the car
            throttle = data["throttle"]
            # The current speed of the car
            speed = data["speed"]
            # The current image from the center camera of the car
            imgString = data["image"]
            image = Image.open(BytesIO(base64.b64decode(imgString)))
            image_array = np.asarray(image)

            # Prepare image for use by a model
            image_array = prepare_image(image_array)

            steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

            throttle = controller.update(float(speed))

            # Slow down on hard turns.
            if abs(steering_angle) >= 0.55:
                throttle = 0.1

            

            print(steering_angle, throttle)
            send_control(steering_angle, throttle)

            # save frame
            if folder_path_sim_record != '':
                timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
                image_filename = path_join(folder_path_sim_record, timestamp)
                image.save('{}.jpg'.format(image_filename))
        else:
            # NOTE: DON'T EDIT THIS.
            sio.emit('manual', data={}, skip_sid=True)


    @sio.on('connect')
    def connect(sid, environ):
        print("connect ", sid)
        send_control(0, 0)


    def send_control(steering_angle, throttle):
        sio.emit(
            "steer",
            data={
                'steering_angle': steering_angle.__str__(),
                'throttle': throttle.__str__()
            },
            skip_sid=True)

    app = Flask(__name__)
    
    controller = SimplePIController(0.1, 0.002)
    controller.set_desired(set_speed)

    # Wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # Deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)