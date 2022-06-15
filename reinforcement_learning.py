#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Keyboard controlling for CARLA. Please refer to client_example.py for a simpler
# and more documented example.

"""
Welcome to CARLA manual control.
Use ARROWS or WASD keys for control.
    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    R            : restart level
STARTING in a moment...
"""

from __future__ import print_function

import argparse
import logging
import random
import time

import os
import subprocess
import numpy as np
from datetime import datetime


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import os
from collections import deque

## Keras
import keras
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D,Lambda, Conv2D, Dropout, Flatten, Dense,GlobalAveragePooling2D,Activation
from keras.models import model_from_json
from keras.models import load_model
from keras.applications.xception import Xception
from keras.models import Model


import cv2
from threading import Thread
import pandas as pd
import random
import ntpath
import pickle
## Sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import ResNet50

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

import carla
from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.driving_benchmark.metrics import Metrics
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from PIL import Image
from pexpect import popen_spawn
import io
import pickle
import time




WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MINI_WINDOW_WIDTH = 36#320
MINI_WINDOW_HEIGHT = 36#160
detected_colour = ''
frame = 0
penalty=0
its_red=False
num=1
brk=0
spd=0
back=False
steers=[]
STEER=0
model=None
target_model=None
center,left,right='C:/Users/User/Desktop/Center.jpg','C:/Users/User/Desktop/Left.jpg','C:/Users/User/Desktop/Right.jpg'
pos=None
L=[]
counter=0
counter_now=0
collision=0
lane=0
offroad=0
SECONDS_PER_EPISODE=60*20
episode_start=None
restart=False
epochs=0

REPLAY_MEMORY_SIZE = 100
MIN_REPLAY_MEMORY_SIZE = 17
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 50

DISCOUNT = 0.99
epsilon = 0.001
EPSILON_DECAY = 0.9975 # 0.9975 99975 0.95
MIN_EPSILON = 0.001
CURRENT_COL=0
FIRST=True
SPEED=0
BEST=0


def load_my_model():
    """model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(64,64,3)))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    #model.summary()

    # load weights into new model
    model.load_weights("model-024.h5")
    print("Loaded model from disk")
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))
    return model"""
    """global MINI_WINDOW_WIDTH,MINI_WINDOW_HEIGHT 
    base_model = Xception(weights=None, include_top=False, input_shape=(MINI_WINDOW_HEIGHT, MINI_WINDOW_WIDTH,3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    predictions = Dense(3, activation="linear")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
    return model"""

    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=(MINI_WINDOW_HEIGHT, MINI_WINDOW_WIDTH,3))) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))


    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))

    model.add(Dense(3, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
    model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model





def run_the_server():
    os.chdir("C:/Users/User/Desktop/Carla")
    popen_spawn.PopenSpawn('CarlaUE4.exe -carla-server -benchmark -fps=7 -windowed -ResX=800 -ResY=600')




def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=False,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=0,
        NumberOfPedestrians=0,
        WeatherId=1,
        QualityLevel=args.quality_level)
    #settings.randomize_seeds()
    camera0 = sensor.Camera('Center_Camera')
    camera0.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera0.set_position(0.90, 0, 1.30)
    camera0.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera0)
    camera1 = sensor.Camera('Left_Camera')
    camera1.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera1.set_position(0.90, -1.30, 1.30)
    camera1.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera1)
    camera2 = sensor.Camera('Right_Camera')
    camera2.set_image_size(MINI_WINDOW_WIDTH,MINI_WINDOW_HEIGHT)
    camera2.set_position(0.90, 1.30, 1.30)
    camera2.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera2)

    return settings


class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class CarlaGame(object):
    def __init__(self, carla_client, args): 
        self.client = carla_client
        self._carla_settings = make_carla_settings(args)
        self._timer = None
        self._display = None
        self._center_image = None
        self._left_image = None
        self._right_image = None
        self._enable_autopilot = args.autopilot
        self._lidar_measurement = None
        self._map_view = None
        self._is_on_reverse = False
        self._city_name = args.map_name
        self._map = CarlaMap(self._city_name, 0.1643, 50.0) if self._city_name is not None else None
        self._map_shape = self._map.map_image.shape if self._city_name is not None else None
        self._map_view = self._map.get_map(WINDOW_HEIGHT) if self._city_name is not None else None
        self._position = None
        self._agent_positions = None
        self._m = ''
        self.state=None 
        self.new_state=None

        global REPLAY_MEMORY_SIZE
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE) 
        self.target_update_counter = 0
        self.terminate = False
        self.training_initialized = False
        self.predict=True
        self.action=1
        self.check_for_immobility=True

        
        


    def execute(self, args):
        """Launch the PyGame."""
        global frame
        pygame.init()
        self._initialize_game()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                frame = frame +1
                self._on_loop(args)
                self._on_render(args)
        finally:
            pygame.quit()

    def _initialize_game(self):
        if self._city_name is not None:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH + int((WINDOW_HEIGHT/float(self._map.map_image.shape[0]))*self._map.map_image.shape[1]), WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

        logging.debug('pygame started')
        self._on_new_episode()

    def _on_new_episode(self):
        #self._carla_settings.randomize_seeds()
        #self._carla_settings.randomize_weather()
        scene = self.client.load_settings(self._carla_settings)
        number_of_player_starts = len(scene.player_start_spots)
        player_start = np.random.randint(number_of_player_starts) #109
        #print('Starting new episode...')
        global epochs
        epochs+=1
        print(f"We are in {epochs} epoch.")
        try:
           model.save('C:/Users/User/Desktop/Carla/PythonClient/reinforcement_learning.h5')
           pickle.dump(self.replay_memory, open("C:/Users/User/Desktop/Carla/PythonClient/replay_memory.json", "wb"))
           pickle.dump(epochs, open("C:/Users/User/Desktop/Carla/PythonClient/epochs.json", "wb"))
        except:
          print("Error but its ok!")

        global counter,counter_now
        counter=0
        counter_now=0
        self.check_for_immobility=True
        self.client.start_episode(player_start)
        self._timer = Timer()
        self.episode_timer=Timer()
        self._is_on_reverse = False

        


    def _on_loop(self, args):
        self._timer.tick()
        measurements, sensor_data = self.client.read_data()

        self._center_image = sensor_data.get('Center_Camera', None)
        """self._left_image = sensor_data.get('Left_Camera', None)
        self._right_image = sensor_data.get('Right_Camera', None)"""

        global counter
        if counter==0 and self._center_image is not None:
            #self.state=cv2.resize(image_converter.to_rgb_array(self._center_image), (36, 36))/255
            self.state=image_converter.to_rgb_array(self._center_image)/255



        
        # Print measurements every second.        
        if self._timer.elapsed_seconds_since_lap() > 1: #0.7

            if self._city_name is not None:
                # Function to get car position on map.
                map_position = self._map.convert_to_pixel([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])
                # Function to get orientation of the road car is in.
                lane_orientation = self._map.get_lane_orientation([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])

                self._print_player_measurements_map(
                    measurements.player_measurements,
                    map_position,
                    lane_orientation)                        
            
            else:
                self._print_player_measurements(measurements.player_measurements, args)

            # Plot position on the map as well.

            self._timer.lap()
        
        control = self._get_keyboard_control(pygame.key.get_pressed(), args)
        # Set the player position
        if self._city_name is not None:
            self._position = self._map.convert_to_pixel([
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.location.z])
            self._agent_positions = measurements.non_player_agents
        if control is None:
            self._on_new_episode()
        elif self._enable_autopilot:
            self.client.send_control(measurements.player_measurements.autopilot_control)
        else:
            self.client.send_control(control)
        
    def _get_keyboard_control(self, keys, args):
        global brk
        global spd
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        global its_red
        global back
        global restart
        global STEER


        if keys[K_r] or restart:
            restart=False
            return None
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            control.steer = -1.0
        if keys[K_RIGHT] or keys[K_d]:
            control.steer = 1.0
        if keys[K_UP] or keys[K_w]:
            control.throttle = 1.0
        if keys[K_DOWN] or keys[K_s]:
            control.brake = 1.0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse
            back=not back
            if back:
                print("NOW WE ARE GOING BACK")
            else:
                print("NOW WE ARE GOING FRONT")
        if keys[K_p]:
            self._enable_autopilot = not self._enable_autopilot

        control.steer=STEER 
        control.throttle=0.75
        control.reverse = self._is_on_reverse
        return control

    def _print_player_measurements_map(
            self,
            player_measurements,
            map_position,
            lane_orientation):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += 'Map Position ({map_x:.1f},{map_y:.1f}) '
        message += 'Lane Orientation ({ori_x:.1f},{ori_y:.1f}) '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            map_x=map_position[0],
            map_y=map_position[1],
            ori_x=lane_orientation[0],
            ori_y=lane_orientation[1],
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)


    def _print_player_measurements(self, player_measurements, args):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        global STEER
        global pos 
        global frame,L 
        global counter
        global counter_now
        global SECONDS_PER_EPISODE
        per_frame=5
        loc=player_measurements.transform.location
        pos=(loc.x,loc.y,loc.z)
        STEER=player_measurements.autopilot_control.steer
        th=player_measurements.autopilot_control.throttle
        counter+=1

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        global SPEED 
        if SPEED>10 or transition[4]:
            self.replay_memory.append(transition)

    def get_qs(self, state):
        global model
        return model.predict(np.array([state]))

    def train(self):
        global model,target_model, MIN_REPLAY_MEMORY_SIZE,MINIBATCH_SIZE,TRAINING_BATCH_SIZE,UPDATE_TARGET_EVERY,PREDICTION_BATCH_SIZE,DISCOUNT
        #print(len(self.replay_memory))
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        
        current_qs_list = model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch])
    
        future_qs_list = target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        model.fit(np.array(X), np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False)


        self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            target_model.set_weights(model.get_weights())
            self.target_update_counter = 0



    def train_in_loop(self):
        global model,MINI_WINDOW_WIDTH,MINI_WINDOW_HEIGHT 
        X = np.random.uniform(size=(1, MINI_WINDOW_HEIGHT , MINI_WINDOW_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        model.fit(X,y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

    def get_metrics(self):
        measurements, _ = self.client.read_data()
        collision=measurements.player_measurements.collision_other
        lane=measurements.player_measurements.intersection_otherlane
        offroad=measurements.player_measurements.intersection_offroad
        speed=measurements.player_measurements.forward_speed
        return collision,lane*100,offroad*100,speed*3.6


    def step(self,metrics):
        collision,lane,offroad,speed=metrics
        done=False
        reward=(-lane-offroad)
        global CURRENT_COL,FIRST,SPEED
        SPEED=speed
        seconds=int(self.episode_timer.elapsed_seconds_since_lap())

        if collision>0 or reward<-50 : #or reward<-40
            reward=-200
            done=True
        
        if speed<1 and seconds>=10:
            reward=-200
            done=True
        
        if seconds>SECONDS_PER_EPISODE:
            done = True
            print("20 min is over RESTART")

        if reward==0: 
            reward=10
        if done:
            print(f"The epoch finished in {seconds} seconds.")
        return reward,done

    


        

    def _on_render(self, args):
        global detected_colour
        global frame
        global penalty
        global num
        global STEER 
        global steers
        global center,left,right
        global model
        global pos,state,new_state
        global L
        global epsilon,MIN_EPSILON,EPSILON_DECAY
        per_frame=5
        global counter,counter_now,restart
        
        self._timer.tick()
        gap_x = (WINDOW_WIDTH - 2 * MINI_WINDOW_WIDTH) / 3
        mini_image_y = WINDOW_HEIGHT - MINI_WINDOW_HEIGHT - gap_x
        if self._center_image is not None:
            self.array = image_converter.to_rgb_array(self._center_image)
            surface = pygame.surfarray.make_surface(self.array.swapaxes(0, 1))      

            self._display.blit(surface, (WINDOW_WIDTH/2-MINI_WINDOW_WIDTH/2, WINDOW_HEIGHT/2-MINI_WINDOW_HEIGHT/2))

            if counter>counter_now:
                #self.new_state= cv2.resize(self.array, (36, 36))/255
                self.new_state= self.array/255
                self.predict=True
                reward,restart=self.step(self.get_metrics())
                self.update_replay_memory((self.state, self.action, reward, self.new_state, restart))
                #print((reward,restart))

                if epsilon > MIN_EPSILON:
                    epsilon *= EPSILON_DECAY
                    epsilon = max(MIN_EPSILON, epsilon)

                counter_now+=1
                self.state=self.new_state

            else:
                if self.predict:
                    self.predict=False
                    if np.random.random() > epsilon:
                        # Get action from Q table
                        self.action = np.argmax(self.get_qs(self.state))
                    else:
                        # Get random action
                        self.action = np.random.randint(0, 3)

                    STEER=self.action-1

        

        pygame.display.flip()


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-m', '--map-name',
        metavar='M',
        default=None,
        help='plot the map of the current city (needs to match active map in '
             'server, options: Town01 or Town02)')
    args = argparser.parse_args()

    global model,target_model
    model=load_my_model()#load_model('C:/Users/User/Desktop/Carla/PythonClient/reinforcement_learning.h5')
    target_model=load_my_model()
    target_model.set_weights(model.get_weights())
    #print(model)

    run_the_server()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    
    while True:
        try:

            with make_carla_client(args.host, args.port) as client:
                game = CarlaGame(client, args)
                trainer_thread = Thread(target=game.train_in_loop, daemon=True)
                trainer_thread.start()
                while not game.training_initialized:
                    time.sleep(0.01)   
                game.execute(args)
                game.terminate = True
                trainer_thread.join()
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)
            break
                


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

