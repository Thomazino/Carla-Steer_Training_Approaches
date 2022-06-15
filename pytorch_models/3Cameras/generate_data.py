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
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from tensorflow.keras.layers import Layer
from keras.initializers import RandomUniform, Initializer, Constant
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from tensorflow.keras.losses import MeanAbsoluteError as MAE
from tensorflow.keras.losses import MeanAbsolutePercentageError as MAPE
import joblib
from keras.models import model_from_json
from math import exp
from tensorflow.keras.activations import tanh
import os
from pexpect import popen_spawn
from keras.models import load_model
import cv2
from PIL import Image 

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from model import *

from pathlib import Path




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

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from sklearn.preprocessing import PolynomialFeatures


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 160

X=[]
y=[] #pickle.load(open("steering.json","rb"))[:NUM]
model=None
STEER=0
thro,multi=0.5,0.7 #0.62
CENTER=np.zeros((160,320,3))
LEFT=None 
RIGHT=None
SAVE=True
AGAIN_RED=False
TURN=False
prob=0.1


def _my_normalization(x):
    return x/255.0 - 0.5

transformations = transforms.Compose([transforms.Lambda(_my_normalization)])

def run_the_server():
    os.chdir("C:/Users/User/Desktop/Carla")
    popen_spawn.PopenSpawn('CarlaUE4.exe -carla-server -benchmark -fps=5 -windowed -ResX=800 -ResY=600')




def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""
    #WeatherId=random.choice([1, 3, 7, 8, 14])
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=False,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=0,
        NumberOfPedestrians=0,
        WeatherId=1,
        QualityLevel=args.quality_level)
    #settings.randomize_seeds()
    camera0 = sensor.Camera('Camera1')
    camera0.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera0.set_position(0.90, 0, 1.30)
    camera0.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera0)
    camera1 = sensor.Camera('Camera2')
    camera1.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera1.set_position(0.90, -1.30, 1.30)
    camera1.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera1)
    camera2 = sensor.Camera('Camera3')
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
        self._main_image = None
        self._mini_view_image1 = None
        self._mini_view_image2 = None
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

    def execute(self):
        """Launch the PyGame."""
        pygame.init()
        self._initialize_game()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                self._on_loop()
                self._on_render()
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
        player_start =np.random.randint(number_of_player_starts) #138
        print(player_start)
        
        print('Starting new episode...')
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False

    def _on_loop(self):
        self._timer.tick()

        measurements, sensor_data = self.client.read_data()

        self._main_image = sensor_data.get('Camera1', None)
        self._mini_view_image1 = sensor_data.get('Camera2', None)
        self._mini_view_image2 = sensor_data.get('Camera3', None)
        
        
        # Print measurements every decisecond.
        if self._timer.elapsed_seconds_since_lap() > 0.1: 

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
                self._print_player_measurements(measurements.player_measurements)

            # Plot position on the map as well.

            self._timer.lap()

        control = self._get_keyboard_control(pygame.key.get_pressed())
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

    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        if keys[K_r]:
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
        if keys[K_p]:
            self._enable_autopilot = not self._enable_autopilot
        control.reverse = self._is_on_reverse
        global thro
        control.throttle=thro
        global STEER 
        control.steer=STEER
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

    def _print_player_measurements(self, player_measurements):
        global X  
        global y  
        global model,linear_model
        global the_bounds
        global prob
        global NUM
        loc=player_measurements.transform.location
        ori=player_measurements.transform.orientation
        rot=player_measurements.transform.rotation
        acc=player_measurements.acceleration
        global CENTER,LEFT,RIGHT 
        global STEER,AGAIN_RED,TURN
        
        
        if random.random()<prob:
            STEER+=random.uniform(-prob, prob)
            

        a = Image.fromarray(CENTER, 'RGB')
        a.save(f'C:/Users/User/Desktop/NEWSteerPhotos/Center{len(y)+1}.jpg')
        a = Image.fromarray(LEFT, 'RGB')
        a.save(f'C:/Users/User/Desktop/NEWSteerPhotos/Left{len(y)+1}.jpg')
        a = Image.fromarray(RIGHT, 'RGB')
        a.save(f'C:/Users/User/Desktop/NEWSteerPhotos/Right{len(y)+1}.jpg')
        y.append(STEER)
        pickle.dump(y, open("C:/Users/User/Desktop/Carla/PythonClient/steering.json", "wb"))
        if len(y)==24000 or (player_measurements.forward_speed*3.6<1 and len(y)>60): exit()

        if not len(y)%10:
            print(f"{len(y)} saved. Prob={prob}")

        if not len(y)%4000:
            prob+=0.1




        

        message = 'Step {step} ({fps:.1f} FPS): '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        #print_over_same_line(message)

    def _on_render(self):
        global CENTER,LEFT,RIGHT
        gap_x = (WINDOW_WIDTH - 2 * MINI_WINDOW_WIDTH) / 3
        mini_image_y = WINDOW_HEIGHT - MINI_WINDOW_HEIGHT - gap_x
        gap_x = (WINDOW_WIDTH - 2 * MINI_WINDOW_WIDTH) / 3
        mini_image_y = WINDOW_HEIGHT - MINI_WINDOW_HEIGHT - gap_x

        if self._main_image is not None:
            array = image_converter.to_rgb_array(self._main_image)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            CENTER=array
            self._display.blit(surface, (WINDOW_WIDTH/2-MINI_WINDOW_WIDTH/2, WINDOW_HEIGHT/2-MINI_WINDOW_HEIGHT))

        if self._mini_view_image1 is not None:
            array = image_converter.to_rgb_array(self._mini_view_image1)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            LEFT=array
            self._display.blit(surface, (gap_x, mini_image_y))

        if self._mini_view_image2 is not None:
            array = image_converter.to_rgb_array(self._mini_view_image2)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            RIGHT=array
            self._display.blit(surface, (2 * gap_x + MINI_WINDOW_WIDTH, mini_image_y))

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

    global model
    checkpoint = torch.load('model.h5')
    model = checkpoint['model']
    model.cuda()
    print(model)

    run_the_server()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)
    


    while True:
        try:

            with make_carla_client(args.host, args.port) as client:
                game = CarlaGame(client, args)
                game.execute()
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
