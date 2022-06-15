# pylint: disable=E1101
from __future__ import print_function
import torch.utils.data as data
from torch._utils import _accumulate
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
import glob
import csv
import random
from plyfile import PlyData, PlyElement
import pdb

def filter_fov(array, degree=180):
  if degree != 180:
    print("FOV different from 180 degrees is not implemented yet.")
    exit()
  res = array[(array < 0)[:,1]]
  return res

def compute_indicator_vector(steerings):
  """
  Computes the steering indicator vector
    length = len(steerings)
    [0,1] if next turn will be a left one
    [1,0] if next turn will be a right one
  """

  # Assign 1 to left curves and 2 to right curves
  steering_direction = np.zeros(steerings.size)
  steering_direction = steering_direction + (steerings < -0.03)
  steering_direction = steering_direction + 2*(steerings > 0.03)

  steering_reversed = steering_direction[::-1] # reverse to process the steerings from behind

  # Define size for steering_indicator vector and set last_direction to 1 as initial default value
  steering_indicator = np.zeros(steerings.size)
  last_direction = 1

  # Iterate over reversed steering vector
  # Always set the steering_indicator to the last steering value different from 0
  for i, direction in enumerate(steering_reversed):
    if direction == 0:
      steering_indicator[i] = last_direction
    else:
      steering_indicator[i] = direction
      last_direction = direction

  # Reverse the steering_indicator to get the original direction back
  # now steering_indicator has a 1 if the next curve will be a left one
  # and 2 if the next curve will be a right one
  steering_indicator = steering_indicator[::-1]

  # bring steering_indicator vector to the desired format:
  # [1,0] if next curve is a left one
  # [0,1] if next curve is a right one
  steering_indicator_onehot = np.zeros(steerings.size * 2, dtype="float32").reshape(-1,2)
  steering_indicator_onehot[
    np.arange(steerings.size),
    np.array(steering_indicator-1, dtype="uint8")] = 1

  return steering_indicator_onehot

def compute_indicator_vector_episode(steerings, num_indicators):
  steerings = np.array(steerings)
  steering_avg = steerings.sum() / steerings.size
  #print("Direction sum:", steering_avg)
  direction =  1 if steering_avg >= 0.05 else (-1 if steering_avg <= -0.05 else 0)
  indicator_vector = [direction] * steerings.size
  return one_hot_indicator(indicator_vector, num_indicators)

def one_hot_indicator(vector, num_indicators):
  #print("Number of indicators: ", num_indicators)
  #print("Before one hot: ", vector)

  # Use numpy to do the transformation
  vector = np.array(vector)
  # Replace 0 with -1 or 1 if only 2 indicators will be used
  # Build one_hot encoded vector
  if(num_indicators == 2):
    vector = np.where(vector == 0, np.random.randint(2, size=(vector.size,))*2-1, vector)
    # Rescale vector to have 0 if left and 1 if right
    vector = (vector+1)//2
    one_hot = np.zeros(vector.size * 2, dtype="float32").reshape(-1,2)
    one_hot[np.arange(vector.size),vector] = 1
    #print("After one hot", one_hot)
  else:
    one_hot = [[0,0]] * len(vector)
    for i in range(len(vector)):
      if(vector[i] == 1):
        one_hot[i] = [0,1]
      if(vector[i] == -1):
        one_hot[i] = [1,0]
  return np.array(one_hot, dtype="float32")

  #---------
  #print("Before one hot: ", vector)
  # #print("After one hot", np.array(one_hot))
  # return np.array(one_hot, dtype="float32")
  #---------


def get_subdir(root):
  return [os.path.join(root, name) for name in os.listdir(root) if os.path.isdir(os.path.join(root,name))]

class SteeringDataset(data.Dataset):
  """
  Loads the driving dataset consisting of point cloud and steering data.
  Root directory of datset should contain
    - a folder PointCloudLocal0 containing the point clouds in .ply format
    - a file driving_data.csv containing the file names in column 1 
      and the respective steering angle (between -1 and 1) in column 4
  Arguments:
    root (string): Path of the root directory of the dataset.
    npoints (number): Number of points that will be sampled per point cloud
  """
  def __init__(self,
         root,
         npoints=2500,
         data_augmentation=True,
         running_mean=False,
         steering_indicator=False,
         num_steering_indicators=2,
         no_norm=False,
         predefined_steering_indicator=False,
         lidar_fov=360):
    self.no_norm = no_norm
    self.npoints = npoints
    self.root = root
    self.data_augmentation = data_augmentation
    self.steering_indicator = steering_indicator
    self.predefined_steering_indicator = predefined_steering_indicator
    self.lidar_fov=lidar_fov
    #print("Data augmentation: ", self.data_augmentation)
    #print("no_norm:", self.no_norm)
    print("Root: {}, Steering: {}".format(root, predefined_steering_indicator))

    # Store all steering angles in self.steerings and respective filenames in images
    self.steerings = np.array([], dtype="float32")
    self.images = []
    with open(os.path.join(self.root, 'driving_data.csv'), 'r') as driving_data:
      csv_reader = csv.reader(driving_data, delimiter=',')
      for index, row in enumerate(csv_reader):
        if(index != 0):
          self.images.append(row[1])
          self.steerings = np.append(self.steerings, np.float32(row[4]))
    
    # Compute the running mean
    if running_mean:
      print("Steering before running mean:", self.steerings.dtype)
      self.steerings = np.convolve(self.steerings, np.ones((running_mean,), dtype="float32")/running_mean, mode='same')
      print("Steering after running mean:", self.steerings.dtype)

    if self.steering_indicator:
      if predefined_steering_indicator:
        self.steering_indicator_vector = [self.predefined_steering_indicator] * len(self.images)
        self.steering_indicator_vector = one_hot_indicator(self.steering_indicator_vector, num_steering_indicators)
      else:
        self.steering_indicator_vector = compute_indicator_vector_episode(self.steerings, num_steering_indicators)
      
      print("Steering indicator vector: ", self.steering_indicator_vector[0])
      # save targets in dict to allow quick access
      self.targets = dict(zip(self.images, zip(self.steerings, self.steering_indicator_vector)))
    else:
      # save targets in dict to allow quick access
      self.targets = dict(zip(self.images, self.steerings))
    
    # storing a list of ply file names in self.fns
    self.fns = sorted(glob.glob(os.path.join(root, "*.ply"))) 

  def __getitem__(self, index):
    # Get the filename of the pointcloud at that index
    fn = self.fns[index]
    index_formatted = "{:0>5d}".format(int(''.join(filter(str.isdigit, os.path.basename(fn)))))
    with open(fn, 'rb') as f:
      plydata = PlyData.read(f)
    pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    
    # only use front 180 degrees
    if self.lidar_fov == 180:
      pts = filter_fov(pts)
    
    try:
      choice = np.random.choice(len(pts), self.npoints, replace=True)
      point_set = pts[choice, :]
    except:
      print("ERROR")
      point_set=np.array([[-1.72,-1.32,8.46]
,[-1.88,-1.29,2.11]
,[-1.83,-1.31,1.95]
,[1.83,-1.33,3.56]
,[2.38,-1.33,2.5]
,[1.93,-1.33,2.57]
,[1.91,-1.3,2.59]
,[2.08,-1.31,3.8]
,[2.17,-1.34,3.56]
,[-1.15,-1.44,18.48]
,[-1.78,-1.33,4.74]
,[-1.89,-1.29,1.92]
,[-1.37,-1.37,9.77]
,[-1.73,-1.33,5.02]
,[5.99,-1.41,11.27]
,[5.52,-1.44,15.36]
,[-1.87,-1.31,2.71]
,[-1.77,-1.32,2.91]
,[1.9,-1.32,2.29]
,[1.89,-1.32,2.01]
,[2.2,-1.3,2.52]
,[2.11,-1.33,2.5]
,[2.08,-1.31,3.8]
,[2.3,-1.36,6.7]
,[2.26,-1.31,4.68]
,[2.46,-1.33,6.55]
,[-1.79,-1.31,2.33]
,[2.52,-1.33,3.04]
,[-1.74,-1.32,3.38]
,[1.92,-1.4,11.16]
,[5.66,-1.51,24.16]
,[2.76,-1.38,8.82]
,[5.58,-1.47,18.78]
,[5.63,-1.41,11.26]
,[-1.74,-1.32,3.01]
,[1.99,-1.36,6.69]
,[7.01,-1.5,32.03]
,[1.66,-1.43,15.21]
,[2.14,-1.32,5.28]
,[2.31,-1.32,5.28]
,[2.32,-1.3,2.52]
,[1.95,-1.33,3.04]
,[2.3,-1.3,2.37]
,[2.24,-1.31,3.99]
,[2.06,-1.3,2.87]
,[1.92,-1.34,3.72]
,[-1.37,-1.37,21.88]
,[1.98,-1.35,5.77]
,[-1.89,-1.27,1.89]
,[5.69,-1.39,8.89]
,[-1.8,-1.31,2.21]
,[1.85,-1.34,4.08]
,[2.22,-1.3,2.68]
,[1.94,-1.29,2.18]
,[2.2,-1.33,2.42]
,[6.36,-1.36,9.69]
,[1.95,-1.3,2.97]
,[1.87,-1.37,7.97]
,[2.34,-1.34,7.12]
,[1.92,-1.3,2.37]
,[2.2,-1.33,2.66]
,[2.48,-1.33,2.84]
,[6.43,-1.32,10.55]
,[6.35,-1.33,9.45]
,[-1.8,-1.33,5.01]
,[-1.82,-1.28,4.31]
,[6.25,-1.44,15.38]
,[-1.58,-1.35,7.21]
,[-1.71,-1.37,21.87]
,[6.27,-1.34,6.58]
,[2.77,-1.39,9.87]
,[2.23,-1.3,2.3]
,[1.94,-1.29,1.97]
,[2.2,-1.33,2.66]
,[2.49,-1.33,2.74]
,[1.92,-1.29,2.24]
,[6.28,-1.3,6.93]
,[2.34,-1.34,7.12]
,[1.91,-1.3,2.59]
,[2.54,-1.35,4.79]
,[2.1,-1.32,2.17]
,[2.32,-1.3,2.52]
,[3.03,-1.41,12.91]
,[-1.75,-1.27,10.18]
,[2.62,-1.35,4.79]
,[2.56,-1.4,14.91]
,[-1.85,-1.24,5.65]
,[3.09,-1.54,32.91]
,[1.86,-1.33,2.42]
,[1.9,-1.35,5.07]
,[5.97,-1.38,7.34]
,[-1.85,-1.32,3.38]
,[1.91,-1.32,2.35]
,[-1.59,-1.39,12.75]
,[1.99,-1.33,6.06]
,[-1.78,-1.29,6.33]
,[5.7,-1.37,6.75]
,[2.11,-1.32,2.29]
,[-1.81,-1.3,4.63]
,[-1.86,-1.26,2.77]
,[6.06,-1.37,6.26]
,[1.92,-1.33,3.16]
,[2.06,-1.3,2.87]
,[2.77,-1.39,9.87]
,[2.45,-1.34,3.56]
,[-1.78,-1.31,2.47]
,[-0.37,-1.49,23.82]
,[-1.76,-1.32,7.04]
,[1.93,-1.33,2.93]
,[-1.84,-1.29,11.75]
,[2.24,-1.3,2.87]
,[1.94,-1.36,6.2]
,[3.15,-1.4,11.19]
,[2.2,-1.32,2.23]
,[2.62,-1.35,5.78]
,[1.95,-1.33,3.56]
,[2.23,-1.3,2.6]
,[2.23,-1.31,4.19]
,[2.49,-1.33,2.74]
,[1.87,-1.33,2.5]
,[-1.88,-1.27,2.08]
,[2.14,-1.32,5.28]
,[1.87,-1.32,2.06]
,[2.,-1.33,7.11]
,[-1.74,-1.32,3.01]
,[-1.83,-1.31,2.55]
,[3.38,-1.5,24.04]
,[2.43,-1.33,3.04]
,[1.88,-1.34,4.29]
,[2.12,-1.33,3.16]
,[1.88,-1.33,2.74]
,[2.08,-1.33,6.06]
,[1.94,-1.34,3.89]
,[2.28,-1.42,18.22]
,[2.12,-1.33,2.42]
,[2.76,-1.38,12.62]
,[-1.81,-1.33,3.86]
,[-1.73,-1.33,4.25]
,[-1.8,-1.32,3.38]
,[1.88,-1.34,4.29]
,[2.29,-1.3,2.87]
,[-1.88,-1.26,8.03]
,[-1.73,-1.36,7.9]
,[1.78,-1.36,6.69]
,[6.28,-1.3,6.93]
,[6.49,-1.3,9.23]
,[1.97,-1.29,2.07]
,[1.9,-1.32,2.29]
,[1.84,-1.36,6.2]
,[1.91,-1.34,4.08]
,[2.07,-1.3,3.48]
,[2.47,-1.32,5.65]
,[2.16,-1.33,2.94]
,[2.49,-1.33,2.74]
,[2.31,-1.32,5.28]
,[1.81,-1.34,4.29]
,[1.99,-1.29,2.12]
,[2.49,-1.33,2.66]
,[-1.43,-1.43,18.32]
,[1.88,-1.36,6.69]
,[1.9,-1.33,3.04]
,[2.22,-1.3,3.09]
,[-1.86,-1.31,2.47]
,[6.6,-1.29,9.18]
,[-1.82,-1.24,4.65]
,[-1.87,-1.27,2.55]
,[1.93,-1.32,2.29]
,[-1.86,-1.32,2.9]
,[2.11,-1.35,9.63]
,[-0.52,-1.57,33.39]
,[-1.84,-1.32,3.68]
,[-1.73,-1.32,3.25]
,[5.45,-1.41,11.25]
,[-1.91,-1.27,10.18]
,[1.82,-1.36,7.27]
,[2.3,-1.3,2.77]
,[-1.87,-1.27,2.71]
,[-1.86,-1.32,2.9]
,[1.88,-1.32,2.12]
,[-1.84,-1.26,3.68]
,[1.93,-1.33,2.93]
,[1.95,-1.3,2.77]
,[6.67,-1.3,11.86]
,[2.23,-1.3,2.97]
,[-1.73,-1.33,4.25]
,[2.1,-1.33,2.36]
,[2.29,-1.3,2.87]
,[-1.83,-1.32,2.72]
,[-1.84,-1.32,3.01]
,[2.36,-1.31,4.2]
,[1.88,-1.32,2.23]
,[2.04,-1.3,2.97]
,[-1.8,-1.31,2.1]
,[-1.89,-1.3,1.98]
,[6.35,-1.33,9.45]
,[2.26,-1.3,2.37]
,[1.87,-1.33,2.65]
,[-1.83,-1.32,4.03]
,[1.94,-1.34,3.89]
,[6.28,-1.3,6.93]
,[1.84,-1.33,2.74]
,[-1.89,-1.3,1.89]
,[6.17,-1.47,18.81]
,[1.91,-1.32,1.97]
,[-1.82,-1.3,4.17]
,[2.13,-1.32,2.23]
,[2.39,-1.32,5.28]
,[2.29,-1.3,2.44]
,[-1.89,-1.27,1.98]
,[2.55,-1.34,4.09]
,[6.88,-1.33,14.2]
,[1.78,-1.36,6.69]
,[2.23,-1.3,2.97]
,[-1.73,-1.32,3.68]
,[-1.78,-1.29,6.33]
,[-1.81,-1.31,2.]
,[2.3,-1.3,2.68]
,[3.23,-1.41,12.92]
,[2.23,-1.31,3.48]
,[2.39,-1.33,2.43]])
      

    if not self.no_norm:
      point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center    
      dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
      point_set = point_set / dist  # scale

    if self.data_augmentation:
      theta = np.random.uniform(0, np.pi * 2)
      rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
      point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
      point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

    point_set = torch.from_numpy(point_set.astype(np.float32))

    # with open(os.path.join(self.root, 'driving_data.csv'), 'r') as driving_data:
    #   csv_reader = csv.reader(driving_data, delimiter=',')
    #   for row in csv_reader:
    #     if row[1] == 'image_' + index_formatted:
    #       steering_angle = np.float32(row[4])

    # Get the target steering angle (and steering_indicator if asked for)
    if self.steering_indicator:
      steering_angle, steering_indicator_value = self.targets['image_' + index_formatted]
      return point_set, steering_angle, steering_indicator_value
      
    else:
      steering_angle = self.targets['image_' + index_formatted]
      return point_set, steering_angle


  def __len__(self):
    return len(self.fns)

  def get_steering_vector(self):
    """
    Returns all steering values of the dataset as a list
    Mainly used for BalancedSteeringSampler
    """
    return self.steerings

  def get_steering(self, index):
    """
    Return steering angle for a given index
    """
    return self.steerings[index]

  def get_steering_indicator(self, index):
    """
    Returns the steering indicator for this episode
    WARNING: Only use when the episode only contains one situation
    """
    return self.steering_indicator_vector[index]
  
  def get_steering_indicator_vector(self):
    return self.steering_indicator_vector


class SteeringDatasetEpisodes(data.Dataset):
  """
  Manages multiple episodes to be trained on as one dataset
  """
  def __init__(self,
         root,
         npoints=2500,
         data_augmentation=True,
         running_mean=False,
         episodes=None,
         steering_indicator=False,
         num_steering_indicators=2,
         no_norm=False,
         lidar_fov=360):
    self.episodes = episodes
    self.datasets = []
    self.lengths = []
    self.steering_indicator = steering_indicator
    for ep in episodes:
      self.datasets.append(
        SteeringDataset(
          os.path.join(root,'episode_{:0>3d}'.format(ep)),
          npoints=npoints,
          data_augmentation=data_augmentation,
          running_mean=running_mean,
          steering_indicator=self.steering_indicator,
          no_norm=no_norm,
          num_steering_indicators=num_steering_indicators,
          lidar_fov=lidar_fov
          )
      )
      self.lengths.append(len(self.datasets[-1]))
    self.cumsum = np.cumsum(self.lengths)

  def __getitem__(self, index):
    dataset = np.argmax(self.cumsum > index)
    dataset_index = index - self.cumsum[dataset-1] if dataset > 0 else index
    return self.datasets[dataset][dataset_index]


  def __len__(self):
    return self.cumsum[-1]

  def get_steering_vector(self):
    """
    Returns all steering values of the dataset as a list
    Mainly used for BalancedSteeringSampler
    """
    steering = []
    for ep in range(len(self.episodes)):
      steering.extend(self.datasets[ep].get_steering_vector())
    return steering

  def get_steering(self, index):
    dataset = np.argmax(self.cumsum > index)
    dataset_index = index - self.cumsum[dataset-1] if dataset > 0 else index
    return self.datasets[dataset].get_steering(dataset_index)

  def get_steering_indicator(self, index):
    dataset = np.argmax(self.cumsum > index)
    dataset_index = index - self.cumsum[dataset-1] if dataset > 0 else index
    return self.datasets[dataset].get_steering_indicator(dataset_index)

  def get_steering_indicator_vector(self):
    """
    Returns all steering indicator values of the dataset as a list
    Mainly used for visualization
    """
    steering_indicator_vector = []
    for ep in range(len(self.episodes)):
      steering_indicator_vector.extend(self.datasets[ep].get_steering_indicator_vector())
    return steering_indicator_vector


class SteeringDatasetNested(data.Dataset):
  """
  Manages multiple nested episodes to be trained on as one dataset
  """
  def __init__(self,
         root,
         npoints=2500,
         data_augmentation=True,
         running_mean=False,
         levels=1,
         steering_indicator=False,
         num_steering_indicators=2,
         no_norm=False,
         data_type='point_clouds',
         lidar_fov=360):

    # Get all subdirectories
    if levels > 0:
      self.dataset_roots = get_subdir(root)
      for level in range(levels-1):
        self.dataset_roots = [subdir for name in self.dataset_roots for subdir in get_subdir(name)]
    else:
      self.dataset_roots = [root]
    print(self.dataset_roots)
    self.datasets = []
    self.lengths = []
    self.steering_indicator = steering_indicator
    for dataset_root in self.dataset_roots:
      if num_steering_indicators == 2:
        if "left" in dataset_root:
          predefined_steering_indicator = -1
        elif "right" in dataset_root:
          predefined_steering_indicator = 1
        else:
          predefined_steering_indicator = 0
      else:
        predefined_steering_indicator = False

      self.datasets.append(
        SteeringDataset(
          root=os.path.join(dataset_root, data_type),
          npoints=npoints,
          data_augmentation=data_augmentation,
          running_mean=running_mean,
          steering_indicator=self.steering_indicator,
          no_norm=no_norm,
          predefined_steering_indicator=predefined_steering_indicator,
          num_steering_indicators=num_steering_indicators,
          lidar_fov=lidar_fov
          )
      )
      self.lengths.append(len(self.datasets[-1]))
    self.cumsum = np.cumsum(self.lengths)

  def __getitem__(self, index):
    dataset = np.argmax(self.cumsum > index)
    dataset_index = index - self.cumsum[dataset-1] if dataset > 0 else index
    return self.datasets[dataset][dataset_index]


  def __len__(self):
    return self.cumsum[-1]

  def get_steering_vector(self):
    """
    Returns all steering values of the dataset as a list
    Mainly used for BalancedSteeringSampler
    """
    steering = []
    for dataset in self.datasets:
      steering.extend(dataset.get_steering_vector())
    return steering
  
  def get_steering(self, index):
    dataset = np.argmax(self.cumsum > index)
    dataset_index = index - self.cumsum[dataset-1] if dataset > 0 else index
    return self.datasets[dataset].get_steering(dataset_index)

  def get_steering_indicator(self, index):
    dataset = np.argmax(self.cumsum > index)
    dataset_index = index - self.cumsum[dataset-1] if dataset > 0 else index
    return self.datasets[dataset].get_steering_indicator(dataset_index)
    # dataset = np.argmax(self.cumsum > index)
    # dataset_index = index - self.cumsum[dataset-1] if dataset > 0 else index
    # if "left" in self.dataset_roots[dataset]:
    #   return -1
    # elif "right" in self.dataset_roots[dataset]:
    #   return 1
    # else:
    #   return 0
  
  def get_steering_indicator_vector(self):
    """
    Returns all steering indicator values of the dataset as a list
    Mainly used for visualization
    """
    steering_indicator_vector = []
    for dataset in self.datasets:
      steering_indicator_vector.extend(dataset.get_steering_indicator_vector())
    return steering_indicator_vector
  
  

class SteeringDatasetCombined(data.Dataset):
  """
  Manages multiple nested episodes to be trained on as one dataset
  """
  def __init__(self,
         roots = [], # list in the form [(root, level)*]
         npoints=2500,
         data_augmentation=False,
         running_mean=False,
         steering_indicator=False,
         no_norm=False,
         num_steering_indicators=2,
         data_type='data_type',
         lidar_fov=360):
    print("Setting up steering dataset combined")
    self.datasets = []
    self.lengths = []
    for root, levels in roots:
      self.datasets.append(
        SteeringDatasetNested(
          root=root,
          npoints=npoints,
          running_mean=running_mean,
          levels=levels,
          steering_indicator=steering_indicator,
          data_augmentation=data_augmentation,
          no_norm=no_norm,
          num_steering_indicators=num_steering_indicators,
          data_type=data_type,
          lidar_fov=lidar_fov
        )
      )
      self.lengths.append(len(self.datasets[-1]))
    self.cumsum = np.cumsum(self.lengths)

  def __getitem__(self, index):
    dataset = np.argmax(self.cumsum > index)
    dataset_index = index - self.cumsum[dataset-1] if dataset > 0 else index
    return self.datasets[dataset][dataset_index]

  def __len__(self):
    return self.cumsum[-1]

  def get_steering_vector(self):
    """
    Returns all steering values of the dataset as a list
    Mainly used for BalancedSteeringSampler
    """
    steering = []
    for ep in range(len(self.datasets)):
      steering.extend(self.datasets[ep].get_steering_vector())
    return steering

  def get_steering(self, index):
    dataset = np.argmax(self.cumsum > index)
    dataset_index = index - self.cumsum[dataset-1] if dataset > 0 else index
    return self.datasets[dataset].get_steering(dataset_index)
  
  def get_steering_indicator(self, index):
    dataset = np.argmax(self.cumsum > index)
    dataset_index = index - self.cumsum[dataset-1] if dataset > 0 else index
    return self.datasets[dataset].get_steering_indicator(dataset_index)
  
  def get_steering_indicator_vector(self):
    """
    Returns all steering indicator values of the dataset as a list
    Mainly used for visualization
    """
    steering_indicator_vector = []
    for dataset in self.datasets:
      steering_indicator_vector.extend(dataset.get_steering_indicator_vector())
    return steering_indicator_vector

def random_split(dataset, lengths):
  """
  Randomly split a dataset into non-overlapping new datasets of given lengths.
  Arguments:
    dataset (Dataset): Dataset to be split
    lengths (sequence): lengths of splits to be produced
  """
  if sum(lengths) != len(dataset):
    raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

  indices = torch.randperm(sum(lengths)).tolist()
  datasets = [data.dataset.Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]
  steering = dataset.get_steering_vector()

  return {
    "datasets": datasets,
    "steering": [steering[indices[i]] for i in range(lengths[0])]
  }

if __name__ == '__main__':
  dataset = sys.argv[1]
  datapath = sys.argv[2]

  if dataset == 'steering':
    d = SteeringDataset(root=datapath)
    print(len(d))
    print(d[0])