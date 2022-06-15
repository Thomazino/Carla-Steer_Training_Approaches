# -*- coding: utf-8 -*-
# Import the Stuff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pickle
import cv2
from sklearn.model_selection import train_test_split

import numpy as np

import csv

steers=pickle.load(open("steering.json","rb"))   
# Step1: Read from the log file
def data_loading():
    no_image=[] #[2, 1357, 11975, 26631]
    features=[]
    labels=[]
    photos=['Center','Left','Right']
    steers=pickle.load(open("steering.json","rb"))   
    for i in range(1,24000+1):
        if not i in no_image:
            sample=[]
            for j in range(3):
                sample.append(f'SteerPhotos/{photos[j]}{i}.jpg')
            sample.append(float(steers[i-1]))
            features.append(sample)          
    
    return features


samples= data_loading()
        
# Step2: Divide the data into training set and validation set
train_len = int(0.8*len(samples))
valid_len = len(samples) - train_len
train_samples, validation_samples = data.random_split(samples, lengths=[train_len, valid_len])

# Step3a: Define the augmentation, transformation processes, parameters and dataset for dataloader
def augment(imgName, angle):
  current_image = cv2.imread(imgName)
  current_image = current_image[65:-25, :, :]
  if np.random.rand() < 0.5:
    current_image = cv2.flip(current_image, 1)
    angle = angle * -1.0  
  return current_image, angle

class Dataset(data.Dataset):

    def __init__(self, samples, transform=None):

        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
      
        batch_samples = self.samples[index]
        
        steering_angle = float(batch_samples[3])
        
        center_img, steering_angle_center = augment(batch_samples[0], steering_angle)
        left_img, steering_angle_left = augment(batch_samples[1], steering_angle + 0.4)
        right_img, steering_angle_right = augment(batch_samples[2], steering_angle - 0.4)

        center_img = self.transform(center_img)
        left_img = self.transform(left_img)
        right_img = self.transform(right_img)

        return (center_img, steering_angle_center), (left_img, steering_angle_left), (right_img, steering_angle_right)
      
    def __len__(self):
        return len(self.samples)

# Step3b: Creating generator using the dataloader to parallasize the process
def _my_normalization(x):
    return x/255.0 - 0.5
transformations = transforms.Compose([transforms.Lambda(_my_normalization)])

params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 4}

training_set = Dataset(train_samples, transformations)
training_generator = data.DataLoader(training_set,**params)

validation_set = Dataset(validation_samples, transformations)
validation_generator = data.DataLoader(validation_set, **params)

#pickle.dump([training_set,training_generator,validation_set,validation_generator], open("C:/Users/User/Desktop/Nmodel3/setgen.json", "wb"))

# Step4: Define the network
class NetworkDense(nn.Module):

    def __init__(self):
        super(NetworkDense, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.25)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64 * 2 * 33, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
        
    def forward(self, input):  
        input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output


class NetworkLight(nn.Module):

    def __init__(self):
        super(NetworkLight, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 48, 3, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.25)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=48*4*19, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
        

    def forward(self, input):
        input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output
import copy


def toDevice(datas, device):
  imgs, angles = datas
  return imgs.float().to(device), angles.float().to(device)


def main():
    # Step5: Define optimizer
    model = NetworkLight()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    """checkpoint = torch.load('model323.h5')
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    optimizer.load_state_dict(checkpoint['optimizer'])"""


    """checkpoint = torch.load('model539.h5', map_location=lambda storage, loc: storage)
    model = checkpoint['model']"""
    

    # Step6: Check the device and define function to move tensors to that device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print('device is: ', device)

    # Step7: Train and validate network based on maximum epochs defined
    max_epochs = 1000
    TR_LOSS=[]
    VL_LOSS=[]
    for epoch in range(max_epochs):
        print(f"We are in {epoch} epoch.")
        lc_tr_loss=[]
        lc_vl_loss=[]
        model.to(device)
        
        # Training
        train_loss = 0
        model.train()
        for local_batch, (centers, lefts, rights) in enumerate(training_generator):
            # Transfer to GPU
            centers, lefts, rights = toDevice(centers, device), toDevice(lefts, device), toDevice(rights, device)
            
            # Model computations
            optimizer.zero_grad()
            datas = [centers, lefts, rights]        
            for data in datas:
                imgs, angles = data
    #             print("training image: ", imgs.shape)
                outputs = model(imgs)
                loss = criterion(outputs, angles.unsqueeze(1))
                loss.backward()
                optimizer.step()
                train_loss += loss.data.item()
                
            if local_batch % 5 == 0:
                print('Loss: %.3f '
                     % (train_loss/(local_batch+1)))
                lc_tr_loss.append(train_loss/(local_batch+1))

        
        # Validation
        model.eval()
        valid_loss = 0
        with torch.set_grad_enabled(False):
            for local_batch, (centers, lefts, rights) in enumerate(validation_generator):
                # Transfer to GPU
                centers, lefts, rights = toDevice(centers, device), toDevice(lefts, device), toDevice(rights, device)
            
                # Model computations
                optimizer.zero_grad()
                datas = [centers, lefts, rights]        
                for data in datas:
                    imgs, angles = data
    #                 print("Validation image: ", imgs.shape)
                    outputs = model(imgs)
                    loss = criterion(outputs, angles.unsqueeze(1))
                    
                    valid_loss += loss.data.item()

                if local_batch % 5 == 0:
                    print('Valid Loss: %.3f '
                         % (valid_loss/(local_batch+1)))
                    lc_vl_loss.append(valid_loss/(local_batch+1))

        TR_LOSS.append(lc_tr_loss)
        VL_LOSS.append(lc_vl_loss)
        pickle.dump(TR_LOSS, open("C:/Users/User/Desktop/Nmodel3/train_loss.json", "wb"))
        pickle.dump(VL_LOSS, open("C:/Users/User/Desktop/Nmodel3/valid_loss.json", "wb"))

        # Step8: Define state and save the model wrt to state
        state = {
                'model': model if device == 'cuda' else model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }

        torch.save(state, f'C:/Users/User/Desktop/Nmodel3/model{epoch+1}.h5')



if __name__ == "__main__":
    main()
