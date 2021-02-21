#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */ImageClassifier/model_utility.py
#                                                                             
# PROGRAMMER: Adamou Nchange Kouotou
# DATE CREATED: 31/01/2021                                 
# REVISED DATE: 03/02/2021
# PURPOSE: Module developed to process input parameters used in train.py and predict.py.
#
##

# Imports python modules
from PIL import Image 
import numpy as np
import PIL
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import argparse
import sys
import os.path
from os import path

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # Resize the images where the shortest side is 256 pixels, keeping the aspect ratio.
    img = Image.open(image)
    #img.thumbnail(256)
    
    # Crop out the center 224x224 of img
    width, height = img.size
    img_size = 224
    left = (width - img_size) / 2
    upper = (height - img_size) / 2
    right = (width + img_size) / 2
    lower = (height + img_size) / 2
    img = img.crop((left, upper, right, lower))
    
    # Normalize the image
    np_img = np.array(img)
    np_img = np_img/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean) /std
       
    # PyTorch tensors assume the color channel is the first dimension
    # Return a transposed tensor of the image ndarray
    return torch.from_numpy(np_img.transpose(2,0,1))

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# Load dataset for trainng the model
def create_dataloader(data_dir = 'flowers', purpose = 'train', batchsize = 64):
    """
    Creat image generators. 
    
    Parameters:
     dir_data : dataset of image containing train , valid and test folder of images
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    directory = data_dir + '/' + purpose
    
    if purpose =='train':
        data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]) 
        dataset = datasets.ImageFolder(directory, transform=data_transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle=True)
        
    else:
        data_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
        dataset = datasets.ImageFolder(directory, transform=data_transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle=True)

    return dataloader


def get_input_train():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Path to folder of flowers images as --data_dir 
      2. Path to folder to save the trained model as --save_dir with default value 'saved_models'
      3. Model architechture use as --arch with default value 'densenet201'
      4. Epochs as --epochs with default value '4'
      5. Learning rate for gradient descent as --learn_rate with default value '0.003'
      6. Gradient descent dropout rate as --dropout_rate with default value '0.2'
      7. Choosing to use the gpu or not as --gpu with default value 'y'
      8. Number of hidden units of the classifier as --hidden_units with default value '4120'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """

    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='INPUT DATA TO PROCESS NOW!!!')
    
    # Set parser.parse_args() parsed arguments
    parser.add_argument("data_dir", help="Path to folder of flowers images", type=str)
    parser.add_argument("-sd", "--save_dir", help="Path to folder to save the trained model", type=str, default="saved_models")
    parser.add_argument("-m", "--arch", help="Pre-trained model to use. Choose between densenet201 and densenet121", type=str, default="densenet201")
    parser.add_argument("-e", "--epochs", help="Number of epochs", type=str, default="4")
    parser.add_argument("-lr", "--learning_rate", help="Learning rate for gradient descent optimization of the classifier", type=str, default="0.003")
    parser.add_argument("-do", "--dropout_rate", help="Dropout rate in the classifier during gradient descent", type=str, default="0.2")
    parser.add_argument("-g", "--gpu", help="Choosing to use the gpu or not. Choose between yes/y or no/n", type=str, default="y")
    parser.add_argument("-hu", "--hidden_units", help="Number of hidden units of the classifier", type=str, default="4120")

    return parser.parse_args()

def get_input_predict():
    """
    Retrieves and parses the 2 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 2 command line arguments. If 
    the user fails to provide some or all of the 2 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Trained Network as --checkpoint with default value 'saved_models/densenet201.pth'
      2. Flower image to predict specy as --flower_image with default value 'image_05270.jpg'
      3. Model architechture use as --arch with default value 'densenet201'
      4. Mapping file use as --mapping_file with default value 'cat_to_name.json'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """

    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='INPUT DATA TO PROCESS NOW!!!')
    
    # Set parser.parse_args() parsed arguments
    parser.add_argument("-c", "--checkpoint", help="Checkpoint: trained Densenet201 network used for prediction of specy of a flower. Must be a .pth file", type=str, default='saved_models/densenet201.pth')
    parser.add_argument("-f", "--flower_image", help="Flower image to predict specy", type=str, default='image_05270.jpg')
    parser.add_argument("-m", "--arch", help="Type of the checkpoint. Choose between densenet201 and densenet121", type=str, default="densenet201")
    parser.add_argument("-mf", "--mapping_file", help="JSON file containing mapping of model categories and flower names", type=str, default="cat_to_name.json")
    parser.add_argument("-g", "--gpu", help="Choosing to use the gpu or not. Choose between yes/y or no/n", type=str, default="y")
    parser.add_argument("-tk", "--topk", help="Number of classe with highest probability", type=str, default="5")

    return parser.parse_args()

def control_input_args_train(data_dir, save_dir, arch, hidden_units, gpu):
    """
    Control consistency of the parameter passed in the argparse module of the train.py program
    Parameters:
     data_dir, save_dir, arch, hidden_units, gpu
    Returns:
     None  
    """    
    # Check the device requirement for the execution of the training
    if gpu.lower() not in ['yes','no','y','n'] :
        sys.exit(f"\nWARNING: Enter a consistent --gpu hyperparameter. Read help section to see how to run the program on CPU .\n")
    if not torch.cuda.is_available() and gpu[:1].lower() =='y':
        sys.exit(f"\nWARNING: GPU is not available in this device. Read the manual to see how to run this program on CPU.\n")
        
    # Check if all the input files exist in the directory
    if not path.exists(data_dir):
        sys.exit(f'\nWARNING: The file named "{data_dir}" was not found. Make sure that location exist.\n')
    if not path.exists(save_dir):
        sys.exit(f'\nWARNING: The file named "{save_dir}" was not found. Make sure that you enter the right location.\n')
        
    # Control the number of architecture parameters
    if arch not in ['densenet201', 'densenet121']:
        sys.exit(f'\nWARNING: The architecture you entered is not supported. Read the help section for more details.\n')
    if int(hidden_units) < 1922 and arch == 'densenet201':
        sys.exit(f'\nWARNING: The number of hidden unit should be greater than 1922 for a densenet201 architecture. Please enter a new value.\n')
    if int(hidden_units) < 1026 and arch == 'densenet121':
        sys.exit(f'\nWARNING: The number of hidden unit should be greater than 1026 for a densenet121 architecture. Please enter a new value.\n')
        
        
def control_input_args_predict(checkpoint, mapping_file, arch, gpu, topk): 
    """
    Control consistency of the parameter passed in the argparse module of the train.py program 
    Parameters:
     checkpoint, mapping_file, arch
    Returns:
     None  
    """ 
    # Check the device requirement for the execution of the training
    if gpu.lower() not in ['yes','no','y','n'] :
        sys.exit(f"\nWARNING: Enter a consistent --gpu hyperparameter. Read help section to see how to run the program on CPU .\n")
    if not torch.cuda.is_available() and gpu[:1].lower() =='y':
        sys.exit(f"\nWARNING: GPU is not available in this device. Read the manual to see how to run this program on CPU.\n")
        
    # Check if all the input files exist in the directory
    if not path.exists(checkpoint):
        sys.exit(f'\nWARNING: The file named "{checkpoint}" was not found. Make sure that you enter the right location.\n')
    if not path.exists(mapping_file):
        sys.exit(f'\nWARNING: The file named "{mapping_file}" was not found. Make sure that you enter the right location.\n')
        
    # Control the number of architecture parameters
    if arch not in ['densenet201', 'densenet121']:
        sys.exit(f'\nWARNING: The architecture you entered is not supported. Read the help section for more details.\n')
        
    # Control topk parameter
    if topk not in range(1, 100):
        sys.exit(f'\nWARNING: The value of --topk Hyperparameter must be an integer of the interval [1,100].\n')
        
        