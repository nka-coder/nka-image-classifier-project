#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */ImageClassifier/train.py
#                                                                             
# PROGRAMMER: Adamou Nchange Kouotou
# DATE CREATED: 31/01/2021                                 
# REVISED DATE: 03/02/2021
# PURPOSE: Train the classifier layer of a Densenet201 CNN to improve its 
#          performance in identifying flower specy into an image.
#          Once the model is trained, its accuracy is displayed and the model
#          is saved in the folder named saved_model for future usage.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py --dir <Path to folder of flowers images> --epochs <Number of epochs>
#             --learning_rate <Learning rate for gradient descent> --model <Pre-trained mode> --dropout_rate <Dropout rate for gradient descent>
#   Example call:
#    python train.py --dir flowers --epochs 4 --learning_rate 0.003 --model densenet201 -- dropout_rate 0.2
##


# Imports python modules
from time import time, sleep
from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
import torch.nn.functional as F
import json

# Imports modules created for this program
from workspace_utils import active_session
from model_functions import train_network, evaluate_network, save_checkpoint
from model_utilities import create_dataloader, get_input_train, control_input_args_train


# Main program function defined below
def main():
    
    # Measures total program runtime by collecting start time
    start_time = time()
    
    # Collect training parameter from the console interface
    in_arg = get_input_train()
    
    # Check consistency of the input parameters
    control_input_args_train(in_arg.data_dir, in_arg.save_dir, in_arg.arch, in_arg.hidden_units, in_arg.gpu)
    
    # Choose betwen GPU and CPU 
    if in_arg.gpu[:1].lower()== 'y':
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(f"\n Running on '{device}' device...\n")
    
    # Load training and validation dataloaders
    purpose = 'train'
    train_dataloader = create_dataloader(in_arg.data_dir , purpose)
    purpose = 'valid'
    valid_dataloader = create_dataloader(in_arg.data_dir , purpose)

    #Build and train the classifier network
    model = train_network(device, train_dataloader, valid_dataloader, in_arg.arch, int(in_arg.hidden_units), int(in_arg.epochs), float(in_arg.learning_rate), float(in_arg.dropout_rate))
    
    #Test accuracy of the classifier netwoork build
    purpose = 'test'
    batch_size = 5
    test_dataloader = create_dataloader(in_arg.data_dir, purpose, batch_size)

    result = evaluate_network(model, test_dataloader)
    
    #Saved the built classifier network if accuracy is higher than 40%
    if result > 40:
        dataset = datasets.ImageFolder(in_arg.data_dir +'/train')
        # Attach category attribute to the model to facilitate classe inference during prediction
        model.class_to_idx = dataset.class_to_idx
        saved_model = save_checkpoint(model, in_arg.save_dir, in_arg.arch, result)
        print(f" GOOD PERFORMANCE!!!"
                f" The trained model was saved...{saved_model}"
                f" Test accuracy: {result:.3f} %")
    else:
        print(f" BAD PERFORMANCE!!!"
                f" The trained model was not saved..."
                f" Test accuracy: {result:.3f} %")
    
    # Measure total program runtime by collecting end time
    end_time =time()    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time-start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

    
# Call to main function to run the program
if __name__ == "__main__":
    with active_session():
        main()