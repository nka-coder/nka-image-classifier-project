#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */ImageClassifier/predict.py
#                                                                             
# PROGRAMMER: Adamou Nchange Kouotou
# DATE CREATED: 31/01/2021                                 
# REVISED DATE: 03/02/2021
# PURPOSE: Identify flower specy from into an image. The user can import  
#          its own trained densenet201 model to perform the prediction.
#
#
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py --checkpoint <Trained densenet201 network. Must be a .pth file> --flower_image <Flower image to identify>
#            
#   Example call:
#    python predict.py --checkpoint densenet201.pth --flower image image_05270.jpg
##

# Imports python modules
from time import time, sleep
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json


# Imports modules created for this program
from workspace_utils import active_session
from model_functions import Classifier, pre_trained_model, predict, load_checkpoint
from model_utilities import get_input_predict, control_input_args_predict


# Main program function defined below
def main():
    
    # Measures total program runtime by collecting start time
    start_time = time()
    
    # Collect training parameter from the console interface
    in_arg = get_input_predict()
       
    # Check consistency of the input parameters    
    control_input_args_predict(in_arg.checkpoint, in_arg.mapping_file, in_arg.arch, in_arg.gpu, int(in_arg.topk))
    
    # Choose betwen GPU and CPU 
    if in_arg.gpu[:1].lower()== 'y':
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(f"\n Running on '{device}' device...\n")
        
    # Predict the class of the entered flower image with the help of the chosen trained model
    probs, classes = predict(in_arg.flower_image, in_arg.checkpoint, in_arg.arch, device, int(in_arg.topk))  
    
    # Determine class with higest probability
    i_max = 0
    prob_max =0
    for i in range(len(probs)):
        if probs[i] > prob_max:
            i_max = i
            prob_max = probs[i]
    
    # Mapping of Class value to category name
    with open(in_arg.mapping_file, 'r') as f:
        cat_to_name = json.load(f)
       
    classes_name = []
    for classe in classes:
        for key, value in cat_to_name.items():
            if str(key) == str(classe) :
                classes_name.append(str(value))
    
    # Printing results of the prediction
    print(f"\nThe network used is for this classification is : {in_arg.checkpoint} \n")
    print(f"Top 5 classes are: {classes} \n ")
    print(f"Top 5 classes names are: {classes_name} \n ")
    print(f"Top 5 classes probabilities  are: {probs} \n ")
    print(f"The classe with highest probability is {classes_name[i_max]} with a probability of  {probs[i_max]} \n ")

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