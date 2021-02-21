#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */ImageClassifier/model_functions.py
#                                                                             
# PROGRAMMER: Adamou Nchange Kouotou
# DATE CREATED: 31/01/2021                                 
# REVISED DATE: 03/02/2021
# PURPOSE: Module developed to train and use models in train.py and predict.py.
#
##

# Imports python modules
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import sys

# Imports modules created for this program
from model_utilities import process_image, imshow


# Define the pre-trained model
def pre_trained_model(checkpoint_name = 'densenet201'):
    #Load pre-trained model
    if (checkpoint_name.find('densenet201') != -1): 
        x = models.densenet201(pretrained=True)
        model_name = 'densenet201'
        
    if (checkpoint_name.find('densenet121') != -1): 
        x = models.densenet121(pretrained=True)
        model_name = 'densenet121'
     
    return x
        
    
# Define the classifier
class Classifier(nn.Module):
    def __init__(self, model_type = 'densenet201', hidden_units = 4120, dropout_rate = 0.2):
        super().__init__()
        
        # dem allow to split hidden units to the two layers. Aproximately (dem-1)/ dem in layer1 and 1/dem in layer2
        dem = 3
        if (model_type.lower().find('densenet201') != -1): 
            nb_outputlayer_units = ((hidden_units -1920) // dem)*(dem-1) + (hidden_units -1920) % dem
            self.fc1 = nn.Linear(1920, nb_outputlayer_units)
            nb_inputlayer_units = nb_outputlayer_units 
            nb_outputlayer_units = (hidden_units -1920) // dem 
            self.fc2 = nn.Linear(nb_inputlayer_units, nb_outputlayer_units)
            nb_inputlayer_units = nb_outputlayer_units 
            self.fc3 = nn.Linear(nb_inputlayer_units, 102)
        
        if (model_type.lower().find('densenet121') != -1): 
            nb_outputlayer_units = ((hidden_units -1024) // dem)*(dem-1) + (hidden_units -1024) % dem
            self.fc1 = nn.Linear(1024, nb_outputlayer_units)
            nb_inputlayer_units = nb_outputlayer_units 
            nb_outputlayer_units = (hidden_units -1024) // dem 
            self.fc2 = nn.Linear(nb_inputlayer_units, nb_outputlayer_units)
            nb_inputlayer_units = nb_outputlayer_units 
            self.fc3 = nn.Linear(nb_inputlayer_units, 102)            
        
        # Dropoup feature
        self.dropout = nn.Dropout(p = dropout_rate)
        
    def forward(self, x):   
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        
        return x

# Loads a checkpoint and rebuilds the model
def load_checkpoint(file_path, model_type):
    
    try:
        state_dict = torch.load(file_path, map_location=lambda storage, loc: storage)
    except:
        sys.exit(f"\nERROR: The Checkpoint file '{file_path}' is not consistent with this program. Please, enter a consistent Checkpoint file\n")
        
    try:
        md = pre_trained_model(state_dict['pre-trained model'])
    except ERROR:
        sys.exit("\nERROR: The Checkpoint file is not consistent. 'pre-trained model' (Achitecture) is missing\n")
    
    for param in md.parameters():
        param.requires_grad = False
        
    try:
        md.classifier = Classifier(model_type, state_dict['number of hidden units'])  
    except:
        sys.exit("\nERROR: The Checkpoint file is not consistent. 'number of hidden units' is missing\n")
        
    try:
        md.load_state_dict(state_dict['state_dict'])
    except:
        sys.exit("\nERROR: The Checkpoint file is not consistent. 'state_dict' is missing\n")
            
    try:
        md.class_to_idx = state_dict['class_to_idx']
    except:
        sys.exit("\nERROR: The Checkpoint file is not consistent. 'class_to_idx' is missing\n")
    
    return md
    
    
# Specy prediction    
def predict(image_path, model_dir, model_type, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    
    md = load_checkpoint(model_dir, model_type)
        
    md.eval()
    img = process_image(image_path)
    img.unsqueeze_(0)
    
    log_output = md.forward(img.float())
    output = torch.exp(log_output)

    top_output, top_class = output.topk(topk, dim=1)
    
    top_category=[]
    for i in range(len(top_class.tolist()[0])):
        for key, value in md.class_to_idx.items():
            if top_class.tolist()[0][i] == value:
                top_category.append(int(key))
    
    return top_output.tolist()[0], top_category

# Train the Network
def train_network(device, train_dataloader, valid_dataloader, model_name, hidden_units, epochs = 4, learn_rate = 0.003, dropout_rate = 0.2):
  
    # Load pre-trained model
    model = pre_trained_model(model_name)

    # Turn off gradient for our pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Update the classifier module of the pre-trained model
    model.classifier = Classifier(model_name, hidden_units, dropout_rate)   

    # Define the loss function. We chosse NLLLoss because the output layer of our classifier is Log_softmax function
    criterion = nn.NLLLoss()

    # Turn on classifier layer gradients. 
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    # Move model to gpu if it is available otherwise, it remain in cpu
    model.to(device)


    ## Use a pretrained model to classify the cat and dog images
    steps = 0
    running_loss = 0
    print_every = 32

    for epoch in range(epochs):
        for image, label in train_dataloader:
            steps += 1
            # Move image and label tensors to the default device
            image, label = image.to(device), label.to(device)
        
            optimizer.zero_grad()     
            log_ps = model.forward(image)
            loss = criterion(log_ps, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
            
                # Stop gradient back propagation
                model.eval()
                with torch.no_grad():
                    for image, label in valid_dataloader:
                        # Move image and label tensors to the default device
                        image, label = image.to(device), label.to(device)
                        
                        log_ps = model.forward(image)
                        batch_loss = criterion(log_ps, label)                 
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == label.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(valid_dataloader):.3f}.. "
                  f"Test accuracy: {100*accuracy/len(valid_dataloader):.3f} %")
                running_loss = 0
                # Re-start gradient back propagation
                model.train() 
    print("Done...")
    
    model.name = model_name
    model.hidden_units = hidden_units
    model.device = device
    
    return model

# Evaluate accuracy of the network
def evaluate_network(model, test_dataloader):
    # Enable switch from cpu to gpu if gpu is avalable 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    accuracy = 0
    sample_size = 4*len(test_dataloader)
    
    for image, label in test_dataloader:
        
        if sample_size == 0:
            break
        # Move image and label tensors to the default device
        image, label = image.to(device), label.to(device)
            
        # Using the model to predict image label
        log_output = model.forward(image)
            
        # Calculate accuracy
        output = torch.exp(log_output)
        top_output, top_class = output.topk(1, dim=1)
        equals = top_class == label.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        sample_size -=1
    
    return 100*accuracy/len(test_dataloader)

# Save the checkpoint 
def save_checkpoint(model, directory = '', model_type ='densenet201', accuracy = ''):
    
    if (model_type.lower().find('densenet201') != -1): 
        input_size = 1920
        
    if (model_type.lower().find('densenet121') != -1): 
        input_size = 1024
    
    checkpoint = {'input_size': input_size,
             'output_size': 102,
             'class_to_idx': model.class_to_idx,
             'pre-trained model': model.name,
              'device': model.device,
              'number of hidden units': model.hidden_units,
             'state_dict': model.state_dict()}
    
        
    path = model_type + '_' + str(int(accuracy)) + ".pth"

    torch.save(checkpoint, directory +'/'+ path)
    
    return path