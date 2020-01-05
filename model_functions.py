# Imports here
import time
import json
import copy

# Standard functionality and visualisation packages/modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import cv2

# Pytorch specific packages/modules
import torchvision
import torch
from torch import nn #Neural Network modules
from torch import optim #Gradient Descent modules
import torch.nn.functional as F # Additional functions if necessary
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms, models # Importing core structures / frameworks
from collections import OrderedDict
from workspace_utils import active_session

# Import functions used for image processing
import utility_functions

# Model Training Function that returns the trained model
def train_model(model, criteria, optimizer, scheduler, num_epochs, dataset_sizes, data_loaders, device):
    ''' Train a deep learning model for image recognition using Transfer Learning.
    '''
    since = time.time()
    model.to(device) # ensure model is moved to user specified space along with tensors
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criteria(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_acc(model, data_loaders, device):
    ''' Test the accuracy of the trained model in predicting image labels.
    '''
    model.eval() # Notify model layers we are in evaluation mode and not training mode
    model.to(device) # Send to User defined device: GPU for faster processing    
    
    with torch.no_grad(): #Speed up computations and save memory by not back propogating as we are in evaluation mode.
        for idx, (inputs, labels) in enumerate(data_loaders[data]):
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # obtain the outputs from the model
            outputs = model.forward(inputs)
            # max provides the (maximum probability, max value)
            _, predicted = outputs.max(dim=1)
            # check the 
            if idx == 0:
                print(predicted) #the predicted class
                print(torch.exp(_)) # the predicted probability
            equals = predicted == labels.data
            if idx == 0:
                print(equals)
            print(equals.float().mean())
    
    
def save_checkpoint(model, optimizer, image_datasets, arch, epochs, lr, input_size, num_hidden):
    ''' Saves the model constructed with relevant hyperparameters and architecture in place.
    '''
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.cpu()
    checkpoint = {'arch': arch, # Key for the architecture chosen - in this case vgg16
                  'class_to_idx': model.class_to_idx, # This holds the flower image label mappings to compare against our 
                                                      # predictions
                  'model_state_dict': model.state_dict(), # Holds all of the weights and biases of our model for each layer
                  'optimizer': optimizer.state_dict, # Holds the optimiser class chosen in place
                  'input_size': (3, 224, 224), # Image input criteria
                  'output_size': 102, # Number of output classes in final layer
                  'batch_size': 32,  # Batch size for loading images
                  'learning_rate': lr, # Learning rate specified
                  'epochs': epochs, # Number of epochs used
                  'clf_input': input_size, # Number of neurons in the initial layer
                  'hidden_layer_neurons': num_hidden,
                 }
    
    torch.save(checkpoint, 'checkpoint.pth') # This takes a dictionary of model configuration and a path as a destination 

    
def get_checkpoint(checkpoint_path):
    ''' Load a saved model checkpoint and rebuild the model to the saved specification.
    '''
    
    chpt = torch.load(checkpoint_path)
    
    #Checking to see if architecture of model is the same as trained model selected
    if chpt['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        
    elif chpt['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)            
    else:
        print("Sorry base architecture note recognized") #Try-catch
        
    for param in model.parameters(): #Freezing the model parameters
            param.requires_grad = False

    model.class_to_idx = chpt['class_to_idx'] #Applying checkpoint to model class to index mapping
    
    # Create the classifier
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(chpt['clf_input'], chpt['hidden_layer_neurons'])), # Number of neurons is a hyperparameter
                                        ('relu', nn.ReLU()), # ReLU activation function will squish the output space of the 
                                                             # previous layer
                                        ('drop', nn.Dropout(p=0.5)), # Used to prevent overfitting - 
                                                                     # for any node in the network this will equate to 50% chance it will be randomly turned off
                                        ('fc2', nn.Linear(checkpoint['hidden_layer_units'], 102)), # Input 5000, Output layer of 102
                                        ('output', nn.LogSoftmax(dim=1))])) # Squishes the output space to be between 0 - 1 i.e. probability of class assignment 
                                                                            # for each image. Ideal for multiclass problems
    
    # Put the classifier on the pretrained network
    model.classifier = classifier
    
    model.load_state_dict(chpt['model_state_dict'])
    
    return model


def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.to(device)
    
    image = process_image(image_path)
    
    # Convert image to PyTorch tensor first
    image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
     
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    image = image.unsqueeze(0)
    
    output = load_model.forward(image)
    
    probabilities = torch.exp(output)
    
    # Probabilities and the indices of those probabilities corresponding to the classes
    top_probabilities, top_indices = probabilities.topk(topk)
    
    # Convert to lists
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    
    # Convert topk_indices to the actual class labels using class_to_idx
    # Invert the dictionary so you get a mapping from index to class.
    
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    
    top_classes = [idx_to_class[index] for index in top_indices]
    
    return top_probabilities, top_classes