# Standard Imports here
import time
import json
import copy

# Standard functionality and visualisation packages/modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import cv2

# Pytorch/ Deep Learning specific packages/modules
import torchvision
import torch
from torch import nn #Neural Network modules
from torch import optim #Gradient Descent modules
import torch.nn.functional as F # Additional functions if necessary
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms, models # Importing core structures / frameworks
from collections import OrderedDict
from PIL import Image

# Import available functions from support files
import model_functions
import utility_functions

# Allows for interpretation of command line arguments
import argparse

parser = argparse.ArgumentParser(description='Train Image Classification Model')

# Command line arguments
parser.add_argument('--arch', type = str, default = 'vgg', help = 'NN Model Architecture')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning Rate')
parser.add_argument('--gamma_lrsched', type = float, default = 0.01, help = 'Learning Rate for Scheduler')
parser.add_argument('--step_size_sched', type = int, default = 3, help = 'Epoch Step Size for Learning Rate')
parser.add_argument('--num_hidden', type = int, default = 5000, help = 'Neurons in the Hidden Layer')
parser.add_argument('--epochs', type = int, default = 12, help = 'Epochs')
parser.add_argument('--device', type = str, default = 'cuda', help = 'GPU or CPU')
parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'Path to checkpoint')

arguments = parser.parse_args()

#Defining the file paths for each folder set of images in the local directory
data_dir = 'flowers'
train_dir = data_dir + '/train' #flowers/train
valid_dir = data_dir + '/valid' #flowers/valid
test_dir = data_dir + '/test' #flowers/test

# Collecting the transforms for the training, validation, and testing sets
data_transforms = utility_functions.data_transforms()

# Load the datasets with ImageFolder
image_dataset, dataset_sizes = utility_functions.load_datasets(train_dir, valid_dir, test_dir, image_transforms)

# Using the image datasets and the trainforms, define the dataloaders
data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid', 'test']}

# Build and train the neural network (Transfer Learning)
if arguments.arch == 'vgg':
    input_size = 25088
    model = models.vgg16(pretrained=True)
elif arguments.arch == 'alexnet':
    input_size = 9216
    model = models.alexnet(pretrained=True)
    
print(model)


# This ensures we don’t update the weights from the pre-trained model.

for parameter in model.parameters():
    parameter.requires_grad = False

# We want to ensure the final layers are consistent with our problem by using an ordered dictionary.
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, arguments.num_hidden)), # Number of neurons in the hidden layer
                                        ('relu', nn.ReLU()), # ReLU activation function will squish the output space of the previous layer
                                        ('drop', nn.Dropout(p=0.5)), # Used to prevent overfitting - for any node in the network this will equate to 50% chance it will be randomly turned off
                                        ('fc2', nn.Linear(arguments.num_hidden, 102)), # Input 5000, Output layer of 102
                                        ('output', nn.LogSoftmax(dim=1))])) # Squishes the output space to be between 0 - 1 i.e. probability of class assignment for each image. Ideal for multiclass problems

# Ensure we overwrite the model classifier with the newly configured ordered dictionary
model.classifier = classifier

# Setting up the model input arguments (hyperparameters)

# Model, criterion, optimizer, scheduler, num_epochs=25, device='cuda'

# Model is initialised in block above to pretrained vgg16 with classifier adjusted

# Criteria here represents the loss function used to evaluate the model fit 
# NLLLoss which is recommended with Softmax final layer
criteria = nn.NLLLoss()

# Observe that all parameters are being optimized with a learning rate of 0.001 for gradient descent
optim = torch.optim.Adam(model.classifier.parameters(), arguments.learning_rate)

# Provides different methods for adjusting the learning rate and step size used during optimisation
# Decay LR by a factor of 0.1 every 3 epochs
sched = lr_scheduler.StepLR(optim, step_size=arguments.step_size_sched, gamma=arguments.gamma_lrsched)

model_functions.train_model(model, criteria, optim, sched, arguments.epochs, dataset_sizes, data_loaders, arguments.device)
model_functions.test_acc(model, data_loaders, arguments.device)
model_functions.save_checkpoint(model, optim, image_datasets, arguments.arch, arguments.epochs, arguments.learning_rate, input_size, arguments.num_hidden)
