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

from PIL import Image

# Pytorch specific packages/modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def data_transforms():
    ''' Define the image  transformations required for the training, validation, and testing sets
    '''
    
    data_transforms = {
    'train': transforms.Compose([ # Notation is used to compose several transforms together using the 'train' keyword
        transforms.RandomRotation(30), #Rotations
        transforms.RandomResizedCrop(224), #Resized cropping
        transforms.RandomVerticalFlip(), # Flipping
        transforms.ToTensor(), # Tensor format
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]) # Normalising colour channel
    ]),
    'valid': transforms.Compose([ # Notation is used to compose several transforms together using the 'val' keyword
        transforms.Resize(256), # Resizing image in line with transfer learning model requirements
        transforms.CenterCrop(224), # Cropping image in line with transfer learning model requirements
        transforms.ToTensor(), # Tensor format
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]) # Normalising colour channel
    ]),
    'test': transforms.Compose([ # Notation is used to compose several transforms together using the 'test' keyword
        transforms.Resize(256), # Resizing image in line with transfer learning model requirements
        transforms.CenterCrop(224), # Cropping image in line with transfer learning model requirements
        transforms.ToTensor(), # Tensor format
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]) # Normalising colour channel
    ]),
}
    return data_transforms

    
def load_datasets(train_dir, valid_dir, test_dir, data_transforms):
    ''' Load the image datasets using the relevant DataLoaders, transformations and directories
    '''
    dirs = {'train': train_dir, 
        'valid': valid_dir, 
        'test' : test_dir}

    image_datasets = {x: datasets.ImageFolder(dirs[x],   transform=data_transforms[x]) for x in ['train', 'valid', 'test']}

# The dataloaders are functions used to load our transformed image datasets in batches for computational effeciency
# There are a couple of hyper parameters available here that can be trained to optimise loading/training/testing time
# Shuffling here simulates generalisation for the model and reduces potential bias

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    
    return image_datasets, dataset_sizes


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    pil_image = Image.open(image_path)
    
    # Resize / Scale
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((10000, 256))
    else:
        pil_image.thumbnail((256, 10000))
    
    # Crop Image
    left_margin = (pil_image.width-224)/2
    bottom_margin = (pil_image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    
    # Normalize
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Move color channels to first dimension as expected by PyTorch
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image


def imshow(image, ax=None, title=None):
    """ Convert a tensor into an image ready format for display.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    if title is not None:
        ax.set_title(title)
    
    ax.imshow(image)
    
    return ax


def load_json(json_file):
    """ Create mappings from json file useful for classification and visualisation 
        of image labels throughout the modelling process.
    """     
    with open('cat_to_name.json', 'r') as f:
        flower_to_name = json.load(f)
    
    return flower_to_name


def display_image(image_dir, flower_to_name, classes):
    """ Displays the image to be classified along with the top 5 probabilities of class assignment.
    """ 

# Plot flower input image
plt.figure(figsize = (6,10))
plot_1 = plt.subplot(2,1,1)

# Process the given image from its file path
image = process_image(image_dir)

# Extract the image title from the file path
key = image_dir.split('/')[-2]

# Set the image title
flower_title = flower_to_name[key]

# Display the image chosen to predict along with its actual title
imshow(image, plot_1, title=flower_title);

# Convert from the class integer encoding to actual flower names
flower_names = [flower_to_name[i] for i in classes]

# Plot the probabilities for the top 5 classes as a bar graph
plt.subplot(2,1,2)

sb.barplot(x=probs, y=flower_names, color=sb.color_palette()[0]);

plt.show()
    