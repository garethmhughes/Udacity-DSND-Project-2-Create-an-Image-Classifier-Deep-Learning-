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

parser = argparse.ArgumentParser(description='Predict Images Using Trained Classifier')

# Command line arguments
parser.add_argument('--image_path', type = str, default = 'flowers/test/30/image_03528.jpg', help = 'Path to image')
parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'Path to checkpoint')
parser.add_argument('--topk', type = int, default = 5, help = 'Top k classes and probabilities')
parser.add_argument('--json', type = str, default = 'flower_to_name.json', help = 'class_to_name json file')
parser.add_argument('--device', type = str, default = 'cuda', help = 'GPU or CPU')

arguments = parser.parse_args()

# Load in a mapping from category label to category name
class_to_name_map = utility_functions.load_json(arguments.json)

# Load pretrained network
model = model_functions.get_checkpoint(arguments.checkpoint)
print(model)  

checkpoint = torch.load(arguments.checkpoint)

# Scales, crops, and normalizes a PIL image for the PyTorch model; returns a Numpy array
image = utility_functions.process_image(arguments.image_path)

# Display image
processing_functions.imshow(image)

# Highest k probabilities and the indices of those probabilities corresponding to the classes (converted to the actual class labels)
probabilities, classes = model_functions.predict(arguments.image_path, model, arguments.topk, arguments.device)  

print(probabilities)
print(classes)

# Display the image along with the top 5 classes
processing_functions.display_image(arguments.image_path, class_to_name_map, classes)