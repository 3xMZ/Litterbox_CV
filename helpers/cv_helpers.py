import cv2
import datetime
import os
import serial
import shutil
import time
import torch

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
import seaborn as sb

from collections import OrderedDict
from PIL import Image
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from typing import Tuple

from skimage.metrics import structural_similarity as compare_ssim


def load_checkpoint(filepath: str) -> models:
    """ Reload a saved .pth model."""
    checkpoint = torch.load(filepath, map_location='cpu')

    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Architecture not recognized.")

    model.class_to_idx = checkpoint['class_to_idx']
    classifier = nn.Sequential(
        OrderedDict([('fc1', nn.Linear(25088, 5000)),
                    ('relu', nn.ReLU()),
                    ('drop', nn.Dropout(p=0.5)),
                    ('fc2', nn.Linear(5000, 102)),
                    ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def process_image(image_path: str) -> np.array:
    ''' Image preprocessor for inference.

        Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image_path)

    # Resize
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((5000, 256))
    else:
        pil_image.thumbnail((256, 5000))

    # Crop
    left_margin = (pil_image.width-224)/2
    bottom_margin = (pil_image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224

    pil_image = pil_image.crop((left_margin, bottom_margin,
                                right_margin, top_margin))

    # Normalize
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # PyTorch expects the color channel to be the first dimension but it's the
    # third dimension in the PIL image and Numpy array
    # Color channel needs to be first; retain the order of the other two
    # dimensions.
    np_image = np_image.transpose((2, 0, 1))

    return np_image


def predict(image_path: str, model: models, topk: int = 4) -> tuple:
    ''' Predict the class of an image using trained deep learning model.'''

    image = process_image(image_path)

    # Convert image to PyTorch tensor first

    #CUDA
    # image = torch.from_numpy(image).type(torch.cuda.FloatTensor)

    #CPU
    image = torch.from_numpy(image).type(torch.FloatTensor)

    # Returns a new tensor with a dimension of size one inserted
    # at the specified position.
    image = image.unsqueeze(0)
    output = model.forward(image)
    probabilities = torch.exp(output)

    # Probabilities and the indices of those probabilities
    # corresponding to the classes
    top_prob, top_ind = probabilities.topk(topk)

    # Convert to lists
    top_prob = top_prob.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_ind = top_ind.detach().type(torch.FloatTensor).numpy().tolist()[0]

    # Convert topk_indices to the actual class labels using class_to_idx
    # Invert the dictionary so you get a mapping from index to class.

    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_classes = [idx_to_class[index] for index in top_ind]

    return top_prob, top_classes

def check_image(image_path: str, inf_model: models) -> tuple:
    probs, classes = predict(image_path, inf_model)
    print(classes)
    print(probs)
    print('\n')
    return classes[0], float(probs[0])
