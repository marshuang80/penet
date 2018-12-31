"""
Created on Thu Oct 26 11:06:51 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import cv2
import torch
import os
from torch.nn import ReLU
import sys, json, time, cv2, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import numpy as np
import pandas as pd
import sklearn.preprocessing
import matplotlib.pyplot as plt
from utils import get_model_and_loader, transform_data
#from misc_functions import get_params, save_class_activation_on_image


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if module_pos == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.fc(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.cuda.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.features.zero_grad()
        self.model.fc.zero_grad()
        # Backward pass with specified target

        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients

        total_cams = []
        for sliceNo in range(input_image.size()[2]):
            grad_slice = guided_gradients[:, sliceNo, :, :].squeeze()
            target_slice = target[:, sliceNo, :, :].squeeze()
            
            weights = np.mean(grad_slice, axis=(1, 2))
            cam = np.ones(target_slice.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * target_slice[i, :, :]
            cam = cv2.resize(cam, (input_image.size()[3], input_image.size()[4])) 
            cam = np.maximum(cam, 0)
            print (np.min(cam), np.max(cam))
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
            total_cams.append(cam)
        return np.array(total_cams)


def save_class_activation_on_image(inputs, cams, outputdir, parent_filename):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # # Grayscale activation map
    # path_to_file = os.path.join(outputdir, file_name+'_Cam_Grayscale.jpg')
    # cv2.imwrite(path_to_file, activation_map)
    # # Heatmap of activation map
    # activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)
    # path_to_file = os.path.join(outputdir, file_name+'_Cam_Heatmap.jpg')
    # cv2.imwrite(path_to_file, activation_heatmap)

    num_images = inputs.shape[0]

    for sliceNo in range(num_images):
        # Heatmap on picture
        org_img = inputs[sliceNo]
        activation_map = cams[sliceNo]
        
        activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)
        org_img = cv2.resize(org_img, (336, 336))
        img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
        img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
        path_to_file = os.path.join(outputdir, file_name + f'{sliceNo}_GradCam.jpg')
        cv2.imwrite(path_to_file, np.uint8(255 * img_with_heatmap))


def get_grad(model, loader, label_names, thresholds, outputdir, save_orig=True):
    grad_cam = GradCam(model, target_layer='conv1') # This is for C3D 

    for i, batch in enumerate(loader):
        # This runs on each datapoint at a point
        inputs, labels = transform_data(batch[:2], True, False)

        labels_npy = labels.data.cpu().numpy().squeeze()
        positive_inds = labels_npy == 1 # keep vectorized for multilabel models
        name = label_names[0] #Only PE
        # Only run CAMS on positive examples.
        if not positive_inds:
            continue

        # Generate cam mask
        inputs = inputs.permute(0, 2, 1, 3, 4)
        cams = grad_cam.generate_cam(inputs, 0)
        parent_filename = f'{name}_{i}_'
        save_class_activation_on_image(inputs, cams, outputdir, parent_filename)

