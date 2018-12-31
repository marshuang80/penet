import os
import torch
from torch.nn import ReLU
import sys, json, time, cv2, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import numpy as np
import pandas as pd
import sklearn.preprocessing
import matplotlib.pyplot as plt

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.cuda.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224) : is this the batch size? maybe, maybe not. 
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        return gradients_as_arr


def get_gbp(model, loader, label_names, thresholds, outputdir, save_orig=False):
    # Guided backprop
    GBP = GuidedBackprop(model)

    for i, batch in enumerate(loader):
        # This runs on each datapoint at a point
        inputs, labels = transform_data(batch[:2], True, False)

        labels_npy = labels.data.cpu().numpy().squeeze()
        positive_inds = labels_npy == 1 # keep vectorized for multilabel models
        positive_names = np.array(label_names)[positive_inds]
        name = label_names[0] #Only PE
        # Only run CAMS on positive examples.
        if not positive_names:
            continue

        # prediction, cams = get_prediction_and_cams(model, inputs)

        # Get gradients: The final shape is (num_img, num_ch, h, w) 
        guided_grads = GBP.generate_gradients(inputs, 0)
        guided_grads = np.transpose(guided_grads, (1, 0, 2, 3))
        # print ('Guided Backprop on ', inputs.data.cpu().numpy().squeeze().shape, ' are ', guided_grads.shape)

        num_images = guided_grads.shape[0]

        for sliceNo in range(num_images):
            grad = guided_grads[sliceNo].squeeze()
            # Save colored gradients
            file = f'{name}_{i}_{sliceNo}'
            save_gradient_images(grad, outputdir, file + '_GBPcolor.jpg')
            # Convert to grayscale
            grayscale_grad = convert_to_grayscale(grad)
            # Save grayscale gradients
            save_gradient_images(grayscale_grad, outputdir, file + '_GBPgray.jpg')
            # Positive and negative saliency maps
            pos_sal, neg_sal = get_positive_negative_saliency(grad)
            save_gradient_images(pos_sal, outputdir, file + '_pos_sal.jpg')
            save_gradient_images(neg_sal, outputdir, file + '_neg_sal.jpg')


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize
    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


def convert_to_grayscale(cv2im):
    """
        Converts 3d image to grayscale
    Args:
        cv2im (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(cv2im), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, outputdir, file_name):
    """
        Exports the original gradient image
    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    gradient = np.uint8(gradient * 255).transpose(1, 2, 0)
    path_to_file = os.path.join(outputdir, file_name)
    # Convert RBG to GBR
    gradient = gradient[..., ::-1]
    cv2.imwrite(path_to_file, gradient)