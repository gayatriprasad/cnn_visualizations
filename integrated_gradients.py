from __future__ import print_function
import matplotlib.cm as mpl_color_map
import copy
import pdb
from transforms.radar_transforms import microdoppler_transform as transform
import matplotlib.pylab as plt
from torch.nn import ReLU
import numpy as np
from PIL import Image
import functools
import csv
import os
from torch.autograd import Variable
import torch
from utils import load_checkpoint
from datasets import RadarDataset
from models.models import CNN3DNet_Modified
import sys
import random
sys.path.insert(0, '../')


class IntegratedGradients():
    """
        Produces gradients generated with integrated gradients from the image
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward
        print('input_image.shape_gg', input_image.shape)
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr[0]

    def generate_integrated_gradients(self, input_image, target_class, steps):
        # consider baseline as a black image
        baseline = torch.zeros(input_image.size())
        # print('avg_grads.shape', avg_grads.shape)
        average_gradients = np.zeros(input_image.size())
        integrated_gradients = np.zeros(input_image.size())
        print('input_image.shape_ig', input_image.shape)
        for i in range(1, steps):
            scaled_inputs = baseline + (float(i) / steps) * (input_image - baseline)
            single_gradient = self.generate_gradients(scaled_inputs, target_class)
            average_gradients += single_gradient
        average_gradients = np.average(average_gradients, axis=0)
        integrated_gradients = (input_image - baseline).data.numpy() * average_gradients
        # print('after', integrated_gradients.shape)
        return integrated_gradients[0]


if __name__ == '__main__':
    # Get params
    target_example = 1  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    # Vanilla backprop
    IG = IntegratedGradients(pretrained_model)
    # Generate gradients
    integrated_grads = IG.generate_gradients(prep_img, target_class)
    # Save colored gradients
    save_gradient_images(integrated_grads, file_name_to_export + '_integrated_gradients_color')
    # Convert to grayscale
    grayscale_integrated_grads = convert_to_grayscale(integrated_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_integrated_grads, file_name_to_export +
                         '_integrated_gradients_gray')
    print('Integrated Gradients completed')
