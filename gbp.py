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
import collections
from collections import Counter
from torch.autograd import Variable
import torch
from utils import load_checkpoint
from datasets import RadarDataset
from models.models import CNN3DNet_Modified
import sys
import random
sys.path.insert(0, '../')


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
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
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


if __name__ == '__main__':
    dataset = 'gestures'
    base_location = '/home/gp/Documents/radar_stuff/radar_data'
    csv_location = '/home/gp/Documents/radar_stuff/radar_split_csv'
    sample_length = 50
    filename_dataset = '/home/gp/Documents/radar_stuff/radar_split_csv/gestures-random-train.csv'
    features = 'range_doppler_log'

    values = dict()
    values['microdoppler'] = {'mean': -18548.79815690202, 'std': 1202.8846406929292,
                              'min': -20102.396484375, 'max': 0.0}
    values['range_doppler_log'] = {'mean': -115.96167423431216, 'std': 47.856299075087996,
                                   'min': -200.0, 'max': -24.600444793701172,
                                   'median': np.load('/home/gp/Documents/radar_stuff/range_doppler_median.npy')}

    with open(os.path.join(csv_location, '%s-labels.csv' % dataset), 'r') as f:
        reader = csv.reader(f, delimiter=';')
        labels = [v for v, in reader]
    print(labels)
    sample_sizes = {
        'microdoppler': {0: (sample_length, 256), 1: (sample_length, 253)},
        'range_doppler_log': {0: (sample_length, 80, 128), 1: (sample_length, 80, 126)},
    }

    # Dataset for forward pass
    transform = functools.partial(transform,
                                  sample_length=sample_length,
                                  features=features,
                                  scaling='standard',
                                  preprocessing=True,
                                  values=values[features])
    dataset2 = RadarDataset(filename_dataset, labels,
                            transform=transform,
                            feature=features,
                            select='center',
                            sample_length=sample_length,
                            base_location=base_location)
    # Model
    net = CNN3DNet_Modified(num_classes=len(
        labels), sample_size=sample_sizes[features][1])
    # Load model
    load_checkpoint(
        net, None, '/home/gp/Documents/radar_stuff/models/gestures_random_nopool_cpu.pt')
    net.cpu()
    net.eval()

    VIS_TARGET_FOLDER = '/home/gp/Documents/results_GBP_random_train_final/'
    if not os.path.exists('/home/gp/Documents/results_GBP_random_train_final/'):
        os.makedirs('/home/gp/Documents/results_GBP_random_train_final/')

    # frame for padding
    pad_sample, pad_im_label = dataset2[254]
    padding = pad_sample[0][49]
    # sample_viz, im_label = dataset2[0]
    # print(net(sample_viz))
    # print('pad_im_label:', im_label)
    # print('pad_sample:', pad_sample)
    print('length of dataset', len(dataset2))

    master_logit_diff_list = [0] * 50
    master_magnitude_list = [0] * 50
    master_stddev_logit_list = []
    std_dev = [0] * 50
    gbp_magnitude_frame_replace = []
    avg_logit_diff_list = []
    upper_confidence_list = [0] * 50
    lower_confidence_list = [0] * 50

    # Guided backprop
    GBP = GuidedBackprop(net)
    sample_amount = len(dataset2)
    for i in range(0, sample_amount):
        print('Iter:', i)
        sample_id = i
        # print('sample_id:', sample_id)
        # Get image for forward pass
        gbp_sample, label1 = dataset2[sample_id]
        # making copies for modification
        gbp_sample_copy = copy.deepcopy(gbp_sample)
        # Forward pass
        out = net(gbp_sample)
        print('out', out)
        # current prediction
        org_pred_logit = out[0][label1].item()
        # print('original_logit', out[0][label1].item())
        # Get gradients
        var_sample_1 = Variable(gbp_sample_copy, requires_grad=True)
        guided_grads = GBP.generate_gradients(var_sample_1, label1)
        # print('Vanilla_grads_np:', vanilla_grads.shape)
        split_grads_np1 = guided_grads[0]
        split_grads_np1 = np.clip(split_grads_np1, a_min=0, a_max=100000000)
        # print('split_grads_np:', split_grads_np.shape)
        gbp_magnitude_list = []
        for j in range(0, 50):
            # Keep this, see if this makes a difference
            gbp_magnitude_list.append(np.linalg.norm(split_grads_np1[j]))
            # vbp_magnitude_list.append(np.sum(split_grads_np1[j]))
        # print('vbp_magnitude_list', vbp_magnitude_list)
        logit_difference_list = []
        sorted_logit_list = []
        for single_frame_replace_id in range(0, 50):
            gbp_sample_copy = copy.deepcopy(gbp_sample)
            gbp_sample_copy[0][single_frame_replace_id] = padding
            # Forwad pas with changed frame
            out = net(gbp_sample_copy)
            # modified_pred_logit = out[0][label1].item()
            # print('modified_out', out)
            # print('modified_logit', out[0][label1].item())
            logit_difference = org_pred_logit - out[0][label1].item()
            # print('single_frame_replace_id:', 'logit_difference:', single_frame_replace_id, logit_difference)
            logit_difference_list.append(logit_difference)

        print()
        # print('logit_difference_list', logit_difference_list)

        print()
        # Sort logit changes according to grad magnitude, the largest magnitude is the last
        # so the first element is the smallest magnitude ( ascending order )
        sorted_logit_list = [x for _, x in sorted(zip(gbp_magnitude_list, logit_difference_list))]
        # print()
        # print('sorted_logit_list', sorted_logit_list)
        master_stddev_logit_list.extend(sorted_logit_list)

        master_logit_diff_list = [x + y for x, y in zip(sorted_logit_list, master_logit_diff_list)]

    # print()
    print('avg_logit_diff_list_len', len(avg_logit_diff_list))

    master_stddev_logit_as_np = np.split(np.array(master_stddev_logit_list), sample_amount)
    # print('master_stddev_logit_as_np', master_stddev_logit_as_np)
    std_dev = np.std(np.vstack(master_stddev_logit_as_np), axis=0)
    # print()
    print('standard_deviation_len', std_dev)
    # avg_magnitude_list to get the change_in_logit-vs-Frame plot (frames in descending order of magnitude)
    # print('master_logit_diff_list', master_logit_diff_list)

    for loc in range(50):
        avg_logit_diff_list = [x / sample_amount for x in master_logit_diff_list]
        master_magnitude_list[loc] = master_magnitude_list[loc] + \
            gbp_magnitude_list[loc]

    print('avg_logit_diff_list', avg_logit_diff_list)
    print()
    # calcuating the 95 percent confidennce interval

    for loc in range(50):
        upper_confidence_list[loc] = avg_logit_diff_list[loc] + \
            std_dev[loc] * (1.96 / np.sqrt(sample_amount))
        lower_confidence_list[loc] = avg_logit_diff_list[loc] - \
            (1.96 / np.sqrt(sample_amount)) * std_dev[loc]
    # print()
    # print('standard_deviation', standard_deviation)
    # avg_magnitude_list to get the change_in_logit-vs-Frame plot (frames in descending order of magnitude)

    print('upper_confidence_list', upper_confidence_list)
    print()
    print('lower_confidence_list', lower_confidence_list)

    plt.figure(51)
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.plot(avg_logit_diff_list, '-b', label='avg_logit_diff')
    plt.plot(upper_confidence_list, '-r', label='UCL')
    plt.plot(lower_confidence_list, '-r', label='LCL')
    plt.legend(loc='upper left')
    plt.savefig(VIS_TARGET_FOLDER + 'avg_logit_diff_list_all_train_samples_' + 'GBP_' +
                ".png", format="PNG")

    plt.figure(52)
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.plot(sorted(master_magnitude_list, reverse=True), '-b', label='master_magnitude')
    plt.legend(loc='upper right')
    plt.savefig(VIS_TARGET_FOLDER + 'master_magnitude_all_train_samples_' + 'GBP_' +
                ".png",  format="PNG")

    plt.figure(53)
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.plot(master_logit_diff_list, '-b', label='master_logit_diff')
    plt.legend(loc='upper left')
    plt.savefig(VIS_TARGET_FOLDER + 'master_logit_diff_list_all_train_samples_' + 'GBP_' +
                ".png", format="PNG")

print('guided backprop completed')
