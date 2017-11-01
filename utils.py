import torch
import torch.nn as nn
import torch.nn.functional as F

def deconv(c_in, c_out, k_size, stride=1, pad=0, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=1, pad=0, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def calculate_deconv_output_dimension(input_dim,k_size,stride=1,pad=0):
    return int((input_dim-1)*stride+k_size-2*pad)

def calculate_conv_output_dimension(input_dim,k_size,stride=1,pad=0):
    return int((input_dim - k_size +2*pad)//stride+1)

def calculate_conv_kernel_size(input_dim,dimension_step_ratio,stride=1,pad=0):
    return int(input_dim+2*pad-(input_dim*dimension_step_ratio-1)*stride)

def calculate_deconv_kernel_size(input_dim,dimension_step_ratio,stride=1,pad=0):
    return int(2*pad+(input_dim*dimension_step_ratio)-stride*(input_dim-1))

def calculate_avgpool_kernel_size(input_dim,dimension_step_ratio,stride=1,pad=0):
    return int(input_dim+2*pad-(input_dim*dimension_step_ratio-1)*stride)
    
def sum(input, axes, keepdim=False):
    # probably some check for uniqueness of axes
    if keepdim:
        for ax in axes:
            input = input.sum(ax, keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(ax)
    return input

