import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

x=np.random.rand(2,2)
x=torch.Tensor(x)
x.unsqueeze_(0)
x.unsqueeze_(0)
print (x.size())

y=Variable(x)
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
    return (input_dim-1)*stride+k_size-2*pad

def calculate_conv_output_dimension(input_dim,k_size,stride=1,pad=0):
    print (int((input_dim - k_size +2*pad)//stride+1))
    return int((input_dim - k_size +2*pad)//stride+1)

def calculate_conv_kernel_size(input_dim,dimension_step_ratio,stride=1,pad=0):
    print (int(input_dim+2*pad-(input_dim*dimension_step_ratio-1)*stride))
    return int(input_dim+2*pad-(input_dim*dimension_step_ratio-1)*stride)

def calculate_deconv_kernel_size(input_dim,dimension_step_ratio,stride=1,pad=0):
    return 2*pad+(input_dim*dimension_step_ratio)-stride*(input_dim-1)

from Network import Generator,Discriminator
g=Generator(2,8,2)
d=Discriminator(2,8,0.5,(1,4,8,8))

z=g(y)
print (z.size())
logits=d(z)
print (logits.size())