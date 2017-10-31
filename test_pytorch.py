import torch
import torch.nn as nn
import numpy as np
x=np.random.rand(2,2)
x=torch.Tensor(x)
x.unsqueeze_(0)
x.unsqueeze_(0)
from torch.autograd import Variable
print (x.size())

import torch
import torch.nn as nn
import torch.nn.functional as F

#def deconv(c_in, c_out, k_size, stride=1, pad=0, bn=True):
#    """Custom deconvolutional layer for simplicity."""
#    layers = []
#    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
#    if bn:
#        layers.append(nn.BatchNorm2d(c_out))
#    layers.append(nn.ReLU())
#    return nn.Sequential(*layers)
#
#
#
y=Variable(x)
#
#deconv1=deconv(1,4,3)
#deconv2=deconv(4,16,5)
#z=deconv1(y)
#print(z.size())
#z1=deconv2(z)
#print (z1.size())
#
#
#def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
#    """Custom convolutional layer for simplicity."""
#    layers = []
#    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
#    if bn:
#        layers.append(nn.BatchNorm2d(c_out))
#    layers.append(nn.ReLU())
#    return nn.Sequential(*layers)

#conv1=conv(16,4,5)
#conv2=conv(4,1,3)
#a1=conv1(z1)
#a2=conv2(a1)
#print (a1.size())
#print (a2.size())
def deconv(c_in, c_out, k_size, stride=1, pad=0, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
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
    print ((input_dim - k_size +2*pad)//stride+1)
    return int((input_dim - k_size +2*pad)//stride+1)

def calculate_conv_kernel_size(input_dim,dimension_step_ratio,stride=1,pad=0):
    print (input_dim+2*pad-(input_dim*dimension_step_ratio-1)*stride)
    return int(input_dim+2*pad-(input_dim*dimension_step_ratio-1)*stride)

def calculate_deconv_kernel_size(input_dim,dimension_step_ratio,stride=1,pad=0):
    return 2*pad+(input_dim*dimension_step_ratio)-stride*(input_dim-1)

def init_layers(least_dimension,max_dimension,dimension_step_ratio,input_dim):
    l_of_layer=[]
    c_in=1
    c_out=2
    output_dim=input_dim*dimension_step_ratio
    while True:
        if output_dim<= max_dimension:
            k_size=calculate_deconv_kernel_size(input_dim,dimension_step_ratio)
            l_of_layer.append(deconv(c_in,c_out,k_size))
            input_dim=input_dim*dimension_step_ratio
            output_dim=input_dim*dimension_step_ratio
            c_in=c_out
            c_out=c_in*2
        else:
            break
    return l_of_layer

model=nn.Sequential(*init_layers(1,8,2,2))
print (model)
out=model(y)
print (out.size())
def d_init_layers(least_dimension,max_dimension,dimension_step_ratio,input_dim):
    l_of_layer=[]
    c_in=1
    c_out=2
    output_dim=input_dim*dimension_step_ratio
    while True:
        if output_dim>= least_dimension:
            k_size=calculate_conv_kernel_size(input_dim,dimension_step_ratio)
            l_of_layer.append(conv(c_in,c_out,k_size))
            input_dim=input_dim*dimension_step_ratio
            output_dim=input_dim*dimension_step_ratio
            c_in=c_out
            c_out=c_in*2
        else:
            break
    return l_of_layer

d_model=nn.Sequential(*d_init_layers(2,8,0.5,8))
in_=d_model(out)
print (in_.size())
