import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class Generator(nn.Module):
    """docstring for Generator"""
    
    def __init__(self,least_size,curr_max_size,size_step_ratio,smoothing_steps=None,learning_rate=0.1):
        super(Generator, self).__init__()
        self.least_size = least_size
        self.size_step_ratio = size_step_ratio
        self.curr_max_size = curr_max_size
        self.smoothing_steps= smoothing_steps
        self.input_dim=least_size
        self.output_dim=None
        self.c_in=1
        self.c_out=self.c_in*2
        self.layer_list=self.init_layers(least_size,curr_max_size,size_step_ratio)
        self.model=self.make_model(self.layer_list,smoothing_steps)
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def make_model(self,layers_list,smoothing_steps=None):
        model=None
        if smoothing_steps:
            pass
        else:
            model=nn.Sequential(*layers_list)
        return model

    def init_layers(self,least_size,curr_max_size,size_step_ratio):
        l_of_layer=[]
        self.c_in=1
        self.c_out=2
        self.input_dim=least_size
        self.output_dim=self.input_dim*size_step_ratio
        while True:
            if self.output_dim<= curr_max_size:
                k_size=calculate_deconv_kernel_size(self.input_dim,size_step_ratio)
                l_of_layer.append(deconv(self.c_in,self.c_out,k_size))
                self.input_dim=self.input_dim*size_step_ratio
                self.output_dim=self.input_dim*size_step_ratio
                self.c_in=self.c_out
                self.c_out=self.c_in*2
            else:
                break
        return l_of_layer

    def add_layer(self,layer_list,max_size,size_step_ratio):
        self.output_dim=self.input_dim*size_step_ratio
        if self.output_dim<=max_size:
            k_size=calculate_deconv_kernel_size(self.input_dim,size_step_ratio) 
            self.layer_list.append(deconv(self.c_in,self.c_out,k_size))
        else:
            print ("MAX SIZE REACHED")
        self.model=self.make_model(self.layer_list)

    def add_smoothing_branch(self):
        pass

    def forward(self,input):
        return self.model(input)
        
class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self,least_size,max_size,size_step_ratio,input_shape,smoothing_steps=None,learning_rate=0.1):
        super(Discriminator, self).__init__()
        self.least_size = least_size
        self.size_step_ratio = size_step_ratio
        self.max_size = max_size
        self.smoothing_steps= smoothing_steps
        self.input_shape=input_shape
        self.layer_list=self.init_layers(least_size,max_size,size_step_ratio,input_shape)
        self.model=self.make_model(self.layer_list,smoothing_steps)
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        
    def make_model(self,layers_list,smoothing_steps=None):
        model=None
        if smoothing_steps:
            pass
        else:
            model=nn.Sequential(*layers_list)
        return model

    def init_layers(self,least_size,max_size,size_step_ratio,input_shape):
        l_of_layer=[]
        c_in=input_shape[1]
        c_out=c_in//2
        input_dim=max_size
        output_dim=max_size*size_step_ratio
        while True:
            if output_dim>= least_size:
                k_size=calculate_conv_kernel_size(input_dim,size_step_ratio)
                l_of_layer.append(conv(c_in,c_out,k_size))
                input_dim=input_dim*size_step_ratio
                output_dim=input_dim*size_step_ratio
                c_in=c_out
                c_out=c_in//2
            else:
                break
        return l_of_layer

    def forward(self,input):
        return self.model(input)

class PGGAN(object):
    """docstring for PGGAN"""
    def __init__(self, least_size,max_size,size_step_ratio):
        super(PGGAN, self).__init__()
        self.least_size = least_size
        self.size_step_ratio = size_step_ratio
        self.max_size = max_size
        self.G=Generator(least_size,max_size,size_step_ratio)
        self.D=Discriminator(least_size,max_size,size_step_ratio)

    def train(self,num_of_epochs,batch_size):
        pass



