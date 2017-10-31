import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class Generator(nn.Module):
    """docstring for Generator"""
    
    def __init__(self,least_dimension,max_dimension,dimension_step_ratio,smoothing_steps,learning_rate=0.1):
        super(Generator, self).__init__()
        self.least_dimension = least_dimension
        self.dimension_step_ratio = dimension_step_ratio
        self.max_dimension = max_dimension
        self.smoothing_steps= smoothing_steps
        self.init_layers(least_dimension,max_dimension,dimension_step_ratio)
        self.model=self.make_model(self.init_layers,smoothing_steps)
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion=#define criteria for loss
    
    def make_model(layers_list,smoothing_steps):
        model=None
        if smoothing_steps:
            pass
        else:
            model=nn.Sequential(*layers_list)
        return model

    def init_layers(least_dimension,max_dimension,dimension_step_ratio):
        l_of_layer=[]
        i_dim=2
        while True:
            if calculate_deconv_output_dimension(input_dim)<= max_dimension:
                
                k_size=calculate_deconv_kernel_size(input_dim,dimension_step_ratio)
                l_of_layer.append(deconv(c_in,c_out,k_size))
                input_dim=input_dim*dimension_step_ratio
            else:
                break
        return l_of_layer


    def forward():
        pass
        
class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self,least_dimension,max_dimension,dimension_step_ratio,smoothing_steps,learning_rate=0.1):
        super(Discriminator, self).__init__()
        self.least_dimension = least_dimension
        self.dimension_step_ratio = dimension_step_ratio
        self.max_dimension = max_dimension
        self.smoothing_steps= smoothing_steps

    def make_model():
        pass

    def init_layers(least_dimension,max_dimension,dimension_step_ratio):
        #list of layers
    def forward():
        pass

class PGGAN(object):
    """docstring for PGGAN"""
    def __init__(self, least_dimension,max_dimension,dimension_step_ratio,smoothing_steps):
        super(PGGAN, self).__init__()
        self.least_dimension = least_dimension
        self.dimension_step_ratio = dimension_step_ratio
        self.max_dimension = max_dimension
        self.smoothing_steps= smoothing_steps
        self.Generator=Generator(least_dimension,max_dimension,dimension_step_ratio,smoothing_steps)
        self.Discriminator=Discriminator(least_dimension,max_dimension,dimension_step_ratio,smoothing_steps)



