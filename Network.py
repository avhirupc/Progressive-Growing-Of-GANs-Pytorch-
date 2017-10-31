import torch
import torch.nn as nn

def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """docstring for Generator"""
    def __init__(self,least_dimension,max_dimension,dimension_step_ratio,smoothing_steps,learning_rate=0.1):
        super(Generator, self).__init__()
        self.least_dimension = least_dimension
        self.dimension_step_ratio = dimension_step_ratio
        self.max_dimension = max_dimension
        self.smoothing_steps= smoothing_steps
        self.init_layers(least_dimension,max_dimension,dimension_step_ratio)
        self.model=self.make_model(least_dimension,max_dimension,dimension_step_ratio,smoothing_steps)
        self.optimizer=torch.optim.Adam(D.parameters(), lr=0.0003)
    def make_model():
        pass

    def init_layers(least_dimension,max_dimension,dimension_step_ratio):
        #list of layers
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



