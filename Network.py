import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import numpy as np
from torch.autograd import Variable
class Generator(nn.Module):
    """docstring for Generator"""
    
    def __init__(self,least_size,max_size,size_step_ratio,learning_rate=0.1):
        super(Generator, self).__init__()
        self.least_size = least_size
        self.max_size=max_size
        self.size_step_ratio = size_step_ratio
        self.input_dim=least_size
        self.curr_max_size=self.input_dim*self.size_step_ratio
        self.output_dim=None
        self.c_in=1
        self.c_out=self.c_in*2
        self.layer_list=self.init_layers(self.least_size,self.curr_max_size,self.size_step_ratio)
        self.model=self.make_model(self.layer_list)
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.smoothing_factor=0.2
        self.will_be_next_layers=None

    def make_model(self,layers_list):
        model=nn.Sequential(*layers_list)
        return model

    def init_layers(self,least_size,curr_max_size,size_step_ratio):
        l_of_layer=[]
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

    def add_layer(self,with_smoothing=False):
        if not with_smoothing:
            if self.output_dim<=self.max_size:
                k_size=calculate_deconv_kernel_size(self.input_dim,self.size_step_ratio) 
                self.layer_list.append(deconv(self.c_in,self.c_out,k_size))
                self.input_dim=self.input_dim*self.size_step_ratio
                self.output_dim=self.input_dim*self.size_step_ratio
                self.c_in=self.c_out
                self.c_out=self.c_in*2
            else:
                print ("MAX SIZE REACHED")
            self.model=self.make_model(self.layer_list)
        else:
            if self.will_be_next_layers==None:
                print ("Smoothing branch not present, kindly call add_smoothing_branch")
                return
            self.input_dim=self.input_dim*self.size_step_ratio
            self.output_dim=self.input_dim*self.size_step_ratio            
            self.c_in=self.c_out
            self.c_out=self.c_in*2
            self.model=self.make_model(self.will_be_next_layers)
            self.layer_list=self.will_be_next_layers
            self.will_be_next_layers=None

    def add_smoothing_branch(self):
        if self.output_dim<=self.max_size:
            k_size=calculate_deconv_kernel_size(self.input_dim,self.size_step_ratio)
            self.will_be_next_layers=self.layer_list+[deconv(self.c_in,self.c_out,k_size)]
        else:
            print ("MAX SIZE REACHED")
            

    def forward(self,input,with_smoothing=False):
        if with_smoothing:
            if self.will_be_next_layers==None:
                print ("call add_smoothing_branch and run for few epochs and then call add_layer with Smoothing")
            A=F.upsample((1-self.smoothing_factor)*self.model(input),scale_factor=self.size_step_ratio)
            B=self.smoothing_factor*self.make_model(self.will_be_next_layers)(input)
            A=sum(A,[0,1],keepdim=True)
            B=sum(B,[0,1],keepdim=True)
            return (1-self.smoothing_factor)*A + self.smoothing_factor*B 
        else:
            A=sum(self.model(input),[0,1],keepdim=True)
            return A
        
class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self,least_size,max_size,size_step_ratio,learning_rate=0.1):
        super(Discriminator, self).__init__()
        self.least_size = least_size
        self.size_step_ratio = size_step_ratio
        self.max_size = max_size
        self.input_dim=4
        self.curr_least_size=int(self.input_dim*self.size_step_ratio)
        self.output_dim=int(self.input_dim*self.size_step_ratio)
        self.least_size=self.least_size
        self.c_in=2
        self.c_out=1
        self.layer_list=self.init_layers()
        self.model=self.make_model(self.layer_list)
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.will_be_next_layers=None
        self.smoothing_factor=0.2

        
    def make_model(self,layers_list):
        model=nn.Sequential(*layers_list)
        return model

    def init_layers(self):
        l_of_layer=[]
        k_size=calculate_conv_kernel_size(self.input_dim,self.size_step_ratio)
        l_of_layer.insert(0,conv(self.c_in,self.c_out,k_size))
        self.output_dim=self.input_dim
        self.input_dim=int(self.output_dim*(1/self.size_step_ratio))
        self.c_out=self.c_in
        self.c_in=self.c_out*2
        return l_of_layer

    def add_layer(self,with_smoothing=False):
        if not with_smoothing:
            if self.output_dim>=self.least_size:
                k_size=calculate_conv_kernel_size(self.input_dim,self.size_step_ratio) 
                self.layer_list.insert(0,conv(self.c_in,self.c_out,k_size))
                self.input_dim=self.input_dim*(1/self.size_step_ratio)
                self.output_dim=self.input_dim*(1/self.size_step_ratio)
                self.c_in=self.c_out
                self.c_out=self.c_in//2
            else:
                print ("Least SIZE REACHED")
            self.model=self.make_model(self.layer_list)
        else:
            if self.will_be_next_layers==None:
                print ("Smoothing branch not present, kindly call add_smoothing_branch")
                return
            self.model=self.make_model(self.will_be_next_layers)
            self.layer_list=self.will_be_next_layers
            self.will_be_next_layers=None 
            self.output_dim=self.input_dim
            self.input_dim=int(self.output_dim*(1/self.size_step_ratio))



    def add_smoothing_branch(self):
        if self.input_dim<=self.max_size:
            k_size=calculate_conv_kernel_size(self.input_dim,self.size_step_ratio)
            self.will_be_next_layers=[conv(self.c_in,self.c_out,k_size)]+self.layer_list
            self.c_out=self.c_in
            self.c_in=self.c_out*2
        else:
            print ("MAX SIZE REACHED")


    def forward(self,input,with_smoothing=False):
        if with_smoothing:
            if self.will_be_next_layers==None:
                print ("call add_smoothing_branch and run for few epochs and then call add_layer with Smoothing")
            input1=input.data.numpy()
            input_to_supply=np.tile(input1,(1,self.c_out,1,1))
            # k_size=calculate_avgpool_kernel_size(self.input_dim,self.size_step_ratio)
            k_size=2
            avg_pool=nn.AvgPool2d(2,stride=0)
            A=avg_pool(input)
            A1=A.data.numpy()
            A_to_supply=np.tile(A1,(1,int(self.c_out/2),1,1))            
            A=(1-self.smoothing_factor)*self.model(Variable(torch.Tensor(A_to_supply)))
            B=self.smoothing_factor*self.make_model(self.will_be_next_layers)(Variable(torch.Tensor(input_to_supply)))
            print (A.size())
            A=sum(A,[0,1],keepdim=True)
            B=sum(B,[0,1],keepdim=True)
            return (1-self.smoothing_factor)*A + self.smoothing_factor*B 
        else:
            input1=input.data.numpy()
            input_to_supply=np.tile(input1,(1,self.c_out,1,1))
            print (input_to_supply.shape)
            A=self.model(Variable(torch.Tensor(input_to_supply)))
            return A

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



