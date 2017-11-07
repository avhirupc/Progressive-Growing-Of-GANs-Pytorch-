import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import numpy as np
from torch.autograd import Variable
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

class Generator(nn.Module):
    """Generator:
        Input : Noise of dimension least_size*least_size
        Ouput : Single channel B/W Image of current output dimension

        Parameters:
            least_size : minimum size you want to start with 
            max_size : maximum size of output after training
            size_step_ratio : ratio with which you want to increase output after each layer"""

    
    def __init__(self,least_size,max_size,size_step_ratio,learning_rate=0.1,batch_size=100):
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
        self.batch_size=batch_size
        self.will_be_next_layers=None
        self.init_data()
        self.learning_rate=learning_rate

    def init_data(self):
        """Initialises data_loader"""
        train_dataset=Noise(60000,self.least_size)
        self.data_loader=torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=self.batch_size, 
                                           shuffle=True)

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
                self.input_dim=self.output_dim
                self.output_dim=self.input_dim*size_step_ratio
                self.c_in=self.c_out
                self.c_out=self.c_in*2
            else:
                break
        return l_of_layer

    def add_layer(self,with_smoothing=False):
        """Adds layer to Generator"""
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
            self.optimizer=torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def add_smoothing_branch(self):
        """Adds smooothing branch with over time turns to new model"""
        if self.output_dim<=self.max_size:
            k_size=calculate_deconv_kernel_size(self.input_dim,self.size_step_ratio)
            self.will_be_next_layers=self.layer_list+[deconv(self.c_in,self.c_out,k_size)]
            self.optimizer=torch.optim.Adam(self.make_model(self.will_be_next_layers).parameters(), lr=self.learning_rate)            
        else:
            print ("MAX SIZE REACHED")
            

    def forward(self,input,with_smoothing=False):
        if with_smoothing:
            if self.will_be_next_layers==None:
                print ("call add_smoothing_branch and run for few epochs and then call add_layer with Smoothing")
                return
            A=F.upsample((1-self.smoothing_factor)*self.model(input),scale_factor=self.size_step_ratio)
            B=self.smoothing_factor*self.make_model(self.will_be_next_layers)(input)
            # A=sum(A,[0,1],keepdim=True)
            A=sum(A,[1],keepdim=True)
            # B=sum(B,[0,1],keepdim=True)
            B=sum(B,[1],keepdim=True)
            C=(1-self.smoothing_factor)*A + self.smoothing_factor*B 
            return C
        else:
            A=sum(self.model(input),[1],keepdim=True)
            return A
        
class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self,least_size,max_size,size_step_ratio,learning_rate=0.1,batch_size=100):
        super(Discriminator, self).__init__()
        self.least_size = least_size
        self.size_step_ratio = size_step_ratio
        self.max_size = max_size
        self.input_dim=int(self.least_size*(1/size_step_ratio))
        self.curr_least_size=int(self.input_dim*self.size_step_ratio)
        self.output_dim=int(self.input_dim*self.size_step_ratio)
        self.least_size=self.least_size
        self.batch_size=batch_size
        self.init_data()
        self.c_in=2
        self.c_out=1
        self.layer_list=self.init_layers()
        self.model=self.make_model(self.layer_list)
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.will_be_next_layers=None
        self.smoothing_factor=0.2
        self.learning_rate=learning_rate

    def init_data(self):
        """Initialises data_loader"""
        t=transforms.Compose([transforms.Scale(self.input_dim),transforms.ToTensor()])
        train_dataset = dsets.MNIST(root='./data/',
                            train=True, 
                            transform=t,
                            download=True)
        self.data_loader=torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=self.batch_size, 
                                           shuffle=True)

    def resize_data(self):
        """Changes data_loader"""
        t=transforms.Compose([transforms.Scale(self.output_dim),transforms.ToTensor()])
        train_dataset = dsets.MNIST(root='./data/',
                            train=True, 
                            transform=t,
                            download=True)
        self.data_loader=torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=self.batch_size, 
                                           shuffle=True)        

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
            self.optimizer=torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)




    def add_smoothing_branch(self):
        if self.input_dim<=self.max_size:
            k_size=calculate_conv_kernel_size(self.input_dim,self.size_step_ratio)
            self.will_be_next_layers=[conv(self.c_in,self.c_out,k_size)]+self.layer_list
            self.c_out=self.c_in
            self.c_in=self.c_out*2
            self.output_dim=self.input_dim
            self.input_dim=int(self.output_dim*(1/self.size_step_ratio))
            self.resize_data()
            self.optimizer=torch.optim.Adam(self.make_model(self.will_be_next_layers).parameters(), lr=self.learning_rate)
        else:
            print ("MAX SIZE REACHED")


    def forward(self,input,with_smoothing=False):
        if with_smoothing:
            if self.will_be_next_layers==None:
                print ("call add_smoothing_branch and run for few epochs and then call add_layer with Smoothing")
                return
            input1=input.clone()
            input_to_supply=input1.repeat(*[1,self.c_out,1,1])

            # input_to_supply=np.tile(input1,(1,self.c_out,1,1))
            # k_size=calculate_avgpool_kernel_size(self.input_dim,self.size_step_ratio)
            k_size=2
            avg_pool=nn.AvgPool2d(2,stride=0)
            A=avg_pool(input)

            # A1=A.data.numpy()
            # A_to_supply=np.tile(A1,(1,int(self.c_out/2),1,1))
            A_to_supply=A.repeat(*[1,int(self.c_out/2),1,1])
            A=(1-self.smoothing_factor)*self.model(A_to_supply)
            
            B=self.smoothing_factor*self.make_model(self.will_be_next_layers)(input_to_supply)
            # A=sum(A,[1],keepdim=True)
            # B=sum(B,[1],keepdim=True)
            return A + B 
        else:
            input1=input.clone()
            input_to_supply=input1.repeat(*[1,self.c_out,1,1])
            A=self.model(input_to_supply)
            return A

class PGGAN(object):
    """docstring for PGGAN"""
    def __init__(self, least_size=2,max_size=16,size_step_ratio=2,learning_rate=0.01,batch_size=100):
        super(PGGAN, self).__init__()
        self.least_size = least_size
        self.size_step_ratio = size_step_ratio
        self.max_size = max_size
        self.G=Generator(least_size,max_size,size_step_ratio,learning_rate=learning_rate,batch_size=batch_size)
        self.D=Discriminator(least_size,max_size,1/size_step_ratio,learning_rate=learning_rate,batch_size=batch_size)

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.D.zero_grad()
        self.G.zero_grad()

    def train(self,num_of_epochs=100):
        smoothing_on=False
        for epoch in range(num_of_epochs):
            avg_d_loss=0
            avg_g_loss=0

            for batch_no,(G_data,D_data) in enumerate(zip(self.G.data_loader,self.D.data_loader)):
                self.reset_grad()
                
                G_data=Variable(G_data)        
                D_data=Variable(D_data[0])
                #resizing d_data to fit currently
                # calculate _loss
                if smoothing_on:
                    outputs=self.D(D_data,with_smoothing=True)
                    real_loss=torch.mean((outputs-1)**2)
                    outputs=self.G(G_data,with_smoothing=True)
                    fake_loss=torch.mean(self.D(outputs,with_smoothing=True)**2)                   
                else:
                    try:
                        outputs=self.D(D_data)
                    except:
                        print (D_data.size())
                        1/0
                    real_loss=torch.mean((outputs-1)**2)
                    outputs=self.G(G_data)
                    fake_loss=torch.mean(self.D(outputs)**2)
                # Backprop + optimize
                d_loss = real_loss + fake_loss
                avg_d_loss+=d_loss.data
                d_loss.backward(retain_graph=True)
                #update weights
                self.D.optimizer.step()
                
                if smoothing_on:
                    outputs=self.G(G_data,with_smoothing=True)
                    fake_loss=torch.mean((self.D(outputs,with_smoothing=True)-1)**2)                   
                else:
                    outputs=self.G(G_data)
                    fake_loss=torch.mean((self.D(outputs)-1)**2)
                # Train G so that D recognizes G(z) as real.
                
                g_loss = fake_loss
                avg_g_loss+=g_loss.data
                g_loss.backward(retain_graph=True)
                #update weights
                self.G.optimizer.step()
                if batch_no%100==0:
                    print ("Batch ",batch_no,"||d_loss",d_loss.data,"||g_loss",g_loss.data)
            print ("epoch",epoch)
            #dump image
            x=np.random.rand(1,1,2,2)
            x=Variable(torch.Tensor(x))
            image=self.G(x)
            image_array=image.data.numpy()
            print (image_array.shape)
            print (type(image_array),image_array.reshape((image_array.shape[2],image_array.shape[3])))
            from PIL import Image

            im = Image.fromarray(image_array.reshape((image_array.shape[2],image_array.shape[3])))
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im.save("your_file.png")
            print ("Avg G Loss",avg_g_loss,"Avg D Loss", avg_d_loss)
            if smoothing_on:
                self.G.smoothing_factor+=0.1
                self.D.smoothing_factor+=0.1
            if epoch%20==0 and epoch!=0:
                self.G.add_layer(with_smoothing=True)
                self.D.add_layer(with_smoothing=True)
                self.G.smoothing_factor=0.2
                self.D.smoothing_factor=0.2
                smoothing_on=False
            elif epoch%10==0 and epoch!=0:
                self.G.add_smoothing_branch()
                self.D.add_smoothing_branch()
                smoothing_on=True







