import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

# x=np.random.rand(1,2,2)
# x=torch.Tensor(x)
# x.unsqueeze_(0)
# print (x.size())
# y=Variable(x)
# x=np.random.rand(1,2,2)
# x=torch.Tensor(x)
# x.unsqueeze_(0)
# y=Variable(x)

# from Network import Generator,Discriminator
# g=Generator(2,16,2)
# for i in range(2):
#     g.add_smoothing_branch()
#     g.add_layer(with_smoothing=True)
# g.add_smoothing_branch()
# print (g)
# d=Discriminator(2,16,0.5)
# for i in range(2):
#     d.add_smoothing_branch()
#     d.add_layer(with_smoothing=True)

# d.add_smoothing_branch()
# print (d(g(y,with_smoothing=True),with_smoothing=True))

from Network import Generator,Discriminator,PGGAN
# g=Generator(2,16,2)
# for i in range(1):
#     g.add_smoothing_branch()
#     g.add_layer(with_smoothing=True)
# g.add_smoothing_branch()

# d=Discriminator(2,16,0.5)
# for i in range(1):
#     d.add_smoothing_branch()
#     d.add_layer(with_smoothing=True)
# d.add_smoothing_branch()

# for i,j in zip(g.data_loader,d.data_loader):
#     print (g(Variable(i),with_smoothing=True).size())
#     print("##########################################")
#     print (d(g(Variable(i),with_smoothing=True),with_smoothing=True).size())
#     print("##########################################")


# # print (d(g(y)))

import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from utils import *

# Hyper Parameters
# num_epochs = 5
# batch_size = 100
# learning_rate = 0.001

pggan=PGGAN()
pggan.train()