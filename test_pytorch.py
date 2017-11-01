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

from Network import Generator,Discriminator
g=Generator(2,16,2)
# print (g(y)[0][0][0])
g.add_smoothing_branch()
# print (g(y,with_smoothing=True).size())
print (g)
g.add_layer()
print (g)
# print (g(y).size())
