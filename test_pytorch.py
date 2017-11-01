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
print (g(y).size())
g.add_layer()
print (g(y).size())
g.add_layer()
print (g(y).size())
fake_image=g(y)
d=Discriminator(2,16,0.5,fake_image.size())
print (d(fake_image).size())

d.add_layer()
print (d(fake_image).size())
d.add_layer()
print (d(fake_image).size())
d.add_layer()
print (d(fake_image).size())