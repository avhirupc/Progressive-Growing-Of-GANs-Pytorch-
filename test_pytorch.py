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

from Network import Generator,Discriminator
# g=Generator(2,16,2)
# # print (g(y)[0][0][0])
# g.add_smoothing_branch()
# # print (g(y,with_smoothing=True).size())
# print (g)
# g.add_layer()
# print (g)
# print (g(y).size())


x=np.random.rand(4,8,8)
x=torch.Tensor(x)
x.unsqueeze_(0)
print (x.size())
y=Variable(x)

# avg_pool=nn.AvgPool2d(2,stride=0)
# print (avg_pool(y))

d=Discriminator(2,16,0.5,y.size())
print (d(y).size())
d.add_smoothing_branch()
print(nn.Sequential(*d.will_be_next_layers))
print ('#@################################')
print (d)
print (d(y,with_smoothing=True))
print (d)
d.add_layer(with_smoothing=True)
print (d(y).size())
print (d)
