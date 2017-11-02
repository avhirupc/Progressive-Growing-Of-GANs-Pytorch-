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
x=np.random.rand(1,2,2)
x=torch.Tensor(x)
x.unsqueeze_(0)
y=Variable(x)

from Network import Generator,Discriminator
g=Generator(2,16,2)
for i in range(2):
    g.add_smoothing_branch()
    g.add_layer(with_smoothing=True)
g.add_smoothing_branch()

print("##########################################")
d=Discriminator(2,16,0.5)
for i in range(2):
    d.add_smoothing_branch()
    d.add_layer(with_smoothing=True)

d.add_smoothing_branch()
print (d(g(y,with_smoothing=True),with_smoothing=True))

from Network import Generator,Discriminator
g=Generator(2,16,2)
for i in range(2):
    g.add_smoothing_branch()
    g.add_layer(with_smoothing=True)


print("##########################################")
d=Discriminator(2,16,0.5)
for i in range(2):
    d.add_smoothing_branch()
    d.add_layer(with_smoothing=True)

print (d(g(y)))