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

# from Network import Generator,Discriminator
# g=Generator(2,16,2)
# for i in range(2):
#     g.add_smoothing_branch()
#     g.add_layer(with_smoothing=True)
#     # print (g)


# x=np.random.rand(2,4,4)
# x=torch.Tensor(x)
# x.unsqueeze_(0)
# print (x.size())
# y=Variable(x)

# # avg_pool=nn.AvgPool2d(2,stride=0)
# # print (avg_pool(y))

# d=Discriminator(2,16,0.5,y.size())
# # print (d)
# for i in range(2):
#     d.add_smoothing_branch()
#     d.add_layer(with_smoothing=True)
# print (d)
# print (d.c_in,d.c_out,d.input_dim,d.output_dim)
A = torch.randn(8,8)
A.unsqueeze_(0)
A.unsqueeze_(0)
print (A.size())
print (A.repeat(5,8,8).size())