import torch
import numpy as np
from torch.distributions import Normal
x = torch.randint(0, 1, (3, 1000)).float()
y = torch.randint(0, 2, (3, 3))
print(x.shape)
print(y.shape)
# x = torch.cat((x, y), 1)
# print(x.shape)
x = torch.full((3, 1000), 1.0)
t = torch.full((3, 1000), 0.0)
# print(t)
print(x.exp())
normal = Normal(t, x)
z = normal.sample()
print(z.shape)
log_prob = normal.log_prob(z)
print(log_prob.shape)
