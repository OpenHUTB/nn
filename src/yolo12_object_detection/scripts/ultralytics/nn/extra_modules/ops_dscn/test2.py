import torch
from dscn import DSCNX,DSCNY
from torch.cuda.amp import autocast

criterion = torch.nn.CrossEntropyLoss().cuda()
inputs = torch.rand([1024,16,56,56]).cuda()
lbl = torch.ones_like(inputs)
dscn_x = DSCNX(16,7,3,pad=3).cuda()
dscn_y = DSCNY(16,7,3,pad=3).cuda()

hid = dscn_x(inputs)
outputs = dscn_y(hid)
loss = criterion(outputs,lbl)
print('forward done')
loss.backward(create_graph=True)
print('backward done')

print('input shape: ', inputs.shape)
print('output shape: ', outputs.shape)
print(outputs)