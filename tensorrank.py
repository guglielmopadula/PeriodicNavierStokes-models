from periodicnavierstokes import PeriodicNavierStokes
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import svd
from tqdm import trange
import torch
import numpy as np
import tensorly
data=PeriodicNavierStokes(10)
a_train,u_train=data.train_dataset.tensors
u_train=u_train.reshape(600,64*64,50)

print(u_train)

'''
a=np.random.rand(600,100)
b=np.random.rand(64*64,100)
c=np.random.rand(50,100)

a=np.einsum('ip,jp,kp->ijk',a,b,c)
'''
class Fun(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A=torch.nn.Parameter(torch.rand(600,100).float())
        self.B=torch.nn.Parameter(torch.rand(64*64,100).float())
        self.C=torch.nn.Parameter(torch.rand(50,100).float())

    def forward(self):
        return torch.einsum('ip,jp,kp->ijk',self.A,self.B,self.C)
fun=Fun()
optimizer=torch.optim.Adam(fun.parameters(),lr=0.1)
for i in trange(1000):
    optimizer.zero_grad()
    pred=fun.forward()
    loss=torch.linalg.norm(pred-u_train)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        print(torch.linalg.norm(pred-u_train)/torch.linalg.norm(u_train))

with torch.no_grad():
    np.save("A.npy",fun.A.numpy())
    np.save("B.npy",fun.B.numpy())
    np.save("C.npy",fun.C.numpy())