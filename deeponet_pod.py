import numpy as np
from ezyrb import RBF, GPR, POD#, ANN
import scipy.linalg
from periodicnavierstokes import PeriodicNavierStokes
from tqdm import trange
import torch.nn as nn
import torch
import numpy as np
from tqdm import trange
np.random.seed(0)
batch_size=10
data=PeriodicNavierStokes(batch_size)
points=data.points_red[:,:-1]
time=data.time_red
points_super=data.points.reshape(128*128,3)[:,:-1]
time_super=data.time
all_points=torch.concatenate((time.unsqueeze(0).unsqueeze(-1).repeat(4096,1,1),points.unsqueeze(-2).repeat(1,50,1)),dim=2)
all_points=all_points.reshape(-1,3)
all_points_super=torch.concatenate((time_super.unsqueeze(0).unsqueeze(-1).repeat(128*128,1,1),points_super.unsqueeze(-2).repeat(1,100,1)),dim=2)
all_points_super=all_points_super.reshape(-1,3)
u_super=data.U_super
a_super=data.A_super
u_=data.U_test
a_test=data.A_test
a_train,u_train=data.train_dataset.tensors
a_test,u_test=data.test_dataset.tensors


a_test=a_test.unsqueeze(0)
u_test=u_test.unsqueeze(0)
a_super=a_super.unsqueeze(0)
u_super=u_super.unsqueeze(0)
u_train=u_train.reshape(u_train.shape[0],-1)
a_train=a_train.reshape(u_train.shape[0],-1)
u_test=u_test.reshape(u_test.shape[0],-1)
a_test=a_test.reshape(u_test.shape[0],-1)
u_super=u_super.reshape(u_super.shape[0],-1)
a_super=a_super.reshape(u_super.shape[0],-1)
a_train=a_train.unsqueeze(-1).repeat(1,1,50).reshape(a_train.shape[0],-1)
a_test=a_test.unsqueeze(-1).repeat(1,1,50).reshape(a_test.shape[0],-1)

points=all_points
points_super=all_points_super

class ANN(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.nn=nn.Sequential(
            nn.Linear(input_size,500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500,output_size)
        )
        self.output_size=output_size
    def forward(self,x):
        return self.nn(x)
    
    def fit(self,input,output,num_epochs,batch_size=-1,lr=0.001):
        if batch_size==-1:
            batch_size=input.shape[0]
        input_torch=torch.tensor(input)
        output_torch=torch.tensor(output)
        print(input_torch.shape)
        print(output_torch.shape)
        dataset=torch.utils.data.TensorDataset(input_torch,output_torch)
        data_loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)
        optimizer=torch.optim.AdamW(self.parameters(),lr=lr)
        for _ in trange(num_epochs):
            sup_m=0
            low_m=0
            for batch in data_loader:
                optimizer.zero_grad()
                v,u=batch
                v=v.cuda().reshape(batch_size,-1)
                u=u.reshape(batch_size,-1)
                u=u.cuda()
                pred=self(v)
                pred=pred.reshape(batch_size,-1)
                loss=torch.linalg.norm(pred-u)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    sup_m+=torch.sum(torch.linalg.norm(pred-u,axis=1)**2)
                    low_m+=torch.sum(torch.linalg.norm(u,axis=1)**2)
            with torch.no_grad():
                print(f'Epoch: {_}, Loss: {torch.sqrt(sup_m/low_m)}')

    def predict(self,input,batch_size=-1):
        a=np.zeros((input.shape[0],self.output_size))
        input_torch=torch.tensor(input)
        dataset=torch.utils.data.TensorDataset(input_torch)
        data_loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)
        i=0
        j=batch_size
        with torch.no_grad():
            for batch in data_loader:
                v,=batch
                v=v.cuda().reshape(batch_size,-1)
                pred=self(v)
                pred=pred.reshape(batch_size,-1).cpu().numpy()
                a[i:j]=pred
                i=i+batch_size
                j=j+batch_size
            return self(torch.tensor(input).cuda()).cpu().numpy().reshape(input.shape[0],-1)
    

class Operator():
    def __init__(self,m,p): #m is the number of sensors
        self.model_trunk=ANN(3,p).cuda()
        self.model_branch=ANN(m,p).cuda()
        self.pod=POD(method='randomized_svd',rank=p)
        self.m=m
        self.p=p
    
    def fit(self,x,y,points):
        self.pod.fit(y)
        self.modes=self.pod.modes
        self.basis=self.pod.transform(y)
        self.indices=np.random.permutation(self.basis.shape[1])[:self.m]

        self.model_branch.fit(x[:,self.indices],self.modes,10000,600)
        self.model_branch.eval()
        print(np.linalg.norm(self.model_branch.predict(x[:,self.indices],batch_size=600)-self.modes)/np.linalg.norm(self.modes))
        self.model_trunk.fit(points,self.basis.T,100,8192,1e-02)
        self.model_trunk.eval()
        print(np.linalg.norm(self.model_trunk.predict(points,8192)-self.basis.T)/np.linalg.norm(self.basis.T))
        
    def predict(self,x,points):
        x_red=x[:,self.indices]
        return self.model_branch.predict(x_red,batch_size=1)@(self.model_trunk.predict(points,batch_size=8192).T)
    

model=Operator(p=500,m=1000)

a_train=a_train.numpy()
points=points.numpy()
u_train=u_train.numpy()
a_test=a_test.numpy()
u_test=u_test.numpy()
a_super=a_super.numpy()
u_super=u_super.numpy()

points=points
model.fit(a_train,u_train,points)

u_pred_test=model.predict(a_test,points)
u_pred_super=model.predict(a_test,points_super)
u_pred_train=model.predict(a_train,points)



loss_train=np.mean(np.linalg.norm(u_train-u_pred_train,axis=1)/np.linalg.norm(u_train,axis=1))
loss_test=np.mean(np.linalg.norm(u_test-u_pred_test,axis=1)/np.linalg.norm(u_test,axis=1))
loss_super=np.mean(np.linalg.norm(u_super-u_pred_super,axis=1)/np.linalg.norm(u_super,axis=1))

print(loss_train)
print(loss_test)
print(loss_super)

u=u_super[0].reshape(128,128,100)
pred=u_pred_super[0].reshape(128,128,100)


import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
import matplotlib.animation as animation
import os
import sys

print(a_super.shape)
fig, axarr = plt.subplots(1,2,figsize=(16,8))




nSeconds = 2
fps=50
times=time_super.cpu().numpy()
name=os.path.splitext(os.path.basename(sys.argv[0]))[0]
u_all=u
u_rec_all=pred

err=u_all-u_rec_all


fig, axarr = plt.subplots(1,3,figsize=(24,8))
a=axarr[0].imshow(u_all[:,:,0],interpolation='none', aspect='auto', vmin=-0.7, vmax=3.3, cmap="inferno")
b=axarr[1].imshow(u_rec_all[:,:,0],interpolation='none', aspect='auto', vmin=-0.7, vmax=3.3,cmap="inferno")
c=axarr[2].imshow(err[:,:,0],interpolation='none', aspect='auto', vmin=-0.7, vmax=3.3, cmap="inferno")
axarr[0].set(xlabel='orig')
axarr[1].set(xlabel='rec')
axarr[2].set(xlabel='err')

def animate_func(i):
    a.set_array(u_all[:,:,i])
    b.set_array(u_rec_all[:,:,i])
    fig.suptitle('t='+str(times[i]))
    c.set_array(err[:,:,i])
    return [a,b,c]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )

anim.save('videos/u_'+name+'.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])


fig, axarr = plt.subplots(1,2,figsize=(16,8))
a=axarr[0].imshow(u_all[:,:,0],interpolation='none', aspect='auto', vmin=-0.7, vmax=3.3, cmap="inferno")
b=axarr[1].imshow(u_rec_all[:,:,0],interpolation='none', aspect='auto', vmin=-0.7, vmax=3.3,cmap="inferno")
axarr[0].set(xlabel='orig')
axarr[1].set(xlabel='rec')


def animate_func(i):
    a.set_array(u_all[:,:,i])
    b.set_array(u_rec_all[:,:,i])
    fig.suptitle('t='+str(times[i]))
    c.set_array(err[:,:,i])
    return [a,b,c]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )

anim.save('videos/u_no_err'+name+'.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])

