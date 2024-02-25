import numpy as np
from ezyrb import RBF, GPR, POD
from sklearn.tree import DecisionTreeRegressor
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
time=data.time_red.numpy().reshape(-1,1)
points_super=data.points.reshape(128*128,3)[:,:-1]
time_super=data.time.numpy().reshape(-1,1)
all_points=points
all_points=all_points.reshape(-1,2)
all_points_super=points_super
all_points_super=all_points_super.reshape(-1,2)
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
u_train=u_train.reshape(u_train.shape[0],-1,50)
a_train=a_train.reshape(u_train.shape[0],-1)
u_test=u_test.reshape(u_test.shape[0],-1,50)
a_test=a_test.reshape(u_test.shape[0],-1)
u_super=u_super.reshape(u_super.shape[0],-1,100)
a_super=a_super.reshape(u_super.shape[0],-1)

points=all_points
points_super=all_points_super

A=np.load("A.npy")
B=np.load("B.npy")
C=np.load("C.npy")

class Operator():
    def __init__(self,m,p): #m is the number of sensors
        self.model_trunk=DecisionTreeRegressor()
        self.model_branch_1=DecisionTreeRegressor()
        self.model_branch_2=DecisionTreeRegressor()
        self.pod=POD(method='randomized_svd',rank=p)
        self.m=m
    
    def fit(self,x,points,times,A,B,C):
        self.indices=np.random.permutation(64*64)[:self.m]
        self.model_branch_1.fit(x[:,self.indices],A)
        self.model_branch_2.fit(times,C) #time is last
        self.model_trunk.fit(points,B)
    def predict(self,x,points,time):
        x_red=x[:,self.indices]
        return np.einsum('ip,jp,kp->ijk',self.model_branch_1.predict(x_red),self.model_trunk.predict(points),self.model_branch_2.predict(time))
    

model=Operator(p=2000,m=1000)

a_train=a_train.numpy()
points=points.numpy()
u_train=u_train.numpy()
a_test=a_test.numpy()
u_test=u_test.numpy()
a_super=a_super.numpy()
u_super=u_super.numpy()

points=points
model.fit(a_train,points,time,A,B,C)

u_pred_test=model.predict(a_test,points,time)
u_pred_super=model.predict(a_test,points_super,time_super)
u_pred_train=model.predict(a_train,points,time)



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
times=time_super.reshape(-1)
name=os.path.splitext(os.path.basename(sys.argv[0]))[0]
u_all=u
u_rec_all=pred

err=u_all-u_rec_all
print(np.min(u))
print(np.max(u))
print(np.min(u_rec_all))
print(np.max(u_rec_all))

print(np.min(err))
print(np.max(err))

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

