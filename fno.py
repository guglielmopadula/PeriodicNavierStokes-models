import numpy as np
from ezyrb import RBF, GPR
from scipy.linalg import svd
from periodicnavierstokes import PeriodicNavierStokes
from neuralop import TFNO
from tqdm import trange
import torch.nn as nn
import torch
import numpy as np
from tqdm import trange
np.random.seed(0)
batch_size=10
data=PeriodicNavierStokes(batch_size)
time=data.time_red
time_super=data.time
u_super=data.U_super
a_super=data.A_super
u_=data.U_test
a_test=data.A_test
a_train,u_train=data.train_dataset.tensors
a_test,u_test=data.test_dataset.tensors


a_test=a_test.unsqueeze(0).unsqueeze(1)
u_test=u_test.unsqueeze(0).unsqueeze(1)
a_super=a_super.unsqueeze(0).unsqueeze(1)
u_super=u_super.unsqueeze(0).unsqueeze(1)
u_train=u_train.unsqueeze(1)
a_train=a_train.unsqueeze(1)
u_test=u_test
a_test=a_test
u_super=u_super
a_super=a_super
a_train=a_train.unsqueeze(-1).repeat(1,1,1,1,50)
time_train=time.reshape(1,1,1,1,50).repeat(a_train.shape[0],1,a_train.shape[2],a_train.shape[3],1)
a_train=torch.concatenate((a_train,time_train),axis=1)
a_test=a_test.unsqueeze(-1).repeat(1,1,1,1,50)
time_test=time.reshape(1,1,1,1,50).repeat(a_test.shape[0],1,a_test.shape[2],a_test.shape[3],1)
a_test=torch.concatenate((a_test,time_test),axis=1)
a_super=a_super.unsqueeze(-1).repeat(1,1,1,1,100)
time_super_2=time_super.reshape(1,1,1,1,100).repeat(a_super.shape[0],1,a_super.shape[2],a_super.shape[3],1)
a_super=torch.concatenate((a_super,time_super_2),axis=1)


class ANN_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq=TFNO(n_modes=(16, 16, 16),in_channels=2 ,hidden_channels=32, out_channels=1)

    def forward(self,x):
        return self.seq(x)
    
    def fit(self,input,output,num_epochs,batch_size=1,lr=0.001):
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
                v=v.cuda()
                u=u.cuda()
                pred=self(v)
                pred=pred
                loss=torch.linalg.norm(pred-u)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    sup_m+=torch.sum(torch.linalg.norm(pred-u,axis=1)**2)
                    low_m+=torch.sum(torch.linalg.norm(u,axis=1)**2)
            with torch.no_grad():
                print(f'Epoch: {_}, Loss: {torch.sqrt(sup_m/low_m)}')

    def predict(self,input,batch_size=1):
        a=np.zeros((input.shape[0],1,input.shape[2],input.shape[3],input.shape[4]))
        input_torch=torch.tensor(input)
        dataset=torch.utils.data.TensorDataset(input_torch)
        data_loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)
        i=0
        j=batch_size
        with torch.no_grad():
            for batch in data_loader:
                v,=batch
                v=v.cuda()
                pred=self(v)
                pred=pred.cpu().numpy()
                a[i:j]=pred
                i=i+batch_size
                j=j+batch_size
            return a




model=ANN_1().cuda()

a_train=a_train.numpy()
u_train=u_train.numpy()
a_test=a_test.numpy()
u_test=u_test.numpy()
a_super=a_super.numpy()
u_super=u_super.numpy()



model.fit(a_train,u_train,100)

u_pred_test=model.predict(a_test)
u_pred_super=model.predict(a_super)
u_pred_train=model.predict(a_train)

u_train=u_train.reshape(u_train.shape[0],-1)
u_super=u_super.reshape(u_super.shape[0],-1)
u_test=u_test.reshape(u_test.shape[0],-1)

u_pred_train=u_pred_train.reshape(u_pred_train.shape[0],-1)
u_pred_super=u_pred_super.reshape(u_pred_super.shape[0],-1)
u_pred_test=u_pred_test.reshape(u_pred_test.shape[0],-1)

loss_train=np.mean(np.linalg.norm(u_train-u_pred_train,axis=1)/(np.linalg.norm(u_train,axis=1)))
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

