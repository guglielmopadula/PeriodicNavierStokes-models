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

a_train=a_train.unsqueeze(-1).repeat(1,1,1,50)
a_test=a_test.unsqueeze(-1).repeat(1,1,1,50)
a_super=a_super.unsqueeze(-1).repeat(1,1,1,100)

u_train=u_train.reshape(u_train.shape[0],-1)
u_test=u_test.reshape(u_test.shape[0],-1)
u_super=u_super.reshape(u_super.shape[0],-1)

a_train=a_train.reshape(a_train.shape[0],-1,1)
a_test=a_test.reshape(a_test.shape[0],-1,1)
a_super=a_super.reshape(a_super.shape[0],-1,1)

train_dataset=torch.utils.data.TensorDataset(a_train,all_points.unsqueeze(0).repeat(a_train.shape[0],1,1),u_train)
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=1)

test_dataset=torch.utils.data.TensorDataset(a_super,all_points_super.unsqueeze(0).repeat(a_super.shape[0],1,1),u_super)
test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=1)


len_points=len(points)
a_max=torch.max(a_train).item()
a_min=torch.min(a_train).item()
b_max=torch.max(u_train).item()
b_min=torch.min(u_train).item()#Averaging Neural Operator
class ANOLayer(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.lin=nn.Linear(hidden_size,hidden_size)

    def forward(self,x):
        return self.lin(x)+torch.mean(x,axis=1,keepdim=True)
    
class LAR(nn.Module):  
    def __init__(self,hidden_size):
        super().__init__()
        self.ano=ANOLayer(hidden_size)
        self.relu=nn.ReLU()
        self.bn=nn.BatchNorm1d(hidden_size)
        
    def forward(self,x):
        x=self.ano(x)
        #s=x.shape[1]
        #n=x.shape[0]
        #x=x.reshape(x.shape[0]*s,-1)
        #x=self.bn(x)
        #x=x.reshape(n,s,-1)
        return self.relu(x)

class ANO(nn.Module):
    def __init__(self,input_size,output_size,hidden_size,BATCH_SIZE):
    
        super().__init__()
        self.R=nn.Linear(input_size,hidden_size)
        self.hidden_layers=nn.Sequential(LAR(hidden_size),LAR(hidden_size),LAR(hidden_size))
        self.Q=nn.Linear(hidden_size+3,output_size)
    def forward(self, x , points):
        x=self.R(torch.cat((x,points),2))
        x=self.hidden_layers(x)
        x=self.Q(torch.cat((x,points),2))
        return x
    
    def forward_eval(self, x, points):
        x=(x-a_min)/(a_max-a_min)
        x=self.R(torch.cat((x,points),2))
        x=self.hidden_layers(x)
        x=self.Q(torch.cat((x,points),2))
        return x*(b_max-b_min)+b_min
    


    
model=ANO(4,1,100,10).cuda()
Epochs=100
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn=nn.MSELoss()
sup_m=0
low_m=0
for epoch in trange(Epochs):
    sup_m=0
    low_m=0
    for batch in train_dataloader:
        optimizer.zero_grad()
        v,points,u=batch
        v=v.cuda()
        u=u.cuda()
        points=points.cuda()
        v=v.reshape(v.shape[0],-1,1)
        u=u.reshape(u.shape[0],-1,1)
        points=points.reshape(points.shape[0],-1,3)
        pred=model.forward(v,points)
        pred=pred.reshape(pred.shape[0],-1,1)
        loss=torch.linalg.norm(pred-u)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            sup_m+=torch.sum(torch.linalg.norm(pred-u,axis=1)**2)/(len(train_dataloader))
            low_m+=torch.sum(torch.linalg.norm(u,axis=1)**2)/(len(train_dataloader))
    with torch.no_grad():
        print(f'Epoch: {epoch}, Loss: {torch.sqrt(sup_m/low_m)}')


model=model.eval()

train_rel_loss=0
sup_train_loss=0
low_train_loss=0


with torch.no_grad():
    for batch in train_dataloader:
        v,points,u=batch
        v=v.cuda()
        u=u.cuda()
        points=points.cuda()
        v=v.reshape(v.shape[0],-1,1)
        u=u.reshape(u.shape[0],-1,1)
        points=points.reshape(points.shape[0],-1,3)
        pred=model.forward(v,points)
        pred=pred.reshape(pred.shape[0],-1,1)
        u=u.cpu().numpy()
        pred=pred.cpu().numpy()
        sup_train_loss+=np.sum(np.linalg.norm(pred-u,axis=1)**2)/(len(train_dataloader))
        low_train_loss+=np.sum(np.linalg.norm(u,axis=1)**2)/(len(train_dataloader))
    train_rel_loss+=np.sqrt(sup_train_loss/low_train_loss)

print("Rel train loss", train_rel_loss)

test_rel_loss=0
sup_test_loss=0
low_test_loss=0
with torch.no_grad():
    for batch in test_dataloader:    
        v,points,u=batch
        v=v.cuda()
        u=u.cuda()
        points=points.cuda()
        v=v.reshape(v.shape[0],-1,1)
        u=u.reshape(u.shape[0],-1,1)
        points=points.reshape(points.shape[0],-1,3)
        pred=model.forward(v,points)
        pred=pred.reshape(pred.shape[0],-1,1)
        u=u.cpu().numpy()
        pred=pred.cpu().numpy()
        test_rel_loss+=np.mean(np.linalg.norm(u-pred,axis=1)/np.linalg.norm(u,axis=1))
    print("Tesr loss is", test_rel_loss)


u=u[0].reshape(128,128,100)
pred=pred[0].reshape(128,128,100)


import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
import matplotlib.animation as animation
import os
import sys
nSeconds = 2
fps=50
times_super=time_super.cpu().numpy()
name=os.path.splitext(os.path.basename(sys.argv[0]))[0]
u_all=u
u_rec_all=pred

err=u_all-u_rec_all
print(np.min(u))
print(np.max(u))

fig, axarr = plt.subplots(1,3,figsize=(24,8))
a=axarr[0].imshow(u_all[:,:,0],interpolation='none', aspect='auto', vmin=-0.7, vmax=3.3,cmap="inferno")
b=axarr[1].imshow(u_rec_all[:,:,0],interpolation='none', aspect='auto', vmin=-0.7, vmax=3.3,cmap="inferno")
c=axarr[2].imshow(err[:,:,0],interpolation='none', aspect='auto', vmin=-0.7, vmax=3.3,cmap="inferno")
axarr[0].set(xlabel='orig')
axarr[1].set(xlabel='rec')
axarr[2].set(xlabel='err')




def animate_func(i):
    a.set_array(u_all[:,:,i])
    b.set_array(u_rec_all[:,:,i])
    fig.suptitle('t='+str(times_super[i]))
    c.set_array(err[:,:,i])
    return [a,b,c]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )

anim.save('videos/u_'+name+'.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])

