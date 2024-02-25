from periodicnavierstokes import PeriodicNavierStokes
import torch.nn as nn
import torch
import numpy as np
from tqdm import trange
batch_size=10
a=PeriodicNavierStokes(batch_size)
points=a.points_red.cuda().reshape(-1,3)[:,:-1]
points_super=a.points.cuda().reshape(-1,3)[:,:-1]

time=a.time_red.cuda()
time_super=a.time.cuda()
print(time.shape)
print(time_super.shape)
print(points.shape)
print(points_super.shape)



train_dataloader=a.train_loader
a_test=a.A_test
u_test=a.U_test
a_super=a.A_super
u_super=a.U_super



len_points=len(points)
a,b=a.train_dataset.tensors
a_max=torch.max(a).item()
a_min=torch.min(a).item()
b_max=torch.max(b).item()
b_min=torch.min(b).item()

class DeepONet(nn.Module):
    def __init__(self,
                num_start_points,
                medium_size
                ):
    
        super().__init__()
        self.num_start_points=num_start_points
        self.medium_size=medium_size
        self.branch_net_1=nn.Sequential(nn.Linear(num_start_points,100),nn.BatchNorm1d(100),nn.ReLU(),nn.Linear(100,100), nn.BatchNorm1d(100),nn.ReLU(),nn.Linear(100,100),nn.BatchNorm1d(100),nn.ReLU(), nn.Linear(100,medium_size))
        self.branch_net_2=nn.Sequential(nn.Linear(1,100),nn.BatchNorm1d(100),nn.ReLU(),nn.Linear(100,100), nn.BatchNorm1d(100),nn.ReLU(),nn.Linear(100,100),nn.BatchNorm1d(100),nn.ReLU(), nn.Linear(100,medium_size))
        self.trunk_net=nn.Sequential(nn.Linear(2,100),nn.BatchNorm1d(100), nn.ReLU(), nn.Linear(100,100), nn.BatchNorm1d(100),nn.ReLU(),nn.Linear(100,100), nn.BatchNorm1d(100),nn.ReLU(), nn.Linear(100,medium_size))
    
    def forward(self, x, points,times):
        x=torch.sum(self.branch_net_1(x).unsqueeze(1).unsqueeze(1)*self.trunk_net(points).unsqueeze(0).unsqueeze(2)*self.branch_net_2(times).unsqueeze(0).unsqueeze(0),axis=3)
        return x
    
    def forward_eval(self, x, points,times):
        x=(x-a_min)/(a_max-a_min)
        x=torch.sum(self.branch_net_1(x).unsqueeze(1).unsqueeze(1)*self.trunk_net(points).unsqueeze(0).unsqueeze(2)*self.branch_net_2(times.unsqueeze(1)).unsqueeze(0).unsqueeze(0),axis=3)
        return x*(b_max-b_min)+b_min
    
index_subset=torch.randperm(len_points)[:1000]

model=DeepONet(1000,100).cuda()
Epochs=500
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)
loss_fn=nn.MSELoss()
sup_m=0
low_m=0
for epoch in trange(Epochs):
    sup_m=0
    low_m=0
    for batch in train_dataloader:
        optimizer.zero_grad()
        v,u=batch
        v=v.reshape(v.shape[0],-1).cuda()
        u=u.reshape(u.shape[0],-1,u.shape[-1])
        u=u.cuda()
        pred=model.forward_eval(v[:,index_subset],points,time)
        pred=pred.reshape(pred.shape[0],-1,u.shape[-1])
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
        v,u=batch
        v=v.cuda().reshape(batch_size,-1)
        v=v.reshape(v.shape[0],-1)
        u=u.reshape(u.shape[0],-1,u.shape[-1])
        u=u.cuda()
        pred=model.forward_eval(v[:,index_subset],points,time)
        pred=pred.reshape(pred.shape[0],-1,u.shape[-1])
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
    v=a_test.unsqueeze(0)
    u=u_super.unsqueeze(0)
    v=v.cuda()
    u=u.cuda()
    v=v.reshape(v.shape[0],-1)
    u=u.reshape(u.shape[0],-1,u.shape[-1])
    pred=model.forward_eval(v[:,index_subset],points_super, time_super)
    pred=pred.reshape(pred.shape[0],-1,pred.shape[-1])
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

