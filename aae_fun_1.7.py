""" 
Adversarial autoencoder over Airline data (/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_airways)

"""

import argparse
import os
import numpy as np
from math import log
import itertools
import time 
import sys

from torch.utils.data.dataset import TensorDataset as TensorDataset


from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import vamb.vambtools as _vambtools
from data_loader import make_dataloader # aae/data_loader.py
from calc_loss import calc_loss

time0=time.time()


parser = argparse.ArgumentParser()
parser.add_argument("--h_n",default=512, type=int, help="# neurons hidden layers ")
parser.add_argument("--l_d",default=32, type=int, help="# neurons latent layer ")
parser.add_argument("--s_l",default=0.005, type=float, help="scale loss")
parser.add_argument("--h_n_d", type=int, help="# neurons hidden layers discriminator")
parser.add_argument("--batch_size", type=int,default=64, help="Batch size (be aware that it will be doubled laters)")


parser.add_argument("-s", type=int,default=25, help="save model every s epochs")

parser.add_argument("-e",default=300, type=int, help="# epochs")
parser.add_argument("-d", type=str, help="Dataset")
parser.add_argument("-G", type=str, help="If grid")
parser.add_argument("-r", type=str, help="Run or version")
parser.add_argument("--Test", type=str, help="Test run for errors check")

args = vars(parser.parse_args())
print(args)

opt = parser.parse_args()

dataset=opt.d

run=opt.r

#savecheck
save_checkpoint=int(opt.s)
## DATA PATHS

path= '/home/projects/cpr_10006/people/paupie/aae/'

if dataset == 'Airways':
    path_data= '/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_airways/'

elif dataset == 'Skin':

    path_data= '/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_skin/'

elif dataset == 'Oral':
    path_data= '/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_oral/'

elif dataset == 'Gi':
    path_data= '/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_gi/'

elif dataset == 'Urog':
    path_data= '/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_urog/'

elif dataset == 'Metahit':
    path_data= '/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_metahit/'

else: 
    print('\nDATASET: ',dataset,'\n')

### HYPERPARAMETERS

h_n = opt.h_n
s_l = opt.s_l
latent_dim = opt.l_d

h_n_d = opt.h_n_d
if not h_n_d: h_n_d = h_n
n_epochs =  opt.e
batch_size = int(opt.batch_size)
lr=1e-3

## GRID OR NOT GRID SEARCH AND NAMING

grid = opt.G

if grid == 'yes':
    grid_dir = 'Grid_'+run
    
    grid_name = 'grid_'+dataset+'_'+run

    train_data=dataset+'_Grid_'+run+'_s_l_'+str(s_l)+'_hn_'+str(h_n)+'_l_d_'+str(latent_dim)+'_bs_'+str(batch_size)   
 



else:
    grid_dir = '.' 
    
    train_data=dataset+'_'+run+'_s_l_'+str(s_l)+'_hn_'+str(h_n)+'_l_d_'+str(latent_dim)+'_bs_'+str(batch_size)   
   
    grid_name = train_data
print('\nGrid dir:',grid_dir,'\nLog name:',grid_name,'\nFile name:',train_data,'\n')
os.system('mkdir -p ' + path+'Models/' +dataset+'/'+ grid_dir )
os.system('mkdir -p ' + path+'Models/' +dataset+'/'+ grid_dir+'/epochs' )
os.system('mkdir -p ' + path+'../Data_plots/' +dataset+'/'+ grid_dir )
os.system('mkdir -p ' + path+'Plots/' +dataset+'/'+ grid_dir )
os.system('mkdir -p ' + path+'Latents/' +dataset+'/'+ grid_dir )
os.system('mkdir -p ' + path+'Clusters_2/' +dataset+'/'+ grid_dir )
os.system('mkdir -p ' + path+'Clusters_3/' +dataset+'/'+ grid_dir )
os.system('mkdir -p ' + path+'Clusters_k/' +dataset+'/'+ grid_dir )

os.system('mkdir -p ' + path+'Clusters_2_split/' +dataset+'/'+ grid_dir )
os.system('mkdir -p ' + path+'Clusters_3_split/' +dataset+'/'+ grid_dir )
os.system('echo -e '+train_data+'_e_0'+ ' >> '+path + 'Logs/'+grid_name+'_log') 
os.system('mkdir -p ' + path+'Clusters_k_split/' +dataset+'/'+ grid_dir )# grid_name_log file will contain the list of models trained, clustered ,etc

### LOAD DATA

depths=np.load(path_data+'depths.npz')
tnf=np.load(path_data+'result/tnf.npz')

depths=depths['arr_0']
tnfs=tnf['arr_0']

Testing =opt.Test 

if Testing :
    print('\n TEST RUN\n')
    n_epochs = 25
    batch_size = 1024
    depths=depths[:1024]
    tnfs=tnfs[:1024]


## Some  model setttings depending on the dataset


input_len = depths.shape[1]+tnfs.shape[1]

num_samples=int(depths.shape[1])

## If available, use GPU leverage 


cuda = True if torch.cuda.is_available() else False


def reparameterization(mu, logvar):
    
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))  
    if cuda:
        sampled_z=sampled_z.cuda()
    z = sampled_z * std + mu
    return z



class Encoder(nn.Module):
    def __init__(self,h_n,latent_dim):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(input_len), h_n),
            nn.BatchNorm1d(h_n),
            nn.LeakyReLU(),
            nn.Linear(h_n, h_n),
            nn.BatchNorm1d(h_n),
            nn.LeakyReLU(),
       )

        self.mu = nn.Linear(h_n, latent_dim)
        self.logvar = nn.Linear(h_n, latent_dim)# ????


    def forward(self,depths,tnfs):
        I=torch.cat((depths,tnfs),1)
        x = self.model(I)#img_flat)
        mu = self.mu(x)
        
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self,h_n,latent_dim):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, h_n),
            nn.BatchNorm1d(h_n),
            nn.LeakyReLU(),
            nn.Linear(h_n, h_n),
            nn.BatchNorm1d(h_n),
            nn.LeakyReLU(),
            nn.Linear(h_n, int(input_len)), 
        )

    def forward(self, z):
        reconstruction = self.model(z)
        
        Depths_out,tnf_out =(reconstruction[:,:num_samples],
                reconstruction[:,num_samples:])
        
        depths_out=F.softmax(Depths_out,dim=1)
        
        return reconstruction,depths_out,tnf_out


class Discriminator(nn.Module):
    def __init__(self,h_n_d,latent_dim):

        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, int(h_n_d)),
            nn.LeakyReLU(),
            nn.Linear(int(h_n_d), int(h_n_d/2)),
            nn.LeakyReLU(),
            nn.Linear(int(h_n_d/2), 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity




###   LOSS
# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCELoss()
#adversarial_loss = torch.nn.BCEWithLogitsLoss()



# Initialize generator and discriminator
encoder = Encoder(h_n,latent_dim)
decoder = Decoder(h_n,latent_dim)
discriminator = Discriminator(h_n_d,latent_dim)






if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    #reconstruction_loss.cuda()


# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()))

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr )

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



# ----------
#  Training
# ----------
def train_epoch(dataloader,epoch_i,epoch_T,G_loss,D_r_loss,D_f_loss,epoch_len):
    
    for i, (depths_in, tnfs_in) in enumerate(dataloader): 
        imgs=torch.cat((depths_in,tnfs_in),dim=1)
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        #encoded_imgs = encoder(real_imgs)
        if cuda == True:
            depths_in=depths_in.cuda()
            tnfs_in=tnfs_in.cuda()

        encoded_imgs = encoder(depths_in,tnfs_in)

        decoded_imgs,depths_out,tnfs_out= decoder(encoded_imgs)

        # Loss measures generator's ability to fool the discriminator
       
        vae_loss,ce,sse = calc_loss(depths_in, depths_out, tnfs_in, tnfs_out,num_samples,alpha=0.15)
         
        g_loss = s_l * adversarial_loss(discriminator(encoded_imgs), valid) + (1-s_l) * vae_loss

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()
        #print(
        #    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [CE: %f] [SSE: %f]  "
        #    % (epoch_i, n_epochs, i, epoch_len, d_loss.item(), g_loss.item(),ce,sse))
        
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D real loss: %f] [D fake loss: %f] [G loss: %f] [CE: %f] [SSE: %f]  "
            % (epoch_i, n_epochs, i, epoch_len, real_loss.item(),fake_loss.item(), g_loss.item(),ce,sse))
        
        G_loss.append(float(g_loss.detach()))
        D_r_loss.append(float(real_loss.detach()))
        D_f_loss.append(float(fake_loss.detach()))       #batches.append(batches_done)
     
     
    return G_loss,D_r_loss,D_f_loss

def train_model(Path=path,batchsize=batch_size,n_epochs=n_epochs,batchsteps=[25, 75, 150, 225]):
    G_loss=[]
    D_loss=[]
    
    print('Training model\n')

    epochs_num=len(make_dataloader(depths,tnfs,batchsize,Shuffle=True,Cuda=cuda)[0])

    for epoch in range(n_epochs):
        print('\n\n EPOCH:',epoch,'\n\n')
        if epoch in batchsteps:
            batchsize *=2
            epochs_num=len(make_dataloader(depths,tnfs,batchsize,Shuffle=True,Cuda=cuda)[0])
        if epoch == 0:
            G_loss=[]
            D_loss=[]
            D_r_loss=[]
            D_f_loss=[]

            G_loss_F,D_r_loss_F,D_f_loss_F=train_epoch(dataloader=make_dataloader(depths,tnfs,batchsize,Cuda=cuda,Shuffle=True)[0],epoch_i=epoch,epoch_T=n_epochs,G_loss=G_loss,D_r_loss=D_r_loss,D_f_loss=D_f_loss,epoch_len=epochs_num)
        else:
            G_loss_i,D_r_loss_i,D_f_loss_i=train_epoch(dataloader=make_dataloader(depths,tnfs,batchsize,Cuda=cuda,Shuffle=True)[0],epoch_i=epoch,epoch_T=n_epochs,G_loss=G_loss_F,D_r_loss=D_r_loss_F,D_f_loss=D_f_loss_F,epoch_len=epochs_num)
            G_loss_F,D_r_loss_F,D_f_loss_F=G_loss_i,D_r_loss_i,D_f_loss_i

        if (epoch+1) % save_checkpoint  == 0:
            print('\n\n Saving model at epoch ',epoch,'\n\n')

            torch.save(decoder.state_dict(), Path+'Models/'+dataset+'/'+grid_dir+'/epochs/Decoder_'+train_data + '_e_'+str(epoch+1))      

            torch.save(encoder.state_dict(), Path+'Models/'+dataset+'/'+grid_dir+'/epochs/Encoder_'+train_data+'_e_'+str(epoch+1))
            torch.save(discriminator.state_dict(), Path+'Models/'+dataset+'/'+grid_dir+'/epochs/Discriminator_'+train_data+'_e_'+str(epoch+1))
            os.system('echo -e '+train_data + '_e_' + str(epoch+1) +' >> '+Path + 'Logs/'+grid_name+'_log')
            g_loss_epoch =np.round(np.mean((np.asarray(G_loss[-save_checkpoint:]))),3)
            d_r_loss_epoch =np.round(np.mean((np.asarray(D_r_loss[-save_checkpoint:]))),3)
            d_f_loss_epoch =np.round(np.mean((np.asarray(D_f_loss[-save_checkpoint:]))),3)


            os.system('echo -e '+train_data + '_e_' + str(epoch+1) +' '+ str(g_loss_epoch) +' '+ str(d_r_loss_epoch) +' '+ str(d_f_loss_epoch) + '  >> '+Path + 'Logs/'+grid_name+'_loss')

            
            

    print('\n\n Saving model at epoch ',epoch,'\n\n')

    torch.save(decoder.state_dict(), Path+'Models/'+dataset+'/'+grid_dir+'/Decoder_'+train_data + '_e_'+str(epoch+1))      

    torch.save(encoder.state_dict(), Path+'Models/'+dataset+'/'+grid_dir+'/Encoder_'+train_data+'_e_'+str(epoch+1))
    torch.save(discriminator.state_dict(), Path+'Models/'+dataset+'/'+grid_dir+'/Discriminator_'+train_data+'_e_'+str(epoch+1))
    os.system('echo -e '+train_data + '_e_' + str(epoch+1) +' >> '+Path + 'Logs/'+grid_name+'_log')
    g_loss_epoch =np.round(np.mean((np.asarray(G_loss[-25:-1]))),3)
    d_r_loss_epoch =np.round(np.mean((np.asarray(D_r_loss[-25:-1]))),3)
    d_f_loss_epoch =np.round(np.mean((np.asarray(D_f_loss[-25:-1]))),3)
                                                                       
    #os.system('echo -e '+train_data + '_e_' + str(epoch+1) +' '+ str(g_loss_epoch) +' '+ str(d_r_loss_epoch) +' '+ str(d_f_loss_epoch)  + '  >> '+Path + 'Logs/'+grid_name+'_loss')



 
    plt.plot(np.arange(len(G_loss))+1,G_loss,'-b',label='Generator')
    plt.plot(np.arange(len(D_r_loss))+1,D_r_loss,'-r',label='Discriminator real',linewidth=0.1)  
    plt.plot(np.arange(len(D_f_loss))+1,D_f_loss,'-g',label='Discriminator fake',linewidth=0.1)
    #plt.ylim([0,1])
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title("#model name : "+train_data)
 
    G_loss_a=np.asarray(G_loss)
    D_r_loss_a=np.asarray(D_r_loss)
    D_f_loss_a=np.asarray(D_f_loss)
    

    np.savez(path+'../Data_plots/'+dataset+'/'+grid_dir+'/'+train_data+'.npz',G_loss_a,D_r_loss_a,D_f_loss_a)   
   
    plt.legend(loc="upper right")
    plt.savefig(path+'Plots/'+dataset+'/'+grid_dir+'/training_loss_'+train_data+'.png')
    
     

train_model()

time_1=time.time()

time_run=np.round((time_1-time0),2)
os.system('echo -e  trained  '+str(time_run) +' >> '+path + 'Logs/'+grid_name+'_log')





