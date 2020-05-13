""" 
1.Obtain the Airways data latent representation (/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_airways) using the trained aae model,  
2. Apply the PCA transformation on the latents and plot the PCA1 and PCA2 of 10 randomly OTUs (/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_airways/jakni/reference_new.tsv)
3. Cluster the latent representation using the Vamb clustering algorithm 

4. Benchmark againts Vamb and MetaBAT2 (aae_benchmark.py)
"""

import argparse
import os
import numpy as np
from math import log
import itertools
import time 
import random
import sys
import gzip


from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt

import vamb.vambtools as _vambtools
import vamb



from data_loader import make_dataloader # aae/data_loader.py


# Flags 
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=512, help="size of the batches")

parser.add_argument("-m", type=str,  help="Model")
parser.add_argument("-d", type=str,  help="Dataset")


parser.add_argument("-t", type=float,default=0.9, help="Precision and recall threshold")
parser.add_argument("-G", type=str,  help="Is model part of grid search?")
parser.add_argument("--load_model", type=str,default='True',  help="Do we need to generate the latents?")
parser.add_argument("--clust", type=str,default='',  help="Clustering algorithms (k32")

parser.add_argument("--epochs", type=str,default='',  help="Run over different epochs")

parser.add_argument("--tag", type=str,default='',  help="extra tag name")
args = vars(parser.parse_args())
print(args)

opt = parser.parse_args()


Load_model=opt.load_model
train_data = opt.m # 'Airways_Grid__hn_d_100'
dataset = opt.d
grid=opt.G
clust_alg = opt.clust
epochs=opt.epochs
print('\nLoad model : ', Load_model,'\n')

origin_dataset=train_data.split('_')[0]
print('\n Model trained on:',origin_dataset,'\n')

tag=opt.tag
# Set some paths with important files 
path='/home/projects/cpr_10006/people/paupie/aae/'# aae directory

if dataset == 'Airways':
    path_data='/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_airways/'# data directory
    path_ref='/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/airways/'


elif dataset == 'Skin':

    path_data= '/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_skin/'
    path_ref='/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/skin/'

elif dataset == 'Oral':
    path_data= '/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_oral/'
    path_ref='/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/oral/'


elif dataset == 'Gi':
    path_data= '/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_gi/'
    path_ref='/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/gi/'


elif dataset == 'Urog':
    path_data= '/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_urog/'
    path_ref='/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/urog/'


elif dataset == 'Metahit':
    path_data= '/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_metahit/'
    path_ref='/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/metahit/'


   

if grid == 'yes':
    
    try:
        grid_dir=train_data.split('_s_l')[0].split(origin_dataset+'_')[1]

    except:
        
        try:
            grid_dir=train_data.split('_sl')[0].split(origin_dataset+'_')[1]
        
        except:
            
            pass



    print('\nGRID DIR:',grid_dir,'\n') 
    h_n=int(train_data.split('_hn_')[1].split('_l_d_')[0])
    latent_dim=int(train_data.split('_hn_')[1].split('_l_d_')[1].split('_bs_')[0])
    
    try:
        epoch =int(train_data.split('_e_')[1])
    except:
        pass
    try:
        log_results_name='grid_'+dataset +'_'+grid_dir.split('_')[1]+tag+'_results'#+clust_alg

    except:
        
        try:
            log_results_name='grid_'+dataset +tag+'_results'#+clust_alg

        except:
            pass
else:
    
    grid_dir='.'
    log_results_name=train_data+tag+'_results'

    h_n=int(train_data.split('_hn_')[1].split('_l_d_')[0])
    latent_dim=int(train_data.split('_hn_')[1].split('_l_d_')[1].split('_bs_')[0])
    try:
        epoch =int(train_data.split('_e_')[1])
    except:
        pass
print('\nLOG FILE',log_results_name,'\n')


if 'k' in  clust_alg:   
    os.system('mkdir ' + path+'Clusters_k_split/' +dataset+'/'+ grid_dir )
    
    os.system('mkdir ' + path+'Clusters_k/' +dataset+'/'+ grid_dir )
        
    path_clu_k = path+'Clusters_k/'+dataset+'/'+grid_dir+'/'

if '2' in  clust_alg:   
    os.system('mkdir ' + path+' Clusters_2_split/' +dataset+'/'+ grid_dir )
    os.system('mkdir ' + path+'Clusters_2/' +dataset+'/'+ grid_dir )
    path_clu_v2=path+'Clusters_2/' +dataset+'/'+ grid_dir +'/'
                                                                              
if '3' in  clust_alg:   
    os.system('mkdir ' + path+'Clusters_3_split/' +dataset+'/'+ grid_dir )
    os.system('mkdir ' + path+'Clusters_3/' +dataset+'/'+ grid_dir )
    path_clu_v3=path+'Clusters_3/' +dataset+'/'+ grid_dir +'/'


os.system('mkdir ' + path+'Latents/' +dataset+'/'+ grid_dir )
    
path_latents = path+'Latents/'+dataset+'/'+grid_dir+'/'

#epochs_num = train_data.split('_e_')[-1]  

p=opt.t
r=p

print('\n\nModel: ',train_data,'\n\n')


depths=np.load(path_data+'depths.npz')# depths
tnf=np.load(path_data+'result/tnf.npz')# frequencies 

depths=depths['arr_0']# contigs x samples_abundances
tnfs=tnf['arr_0']# contigs x frequencies 


# Reference data
print('\n\nLoading data\n\n')

# reference_new.tsv:  contigname, OTU, ref_contigname,start,end





print('Before masking','rpkm shape = ',depths.shape,'tnf shape = ',tnfs.shape)


# Some model hyperparameters definition

input_len = depths.shape[1]+tnfs.shape[1] # model input size
 
num_samples=int(depths.shape[1]) 


# GPU

cuda = True if torch.cuda.is_available() else False # use GPUs if they are avialable
cuda = False

# Build the model (untrained model)
print('\n\nBuilding the model\n\n')

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



# Initialize encoder, aka get the model parameters from previous training
if Load_model == 'True' :
    print('\n LOADING MODEL\n')
    encoder=Encoder(h_n,latent_dim)
    if cuda:
        encoder.load_state_dict(torch.load(path+'Models/'+origin_dataset+'/'+grid_dir+'/'+'Encoder_'+train_data),strict=False)
    else:
        encoder.load_state_dict(torch.load(path+'Models/'+origin_dataset+'/'+grid_dir+'/'+'Encoder_'+train_data,map_location=torch.device('cpu') ),strict=False)

    encoder.eval()

    if cuda:
        encoder.cuda()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor 


# Encoding function

def get_latents(latent_dim,Dataloader,path_L):
    """ Retrieve the latent representation of the inputs
        
        Inputs: dataloader that provides the batches of tnfs and depths
        Output: latent representation array and mask  """    

    latent_matrix=torch.zeros([depths.shape[0],latent_dim])
    dataloader,mask=Dataloader
    index_i=0
    for i, (depths_in, tnfs_in) in enumerate(dataloader):                 
        if cuda == True:
            depths_in=depths_in.cuda()
            tnfs_in=tnfs_in.cuda()

        latent_sample = encoder(depths_in,tnfs_in) # encode raw input data into 32 gausian
        #print(latent_sample)
        latent_matrix[index_i:index_i+latent_sample.shape[0],:]=latent_sample # store encoding in a matrix
        index_i+= latent_sample.shape[0]

    latent_array=(latent_matrix.detach().numpy()) #transform the matrix into a np array 

    print('\n\n',path_L+train_data,latent_array.shape,'\n\n')
    np.savez(path_L+train_data+'_latents.npz',latent_array)
    np.savez(path_L+train_data+'_mask.npz',mask)
    
    return latent_array,mask


if Load_model == 'True':
    print('\n ENCODING THE DATASET\n')
    print('\n Storing latents in:',path_latents,'\n')
    latents,mask =get_latents(latent_dim,Dataloader=make_dataloader(depths,tnfs,batchsize=opt.batch_size),path_L=path_latents)

#print(' Contigs #= ',latents.shape)

# CLUSTER 
grid_name='grid_'+dataset+'_'+grid_dir.split('_')[-1]
time0=time.time()


path_latents_file= path_latents+train_data+'_latents.npz'

if 'k' in  clust_alg:
    os.system("python3 aae_clustering_kmeans.py -L "+path_latents_file + ' -d  '+dataset)
    path_clu_k_file= path_clu_k+train_data +'_new_clusters.tsv'

if '2'  in clust_alg:
    os.system("python3 aae_clustering_2.py -L "+ path_latents_file+ ' -d  '+dataset  )
    path_clu_v2_file= path_clu_v2+train_data +'_new_clusters.tsv '
if '3' in  clust_alg :
    os.system("python3 aae_clustering_3.py -L "+ path_latents_file + ' -d  '+dataset )

    path_clu_v3_file= path_clu_v3+train_data +'_new_clusters.tsv'

time1=time.time()
clustering_time=str(time1-time0)


os.system(' echo clustering '+clustering_time+ ' >> Logs/'+grid_name+'_log'    )



# BENCHMARKING
if epochs=='yes':
    if 'k' in  clust_alg:
        os.system("python3 aae_benchmark_2.py -C " +path_clu_k_file + ' --Log_name '+ log_results_name+' --clust_alg '+'k' + ' -G  '+ grid  + ' -d  '+dataset + ' --epochs yes')

    if '3' in   clust_alg:
        os.system("python3 aae_benchmark_2.py -C " +path_clu_v3_file + ' --Log_name '+ log_results_name+' --clust_alg '+'3' + ' -G  '+ grid  + ' -d  '+dataset + ' --epochs yes')

    if '2' in  clust_alg :
        os.system("python3 aae_benchmark_2.py -C " +path_clu_v2_file + ' --Log_name '+ log_results_name+' --clust_alg '+'2' + ' -G  '+ grid  + ' -d  '+dataset + ' --epochs yes')


else:

    if 'k' in  clust_alg:
        os.system("python3 aae_benchmark_2.py -C " +path_clu_k_file + ' --Log_name '+ log_results_name+' --clust_alg '+'k' + ' -G  '+ grid  + ' -d  '+dataset  )

    if '3'  in  clust_alg:
        os.system("python3 aae_benchmark_2.py -C " +path_clu_v3_file + ' --Log_name '+ log_results_name+' --clust_alg '+'3' + ' -G  '+ grid  + ' -d  '+dataset )

    if '2' in  clust_alg :
        os.system("python3 aae_benchmark_2.py -C " +path_clu_v2_file +  ' --Log_name '+ log_results_name+' --clust_alg '+'2' + ' -G  '+ grid  + ' -d  '+dataset )


