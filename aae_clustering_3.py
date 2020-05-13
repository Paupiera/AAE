
import numpy as np
import itertools
import gzip


import torch


#import vamb_3.vambtools as _vambtools
import vamb

from collections import Counter

from data_loader import make_dataloader # aae/data_loader.py

import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("-L", type=str, help="Latents path")
parser.add_argument("-d", type=str,default='Airways', help="Dataset")

opt = parser.parse_args()
args = vars(parser.parse_args())
print(args)

latents_path=opt.L
dataset=opt.d

Cuda = True if torch.cuda.is_available() else False
print('\n Cuda,',Cuda,'\n')

def clustering_aae_3(latents_path,dataset):

    time_0=time.time()
    #CLUSTER 
    Cuda = True if torch.cuda.is_available() else False
    
    print('\nClustering algorithm VAMB-3\n')




    print('\n\nCLUSTERING '+ latents_path +' \n\n')
    if dataset=='Airways':
        path_ref='/home/projects/cpr_10006/people/paupie/Data/data/airways/'
    
    elif dataset == 'Skin':
        path_ref='/home/projects/cpr_10006/people/paupie/Data/data/skin/'

    elif dataset == 'Oral':
        path_ref='/home/projects/cpr_10006/people/paupie/Data/data/oral/'


    elif dataset == 'Gi':
        path_ref='/home/projects/cpr_10006/people/paupie/Data/data/gi/'


    elif dataset == 'Urog':
        path_ref='/home/projects/cpr_10006/people/paupie/Data/data/urog/'


    elif dataset == 'Metahit':
        path_ref='/home/projects/cpr_10006/people/paupie/Data/data/metahit/'
    
    with open(path_ref+'contigs.fna', 'rb') as contigfile:
        tnfs, contignames, contiglengths = vamb.parsecontigs.read_contigs(contigfile)


    mask_path = latents_path.split('latents.npz')[0]+'mask.npz'
    
    path_clu = 'Clusters_3'+latents_path.split('Latents')[1].split('latents.npz')[0]+'new_clusters.tsv'
    
    print('\n\n Latents Path: ',latents_path,'\n Mask path: ',mask_path,'\n Clusters path: ',path_clu,'\n')
    # Load aae latent representation
    latent = vamb.vambtools.read_npz(latents_path)

    # Load aae the mask
    mask = vamb.vambtools.read_npz(mask_path)
    
    # Vamb latents and masks

    #latent = vamb.vambtools.read_npz('/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_airways/result/latent.npz')
    # Load the mask
    #mask = vamb.vambtools.read_npz('/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_airways/result/mask.npz')


    #print('Latent shape=',latent.shape,'mask shape=',mask.shape)
    # Notice we mask the contignames, since the dataloader could havie filtered some contigs away

    cluster_iterator = vamb.cluster.cluster(latent,cuda=Cuda)#, labels=np.array(contignames)[mask],cuda=Cuda) # Iterative medoid cluster generator, The generator will compute the clusters on-the-fly, meaning it will only compute the next cluster once you ask for it.
    clusters = dict((c.as_tuple(labels=np.array(contignames)[mask])) for c in cluster_iterator)
    
    medoid, contigs = next(iter(clusters.items()))
    print('First key:', medoid, '(of type:', type(medoid), ')')
    print('Type of values:', type(contigs))
    print('First element of value:', next(iter(contigs)), 'of type:', type(next(iter(contigs))))

    # This writes a .tsv file with the clusters and corresponding sequences
    with open(path_clu, 'w+') as file:
        vamb.cluster.write_clusters(file,clusters)
    time_1=time.time()
    clustering_time=time_1-time_0
    print('\n\n Clusters file located in:',path_clu,'\n\n')
    print('\n Clustering time=',clustering_time,'\n')
    


    ### From here I will save the bins and clusters splitted and filtered size

    with open(path_ref+'reference.tsv') as reference_file:
        reference = vamb.benchmark.Reference.from_file(reference_file)


    with open(path_ref+'taxonomy.tsv') as taxonomy_file:
        reference.load_tax_file(taxonomy_file)

    # AAE clusters


    def filterclusters(clusters, lengthof):
        filtered_bins = dict()
        for medoid, contigs in clusters.items():
            binsize = sum(lengthof[contig] for contig in contigs)

            if binsize >= 200000:
                filtered_bins[medoid] = contigs

        return filtered_bins

    lengthof = dict(zip(contignames, contiglengths))
    filtered_clusters = filterclusters(vamb.vambtools.binsplit(clusters, 'C'), lengthof)
    print('Number of bins before splitting and filtering:', len(clusters))
    print('Number of bins after splitting and filtering:', len(filtered_clusters))

    # Split clusters file 
    path_clu='Clusters_3_split'+latents_path.split('Latents')[1].split('_latents.npz')[0]+'_S_new_clusters.tsv'

    keptcontigs = set.union(*filtered_clusters.values())
    with open(path_clu, 'w') as file:
        vamb.cluster.write_clusters(file, filtered_clusters)

    with open(path_ref+'contigs.fna', 'rb') as file_1:
        fastadict = vamb.vambtools.loadfasta(file_1, keep=keptcontigs)

    bindir='./Clusters_3_split/Airways/bins/'+latents_path.split('/')[-1].split('_latents.npz')[0]
    print('\n Bin dir:',bindir,'\n')
    vamb.vambtools.write_bins(bindir, filtered_clusters, fastadict, maxbins=2000)
    print('\n Clusters file:',path_clu,'\n ')



clustering_aae_3(latents_path,dataset)




