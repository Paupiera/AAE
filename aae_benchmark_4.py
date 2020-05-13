''' Given the model, generates a 3x3x3 array where 1-d (row) is precision (0.9,0.95,0.99), 2-d is recall (0.9,0.95,0.99) and 3d is tax rank (OTU,Species,Genus) '''

import vamb
import argparse 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vamb.vambtools as _vambtools
import numpy as np

# Flags 
parser = argparse.ArgumentParser()
parser.add_argument("-C", type=str,default='',  help="Clusters path")
parser.add_argument("--Vamb", type=str,default='',  help="running over Vamb")
parser.add_argument("--DAS", type=str,default='',  help="running over DAS bin combination")



parser.add_argument("-d", type=str, help="Dataset")

args = vars(parser.parse_args())
print(args)

opt = parser.parse_args()

path_clu=opt.C
dataset=opt.d
Vamb=opt.Vamb
DAS=opt.DAS
def benchmark_4(path_clu,dataset):
    path='/home/projects/cpr_10006/people/paupie/aae/'
    
    name=path_clu.split('/')[-1].split('_clusters.tsv')[0]
    if DAS:
        grid_dir ='DAS'
        clust_alg='DAS'
    elif Vamb:
        pass
    else:
        grid_dir=name.split('_sl')[0].split(dataset+'_')[1]
        clust_alg=path_clu.split('Clusters_')[1].split('/')[0]

    
    path_plt=path+'/Plots/'



    if dataset == 'Airways':

        taxonomy_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/airways/taxonomy.tsv'
        path_clu_vamb = '/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_airways/result/clusters.tsv'
        reference_path = '/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_airways/jakni/reference_new.tsv'  

    elif dataset == 'Skin':
        taxonomy_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/skin/taxonomy.tsv'
        path_clu_vamb = '/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_skin/result/clusters.tsv'

        reference_path = '/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_skin/reference.tsv'

    elif dataset == 'Oral':

        taxonomy_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/oral/taxonomy.tsv'
        path_clu_vamb = '/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_oral/result/clusters.tsv'

        reference_path = '/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_oral/reference.tsv'

    elif dataset == 'Gi':
        taxonomy_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/gi/taxonomy.tsv'

        reference_path = '/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_gi/reference.tsv'
    elif dataset == 'Urog':
        taxonomy_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/urog/taxonomy.tsv'
        reference_path = '/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_urog/reference.tsv'

    elif dataset == 'Metahit':
        taxonomy_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/metahit/taxonomy.tsv'

        reference_path = '/home/projects/cpr_10006/projects/vamb/data/datasets/cami_high/reference.tsv'



    with open(reference_path) as reference_file:
        reference = vamb.benchmark.Reference.from_file(reference_file)


    with open(taxonomy_path) as taxonomy_file:
        reference.load_tax_file(taxonomy_file)

    if Vamb:
    # Vamb clusters
        print('\n VAMB CLUSTERS\n')
        with open(path_clu_vamb) as clusters_file:
            vamb_clusters = vamb.cluster.read_clusters(clusters_file)

            vamb_bins = vamb.benchmark.Binning(vamb_clusters, reference, minsize=200000,binsplit_separator='C')
            _bins = vamb_bins

    else:
    # AAE clusters
        print('\n AAE ',name,'CLUSTERS \n')
        with open(path_clu) as clusters_file:
         
            aae_clusters = vamb.cluster.read_clusters(clusters_file)

            aae_bins = vamb.benchmark.Binning(aae_clusters, reference, minsize=200000,binsplit_separator='C')
            _bins = aae_bins
   
    taxonomic_rank=['OTU','Species','Genus']
    #%matplotlib inline
 
    scores_array = np.zeros((3,3,3))

    for t in range(3):
        print('Tax rank =',taxonomic_rank[t])

        for p,i in zip([0.9,0.95,0.99],range(3)):
            for r,j in zip([0.9,0.95,0.99],range(3)):
                
                n_bins = _bins.counters[t][(r,p)]

                print('Precision=',p,' Recall=',r,' bins=',n_bins)
                scores_array[t,i,j]= n_bins
    if Vamb:
        np.savez('../Data_plots/'+dataset+'/Benchmarking_4_PRT_Vamb.npz',scores_array)
        print('\nClu file',path_clu_vamb,'\n')
        print('\nArray file','../Data_plots/'+dataset+'/Benchmarking_4_PRT_Vamb.npz\n') 
    elif DAS:
        np.savez('../Data_plots/'+dataset+'/'+grid_dir+'/Benchmarking_4_PRT_'+name+'.npz',scores_array)
        print('\nClu file',path_clu_vamb,'\n')
        print('\nArray file','../Data_plots/'+dataset+'/'+grid_dir+'/Benchmarking_4_PRT_'+name+'.npz\n') 

    else:
        np.savez('../Data_plots/'+dataset+'/'+grid_dir+'/Benchmarking_4_PRT_clust_alg'+clust_alg+'_'+name+'.npz',scores_array)
        print('\nClu file',path_clu,'\n')
        print('\nArray file','../Data_plots/'+dataset+'/'+grid_dir+'/Benchmarking_4_PRT_clust_alg'+clust_alg+'_'+name+'.npz\n') 
    print(scores_array)
   
benchmark_4(path_clu,dataset)   
