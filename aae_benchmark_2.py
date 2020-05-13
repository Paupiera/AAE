import vamb
import vamb.vambtools as _vambtools
import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("-C", type=str, help="path_clusters")
parser.add_argument("-p", type=float,default=0.9, help=" precision thresholds")
parser.add_argument("-r", type=float,default=0.9, help=" recall thresholds")
parser.add_argument("-G", type=str,default='', help=" Are the clusters part of a grid search?  ")
parser.add_argument("-d", type=str, help="Dataset")
parser.add_argument("-S", type=str,default='', help="Slope")
parser.add_argument("--clust_alg", type=str,default='', help="clustering algorithm used ")



parser.add_argument("--Log_name", type=str,default='', help="Log_results_name")
parser.add_argument("--epochs", type=str ,default='', help="Log_results_name")


args = vars(parser.parse_args())
print(args)
opt = parser.parse_args()
output='Yes' # no echo into file
Grid=opt.G
path_clusters= opt.C
dataset=opt.d
slope=opt.S
epochs=opt.epochs
log_results_name=opt.Log_name
clust_alg=opt.clust_alg


def benchmark_2(path_clusters,dataset,precision=0.9,recall=0.9):
    path='/home/projects/cpr_10006/people/paupie/aae/'
    
    path_clu= path_clusters
    if dataset == 'Airways':
        taxonomy_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/airways/taxonomy.tsv'
        binseparator='C'
        reference_path = '/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_airways/jakni/reference_new.tsv'  
    elif dataset == 'Skin':
        taxonomy_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/skin/taxonomy.tsv'
        binseparator='C'
        reference_path = '/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_skin/reference.tsv'  
    elif dataset == 'Oral':

        taxonomy_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/oral/taxonomy.tsv'
        binseparator='C'
        reference_path = '/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_oral/reference.tsv'  
    elif dataset == 'Gi':
        taxonomy_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/gi/taxonomy.tsv'

        reference_path = '/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_gi/reference.tsv'  
        binseparator='C'
    elif dataset == 'Urog':
        taxonomy_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/urog/taxonomy.tsv'

        reference_path =  '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/urog/reference.tsv'  
        binseparator='C'
    elif dataset == 'Metahit':
        taxonomy_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/metahit/taxonomy.tsv'

        reference_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/metahit/reference.tsv'  
        binseparator='_'


    with open(reference_path) as reference_file:
        reference = vamb.benchmark.Reference.from_file(reference_file)

    
    with open(taxonomy_path) as taxonomy_file:
        reference.load_tax_file(taxonomy_file)

# AAE clusters
    print('\n\n\n CLUSTERS FILE:    ', path_clu,'\n\n\n')
    with open(path_clu) as clusters_file:
        aae_clusters = vamb.cluster.read_clusters(clusters_file)

        aae_bins = vamb.benchmark.Binning(aae_clusters, reference, minsize=200000,binsplit_separator=binseparator)

    
    scores_list=[]
    taxonomic_rank=['OTU','Species','Genus']

    for t in range(len(taxonomic_rank)):            
        tax_rank=taxonomic_rank[t]
        scores_list.append(aae_bins.counters[t][(recall, precision)])
        print('recall=',recall,' precision=',precision,' taxonomik rank=',tax_rank)

        print(aae_bins.counters[t][(recall, precision)])# [1] for the 2nd taxonomic rank,(strain, species, genus)
    print('\n\nTrain_data=',path_clusters,'\n\n')
    print(scores_list)
    
    

    return scores_list
p = opt.p
r = opt.r

scores_list=benchmark_2(path_clusters,dataset,precision=p,recall=r)

if Grid:
    ## get scale loss from clusters path

    try:
        sl = path_clusters.split('s_l_')[1].split('_hn_')[0]
    except:
        
        try:
            sl = path_clusters.split('sl_')[1].split('_hn_')[0]
        except:
            pass

    ## get hidden neurons number from clusters path
    h_n = path_clusters.split('hn_')[1].split('_l_d')[0]
    

    ## get latent dim from clusters path
    try:
        latent_dim = path_clusters.split('_l_d_')[1].split('_')[0]
    except:
        
        try:
            latent_dim = path_clusters.split('_l_d_')[1].split('_e_')[0]
        except:
            pass
    
    ## get batch size from cluster path
    try:
        batch_size = path_clusters.split('_bs_')[1].split('_')[0]
    except:
        pass

    log_results_name = opt.Log_name
    try:
        epoch=path_clusters.split('_e_')[1].split('_')[0]
        
    except:
        log_type='no_epoch'
    path='/home/projects/cpr_10006/people/paupie/aae/'

    if slope=='yes':
        S=path_clusters.split('S_')[1].split('_e_')[0]
        
        os.system('echo -e '+str(scores_list[0]) +'  '+str(scores_list[1])+' '+str(scores_list[2]) +' '+str(S)+' '+str(epoch) +' '+clust_alg+ ' >> '+path +'Logs/'+ log_results_name + '_' + str(p))



    elif epochs=='yes':
        os.system('echo -e '+str(scores_list[0]) +'  '+str(scores_list[1])+' '+str(scores_list[2]) +' '+str(epoch) +' '+clust_alg+ ' >> '+path +'Logs/'+ log_results_name + '_' + str(p))
        
    #elif log_type =='no_epoch':
        #os.system('echo -e '+str(scores_list[0]) +'  '+str(scores_list[1])+' '+str(scores_list[2]) +' '+str(sl)+' '+str(h_n)+' '+str(latent_dim) +' '+clust_alg+' >> '+path +'Logs/'+ log_results_name + '_' + str(p))

    elif output == None:
        print('NO OUTPUT')

    else:
        #batch_size=64
        os.system('echo -e '+str(scores_list[0]) +'  '+str(scores_list[1])+' '+str(scores_list[2]) +' '+str(sl)+' '+str(h_n)+' '+str(latent_dim) +' '+str(batch_size)+' '+clust_alg+' >> '+path +'Logs/'+ log_results_name + '_' + str(p))




