import vamb
import argparse 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vamb.vambtools as _vambtools
import time
import os


parser = argparse.ArgumentParser()
parser.add_argument("-C", type=str, help="path_clusters")
parser.add_argument("-d", type=str, help="Dataset")


args = vars(parser.parse_args())
print(args)
opt = parser.parse_args()

path_clusters= opt.C
dataset=opt.d


def benchmark(path_clu,dataset):
    path='/home/projects/cpr_10006/people/paupie/aae/'
    #clust_alg=path_clu.split('Clusters_')[1].split('/')[0]
    clust_alg=''
    print('\nCLUSTERING ALGORITHM',clust_alg,'\n')
    name=path_clu.split('/')[-1].split('_clusters.tsv')[0]
    #grid_dir=name.split('_s_l')[0].split(dataset+'_')[1]
    grid_dir='./'
    path_plt=path+'Plots/'+dataset+'/'+grid_dir+'/'
    os.system('mkdir -p '+path_plt)    
    print('\n PLOT PATH: ',path_plt,'\n',name,'\n',grid_dir,'\n')
    
    if dataset == 'Airways':
        taxonomy_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/airways/taxonomy.tsv'

        reference_path = '/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_airways/jakni/reference_new.tsv'
        path_clu_vamb='/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_airways/result/clusters.tsv'
    elif dataset == 'Skin':
        taxonomy_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/skin/taxonomy.tsv'

        reference_path = '/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_skin/reference.tsv'
        path_clu_vamb='/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_skin/result/clusters.tsv'

    elif dataset == 'Oral':

        taxonomy_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/oral/taxonomy.tsv'

        reference_path = '/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_oral/reference.tsv'
        path_clu_vamb='/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_oral/result/clusters.tsv'

    elif dataset == 'Gi':
        taxonomy_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/gi/taxonomy.tsv'

        reference_path = '/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_gi/reference.tsv'
        path_clu_vamb='/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_gi/result/clusters.tsv'

    elif dataset == 'Urog':
        taxonomy_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/urog/taxonomy.tsv'
        reference_path = '/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_urog/reference.tsv'
        path_clu_vamb='/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_urog/result/clusters.tsv'

    elif dataset == 'Metahit':
        taxonomy_path = '/home/projects/cpr_10006/people/sira/share/vamb_codeocean2/data/metahit/taxonomy.tsv'

        reference_path = '/home/projects/cpr_10006/projects/vamb/data/datasets/cami_high/reference.tsv'
        path_clu_vamb='/home/projects/cpr_10006/projects/vamb/paper_revised/vamb_on_metahit/result/clusters.tsv'



    with open(reference_path) as reference_file:
        reference = vamb.benchmark.Reference.from_file(reference_file)


    with open(taxonomy_path) as taxonomy_file:
        reference.load_tax_file(taxonomy_file)

# AAE clusters
      
    with open(path_clu) as clusters_file:
        aae_clusters = vamb.cluster.read_clusters(clusters_file)
        #aae_clusters = vamb.vambtools.read_clusters(clusters_file) # jakob way

        aae_bins = vamb.benchmark.Binning(aae_clusters, reference, minsize=200000,binsplit_separator='C')

# METABAT2 clusters

#    with open('/home/projects/cpr_10006/projects/vamb/paper/metabat_on_cami2/clusters.tsv') as clusters_file:
#        metabat_clusters = vamb.cluster.read_clusters(clusters_file)
#        metabat_bins = vamb.benchmark.Binning(metabat_clusters, reference, minsize=200000,binsplit_separator='C')

# vamb clusters
    with open(path_clu_vamb) as clusters_file:
        vamb_clusters = vamb.cluster.read_clusters(clusters_file)
        vamb_bins = vamb.benchmark.Binning(vamb_clusters, reference, minsize=200000,binsplit_separator='C')

    print('Vamb bins:')
    for rank in vamb_bins.summary():
        print('\t'.join(map(str, rank)))

#    print('\nMETABAT2 bins:')
#    for rank in metabat_bins.summary():
#        print('\t'.join(map(str, rank)))

    print('\nAAE bins:')
    for rank in aae_bins.summary():
        print('\t'.join(map(str, rank)))

    #%matplotlib inline
    taxonomic_rank=['OTU','Species','Genus']
    #names=['Vamb','METABAT2',name]
    for precision in 0.99,0.95, 0.9:
        print(precision)
        for t in range(len(taxonomic_rank)):            
            tax_rank=taxonomic_rank[t]

            plt.subplot(3,1,t+1)

            #colors = ['#DDDDDD', '#AAAAAA', '#777777', '#444444', '#000000']
            colors = ['#DDDDDD', '#AAAAAA','#999999','#777777', '#555555', '#222222', '#000000']
            recalls = [0.5, 0.6, 0.7, 0.8, 0.9,0.95,0.99]
 
            for y,bins in zip((0,1),(vamb_bins,aae_bins)):
     
            #for y,bins in zip((0,1,2),(vamb_bins,metabat_bins,aae_bins)):
                for color, recall in zip(colors, recalls):
                    plt.barh(y, bins.counters[t][(recall, precision)], color=color)# [1] for the 2nd taxonomic rank,(strain, species, genus)

            plt.title('Prec= '+str(precision)+',Taxonomy rank: '+tax_rank, fontsize=12)
            plt.yticks([0,1,2], ['Vamb','AAE'], fontsize=12)

#            plt.yticks([0,1,2], ['Vamb','METABAT2','AAE'], fontsize=12)
            plt.xticks([i*25 for i in range(5)], fontsize=8)
            plt.legend([str(i) for i in recalls],bbox_to_anchor=(1.2, 1.1) , title='Recall', fontsize=8) #            
            if precision == 0.9:
                plt.xlabel('# of Genomes Identified', fontsize=10)
            plt.gca().set_axisbelow(True)
            plt.grid()
            #plt.savefig(path_plt+'Precision_'+str(precision)+'_'+tax_rank+'_'+name+'.png')
        plt.tight_layout()    
        plt.savefig(path_plt+name+'_Precision_'+str(precision)+'clust_alg'+clust_alg+'.png',dpi=96*2) 
        plt.close()
        

benchmark(path_clusters,dataset)
