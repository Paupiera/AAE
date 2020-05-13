from sklearn.cluster import MiniBatchKMeans
import numpy as np
import argparse
import time
import vamb


parser = argparse.ArgumentParser()
parser.add_argument("-L", type=str, help="Latents path")
parser.add_argument("-d", type=str,default='Airways', help="Dataset")


opt = parser.parse_args()


latents_path=opt.L
dataset=opt.d
def clustering_aae_kmeans(latents_path,dataset):
    time_0 = time.time()

    if dataset=='Airways':  
        path_ref='/home/projects/cpr_10006/people/paupie/Data/data/airways/'
        binseparator='C'

    elif dataset == 'Skin':
        path_ref='/home/projects/cpr_10006/people/paupie/Data/data/skin/'
        binseparator='C'
    elif dataset == 'Oral':
        path_ref='/home/projects/cpr_10006/people/paupie/Data/data/oral/'
        binseparator='C'

    elif dataset == 'Gi':
        path_ref='/home/projects/cpr_10006/people/paupie/Data/data/gi/'
        binseparator='C'

    elif dataset == 'Urog':
        path_ref='/home/projects/cpr_10006/people/paupie/Data/data/urog/'
        binseparator='C'

    elif dataset == 'Metahit':
        path_ref='/home/projects/cpr_10006/people/paupie/Data/data/metahit/'
        binseparator='_'
    #contignames = np.genfromtxt(fname=path_ref+'names',  dtype=str) 

    with open(path_ref+'contigs.fna', 'rb') as contigfile:
        tnfs, contignames, contiglengths = vamb.parsecontigs.read_contigs(contigfile)




    clu_path='Clusters_k'+latents_path.split('Latents')[1].split('_latents.npz')[0]+'_new_clusters.tsv'

    grid_dir=clu_path.split(dataset+'/')[1].split('/')[0]


    print('\n Grid dir',grid_dir,'\n')
    

    latents = np.load(latents_path)['arr_0']

    print('\nLatents file:',latents_path,'\n ')

    print('\nClustering algorithm K-mean sklearn MiniBatchKMeans\n')



    kmeans=MiniBatchKMeans(n_clusters=750, random_state=0, batch_size=4096, max_iter=25, init_size=20000, reassignment_ratio=0.02).fit(latents)


    #kmeans.cluster_centers_


        
    Ys = kmeans.predict(latents)

    print(Ys.shape)
    print('\n Clusters file:',clu_path,'\n ')




    with open(clu_path,"w") as record_file:
        for i in range(latents.shape[0]):
            contig_name = contignames[i]
            contig_cluster = Ys[i]+1
            contig_cluster_str = 'cluster_'+str(contig_cluster)
            record_file.write((contig_cluster_str+'\t'+contig_name+'\n'))

    time_1 = time.time()

    clustering_time = time_1 - time_0

    print('\n Clustering time',clustering_time,'\n')




    ### From here I will save the bins and clusters splitted and filtered size
    
    with open(path_ref+'reference.tsv') as reference_file:
        reference = vamb.benchmark.Reference.from_file(reference_file)


    with open(path_ref+'taxonomy.tsv') as taxonomy_file:
        reference.load_tax_file(taxonomy_file)

    # AAE clusters

    with open(clu_path) as clusters_file:
        clusters = vamb.cluster.read_clusters(clusters_file)
    
    def filterclusters(clusters, lengthof):
        filtered_bins = dict()
        for medoid, contigs in clusters.items():
            binsize = sum(lengthof[contig] for contig in contigs)
        
            if binsize >= 200000:
                filtered_bins[medoid] = contigs
        
        return filtered_bins
            
    lengthof = dict(zip(contignames, contiglengths))
    filtered_clusters = filterclusters(vamb.vambtools.binsplit(clusters, binseparator), lengthof)
    print('Number of bins before splitting and filtering:', len(clusters))
    print('Number of bins after splitting and filtering:', len(filtered_clusters))
    
    # Split clusters file 
    clu_path='Clusters_k_split'+latents_path.split('Latents')[1].split('_latents.npz')[0]+'_new_clusters.tsv'

    keptcontigs = set.union(*filtered_clusters.values())
    with open(clu_path, 'w') as file:
        vamb.cluster.write_clusters(file, filtered_clusters)

    with open(path_ref+'contigs.fna', 'rb') as file_1:
        fastadict = vamb.vambtools.loadfasta(file_1, keep=keptcontigs)
    
    bindir='./Clusters_k_split/'+dataset+'/'+grid_dir+'/bins/'+latents_path.split('/')[-1].split('_latents.npz')[0]
    print('\n Bin dir:',bindir,'\n')
    vamb.vambtools.write_bins(bindir, filtered_clusters, fastadict, maxbins=2000)
    print('\n Clusters file:',clu_path,'\n ')


                                                                                                                                                                       

    

clustering_aae_kmeans(latents_path,dataset)





