# Encoding function
import torch
import numpy as np

def get_latents(Dataloader):
    """ Retrieve the latent representation of the inputs
        
        Inputs: dataloader that provides the batches of tnfs and depths
        Output: latent representation array and mask  """

    latent_matrix=torch.zeros([depths.shape[0]-(depths.shape[0]%opt.batch_size),32])
    dataloader,mask=Dataloader
    for i, (depths_in, tnfs_in) in enumerate(dataloader):
        if cuda == True:
            depths_in=depths_in.cuda()
            tnfs_in=tnfs_in.cuda()

        latent_sample = encoder(depths_in,tnfs_in) # encode raw input data into 32 gausian

        latent_matrix[i*latent_sample.shape[0]:(i+1)*latent_sample.shape[0],:]=latent_sample # store encoding in a matrix

    latent_array=(latent_matrix.detach().numpy()) #transform the matrix into a np array
    np.savez(path+'Latents/Airways/'+train_data+'_latents'+'.npz',latent_array)
    np.savez(path+'Latents/Airways/'+train_data+'_mask'+'.npz',mask)

    return latent_array,mask


