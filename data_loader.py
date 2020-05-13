import numpy as np  
import torch

from torch.utils.data import DataLoader
import vamb.vambtools as _vambtools
from torch.utils.data.dataset import TensorDataset as TensorDataset  

### DATA LOADER
# Configure data loader
def make_dataloader(rpkm, tnf, batchsize, destroy=False, Cuda=False,Shuffle=False,Drop_last=True):

    """Create a DataLoader and a contig mask from RPKM and TNF.

    The dataloader is an object feeding minibatches of contigs to the model.
    The data are normalized versions of the input datasets, with zero-contigs,
    i.e. contigs where a row in either TNF or RPKM are all zeros, removed.
    The mask is a boolean mask designating which contigs have been kept.

    Inputs:
        rpkm: RPKM matrix (N_contigs x N_samples)
        tnf: TNF matrix (N_contigs x 136)
        batchsize: Starting size of minibatches for dataloader
        destroy: Mutate rpkm and tnf array in-place instead of making a copy. ????#Save memory by destroying matrix while clustering
        Cuda: Pagelock memory of dataloader (use when using GPU acceleration)

    Outputs:
        DataLoader: An object feeding data to the model
        mask: A boolean mask of which contigs are kept
    """

    if not isinstance(rpkm, np.ndarray) or not isinstance(tnf, np.ndarray):
        raise ValueError('TNF and RPKM must be Numpy arrays')

    if batchsize < 1:
        raise ValueError('Minimum batchsize of 1, not {}'.format(batchsize))

    if len(rpkm) != len(tnf): ## aren't len(rpkm)= (N_contigs * N_samples) and len(tnf)= (N_contigs * 136) ??
        raise ValueError('Lengths of RPKM and TNF must be the same')

    if tnf.shape[1] != 136: # primary k-mers
        raise ValueError('TNF must be length 136 long along axis 1')

    if not (rpkm.dtype == tnf.dtype == np.float32):
        raise ValueError('TNF and RPKM must be Numpy arrays of dtype float32')

    mask = tnf.sum(axis=1) != 0 # each tnf column j is a k-mer frequency, and each row is the i-th contig
                                # mask.shape = N_contigs,1, mask=[True,True,....,True], 

    # If multiple samples, also include nonzero depth as requirement for accept
    # of sequences
    if rpkm.shape[1] > 1: # if there is more than 1 sample
        depthssum = rpkm.sum(axis=1) # sum of contig coabundances, each column is a Sample_abundance and each row is a contig, 
                                     # depthssum.shape = N_contigs,1
        mask &= depthssum != 0 # mask False contigs with sum_freqs =0 and abundance_samples=0
        depthssum = depthssum[mask]

    if mask.sum() < batchsize: #
        raise ValueError('Fewer sequences left after filtering than the batch size.')

    if destroy: #Save memory by destroying matrix while clustering
        rpkm = _vambtools.numpy_inplace_maskarray(rpkm, mask)
        tnf = _vambtools.numpy_inplace_maskarray(tnf, mask)
    else:
        # The astype operation does not copy due to "copy=False", but the masking
        # operation does.
        rpkm = rpkm[mask].astype(np.float32, copy=False)
        tnf = tnf[mask].astype(np.float32, copy=False)

    # If multiple samples, normalize to sum to 1, else zscore normalize
    if rpkm.shape[1] > 1: # if there is more than 1 sample
        rpkm /= depthssum.reshape((-1, 1)) # dvide each contig abundance by the sum of aboundances of this contig
    else:
        _vambtools.zscore(rpkm, axis=0, inplace=True) # normalize rpkm columns (aka Sample aboundances) to standard normal (0,1)

## normalize rows (contigs) to sum to 1 vs zscore columns (k-mer or Sample) normalize 


    # Normalize TNF arrays and create the Tensors (the tensors share the underlying memory)
    # if the Numpy arrays
    _vambtools.zscore(tnf, axis=0, inplace=True)# normalize tnf rows (aka Contig_i 4-mer_j frequencies) to standard normal (0,1)

    print('After masking','rpkm shape = ',rpkm.shape,'tnf shape = ',tnf.shape,' any contig with all 0 ?   ',any(mask==False))


    depthstensor = torch.from_numpy(rpkm)

    tnftensor = torch.from_numpy(tnf)

    # Create dataloader
    dataset = TensorDataset(depthstensor, tnftensor) # Dataset wrapping tensors.

    dataloader = DataLoader(dataset=dataset,   # dataset object to load data from
                             batch_size=batchsize,
                             drop_last=Drop_last,#drops the last non-full batch of each worker’s dataset replica
                             shuffle=Shuffle,
                             num_workers=0,# multi-process data loading with the specified number of loader worker processes. I set it to 0 to avoid [Errno 104]
                             pin_memory=Cuda)

    return dataloader, mask


