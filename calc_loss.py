from math import log

def calc_loss(depths_in, depths_out, tnf_in, tnf_out,num_samples,alpha=0.15):
# If multiple samples, use cross entropy, else use SSE for abundance 
    if num_samples > 1:
        # Add 1e-9 to depths_out to avoid numerical instability.
        ce = - ((depths_out + 1e-9).log() * depths_in).sum(dim=1).mean()
        ce_weight = (1-alpha) / log(num_samples)
        #print('CE=',ce*ce_weight)
    else:
        ce = (depths_out - depths_in).pow(2).sum(dim=1).mean()
        ce_weight = 1 - alpha
        #print('only 1 sample')
    #print(tnf_out,tnf_in)
    sse = (tnf_out - tnf_in).pow(2).sum(dim=1).mean()
    sse_weight =alpha/(136*2)
    #print('SEE=',sse*sse_weight)
    loss = ce * ce_weight + sse * sse_weight 
    #print(ce*ce_weight,sse*sse_weight)
    return loss, ce, sse


