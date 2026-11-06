from typing import Optional, Union, Tuple, List 

import numpy as np 
import torch 
from torch import Tensor 
from torch_geometric.utils import scatter 
from torch_geometric.utils.num_nodes import maybe_num_nodes

epsilon = 5e-8

def expand_then_cat(tensors:Union[Tuple[Tensor,...],List[Tensor]], dim=-1) -> Tensor:
    shapes=[t.shape for t in tensors]
    expand_dims=list(np.max(shapes,0))
    expand_dims[dim]=-1
    tensors=[t.expand(*expand_dims) for t in tensors]
    return torch.cat(tensors,dim=dim)

def sparse_softmax(src:Tensor, index:Optional[Tensor]=None,ptr:Optional[Tensor]=None,num_nodes:Optional[int]=None,dim:int=-2) -> Tensor:
    if ptr is not None or index is None:
        raise NotImplementedError
    N=maybe_num_nodes(index,num_nodes)
    src_max=scatter(src,index,dim,dim_size=N,reduce="max")
    src_max=src_max.index_select(dim,index)
    src_max=src_max.masked_fill(torch.isinf(src_max),0)
    out=(src-src_max).exp()
    out_sum=scatter(out,index,dim,dim_size=N,reduce="sum")
    out_sum=out_sum.index_select(dim,index)
    return out/(out_sum+epsilon)
