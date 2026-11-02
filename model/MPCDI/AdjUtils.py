import numpy as np
import torch
import scipy.sparse as sp
def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def compute_support_gwn(adj, device=None):
    adj_mx = [asym_adj(adj), asym_adj(np.transpose(adj))]
    support = [torch.tensor(i).to(device) for i in adj_mx]
    return support