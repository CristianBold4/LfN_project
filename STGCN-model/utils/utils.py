import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch


def compute_gso(adj, gso_type):
    n_sensors = adj.shape[0]

    if not sp.issparse(adj):
        adj = sp.csc_matrix(adj)
    elif adj.format != 'csc':
        # Compressed Sparse Column format matrix
        adj = adj.tocsc()

    I = sp.identity(n_sensors, format='csc')

    # Symmetrizing adj
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
        adj = adj + I

    if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_adj':
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        # A_sym = D^{-0.5} * A * D^{-0.5}
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type == 'sym_norm_lap':
            sym_norm_lap = I - sym_norm_adj
            gso = sym_norm_lap
        else:
            gso = sym_norm_adj

    elif gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_adj':
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        # A_rw = D^{-1} * A
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type == 'rw_norm_lap':
            rw_norm_lap = I - rw_norm_adj
            gso = rw_norm_lap
        else:
            gso = rw_norm_adj
    else:
        raise ValueError(f'{gso_type} is not defined.')

    return gso


def compute_cheby_gso(gso):
    if not sp.issparse(gso):
        gso = sp.csc_matrix(gso)
    elif gso.format != 'csc':
        gso = gso.tocsc()

    I = sp.identity(gso.shape[0], format='csc')
    max_eigenvalue = max(eigsh(A=gso, k=6, which='LM', return_eigenvectors=False))

    # if the gso is symmetric or rw normalized Laplacian, then the max eigenvalue <= 2
    if max_eigenvalue <= 2:
        gso = gso - I
    else:
        gso = 2 * gso/max_eigenvalue - I


    return gso


