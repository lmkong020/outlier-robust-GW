import random
import copy
from loaddata import load
import numpy as np
import torch
import itertools
import torch.nn.functional as F
import time
import argparse
import scipy
import dgl
import pickle
from dgl.nn.pytorch import GINConv, GraphConv
from sklearn.decomposition import PCA
seed = 1
torch.manual_seed(seed)
random.seed(seed)


def my_check_align(pred, ground_truth, result_file=None):
    g_map = {}
    for i in range(ground_truth.size(1)):
        g_map[ground_truth[1, i].item()] = ground_truth[0, i].item()
    g_list = list(g_map.keys())
    ind = (-pred).argsort(axis=1)[:, :30]
    a1, a5, a10, a30 = 0, 0, 0, 0
    for i, node in enumerate(g_list):
        for j in range(30):
            if j >= pred.shape[1]:
                break
            if ind[node, j].item() == g_map[node]:
                if j < 1:
                    a1 += 1
                if j < 5:
                    a5 += 1
                if j < 10:
                    a10 += 1
                if j < 30:
                    a30 += 1
    a1 /= len(g_list)
    a5 /= len(g_list)
    a10 /= len(g_list)
    a30 /= len(g_list)
    # print('H@1 %.2f%% H@5 %.2f%% H@10 %.2f%% H@30 %.2f%%' % (a1 * 100, a5 * 100, a10*100, a30*100))
    return a1,a5,a10,a30


def my_check_align1(pred, ground_truth):
    g_map = {}
    for i in range(ground_truth.size(1)):
        g_map[ground_truth[1, i].item()] = ground_truth[0, i].item()
    g_list = list(g_map.keys())
    ind = (-pred).argsort(axis=1)[:, :10]
    a1, a5, a10 = 0, 0, 0
    for i, node in enumerate(g_list):
        for j in range(10):
            if ind[node, j].item() == g_map[node]:
                if j < 1:
                    a1 += 1
                if j < 5:
                    a5 += 1
                if j < 10:
                    a10 += 1
    a1 /= len(g_list)
    a5 /= len(g_list)
    a10 /= len(g_list)
    print('H@1 %.2f%% H@5 %.2f%% H@10 %.2f%%' % (a1 * 100, a5 * 100, a10*100))
    return a1,a5,a10


def cosine_similarity(Afeat, Bfeat):
    Adim, Bdim = Afeat.shape[0], Bfeat.shape[0]
    cossim = torch.zeros(Bdim, Adim, device=Afeat.device)
    for i in range(Bdim):
        cossim[i] = F.cosine_similarity(Afeat, Bfeat[i:i + 1].expand(Adim, Bfeat.shape[1]), dim=-1).view(-1)
    return cossim


def myload(dataset_name='douban', edge_noise=0.):
    print('dataset: {}, edge_noise: {}'.format(dataset_name, edge_noise))

    if dataset_name in ['douban']:
        a1, f1, a2, f2, ground_truth, prior = load(dataset_name)
        Aedge, Bedge = a1.nonzero(), a2.nonzero()
        Afeat, Bfeat = np.array(f1.todense()), np.array(f2.todense())
        ground_truth = torch.tensor(np.array(ground_truth, dtype=int)) - 1  # Original index start from 1
    elif dataset_name == 'dblp':
        f = np.load('data/ACM-DBLP_0.2.npz')
        Afeat, Bfeat = f['x1'].astype('float32'), f['x2'].astype('float32')
        Aedge = (f['edge_index1'][0], f['edge_index1'][1])
        Bedge = (f['edge_index2'][0], f['edge_index2'][1])
        ground_truth = torch.tensor(np.concatenate([f['pos_pairs'], f['test_pairs']],0).astype('int32')).T
    elif dataset_name == 'cora':
        dataset = dgl.data.CoraGraphDataset()
    elif dataset_name == 'citeseer':
        dataset = dgl.data.CiteseerGraphDataset()
    elif dataset_name == 'ppi':
        graph = dgl.data.PPIDataset(mode='train')[0]
        Afeat = graph.ndata['feat']
        Bfeat = Afeat.clone().detach()
        Aedge = graph.edges()
        Bedge = graph.edges()
        ground_truth = torch.cat([graph.nodes().unsqueeze(0), graph.nodes().unsqueeze(0)], 0)
    elif dataset_name == 'facebook':
        from torch_geometric import datasets
        graph = datasets.AttributedGraphDataset('.', 'Facebook')[0]
        Afeat = graph.x
        Bfeat = Afeat.clone().detach()
        Aedge = graph.edge_index
        Bedge = graph.edge_index
        ground_truth = torch.cat([torch.tensor(list(range(Afeat.shape[0]))).unsqueeze(0),
                                  torch.tensor(list(range(Afeat.shape[0]))).unsqueeze(0)], 0)
    elif dataset_name == 'dbp':
        data = dgl.load_graphs('kg/dbp_ja_en_graph_new')[0]
        Agraph, Bgraph = data[0], data[1]
        Afeat = Agraph.ndata['featBSEraw']
        Bfeat = Bgraph.ndata['featBSEraw']
        Aedge = Agraph.edges()
        Bedge = Bgraph.edges()
        ground_truth = torch.cat([torch.tensor(list(range(Afeat.shape[0]))).unsqueeze(0),
                                  torch.tensor(list(range(Afeat.shape[0]))).unsqueeze(0)], 0)
    if dataset_name in ['cora', 'citeseer']:
        graph = dataset[0]
        Afeat = graph.ndata['feat']
        Bfeat = Afeat.clone().detach()
        Aedge = graph.edges()
        Bedge = graph.edges()
        ground_truth = torch.cat([graph.nodes().unsqueeze(0), graph.nodes().unsqueeze(0)], 0)
    Adim = Afeat.shape[0]
    Bdim = Bfeat.shape[0]

    Aadj, Badj = np.zeros([Adim, Adim]), np.zeros([Bdim, Bdim])

    for u, v in zip(Aedge[0], Aedge[1]):
        Aadj[u][v] = 1
        Aadj[v][u] = 1

    for u, v in zip(Bedge[0], Bedge[1]):
        Badj[u][v] = 1
        Badj[v][u] = 1

    if edge_noise > 0.001:
        Badj = add_noise_edge(Bedge, Badj, edge_noise)
    return Aadj, Badj, Afeat, Bfeat, ground_truth


def add_noise_edge(Aedge, Aadj, edge_noise):
    Aadjn = np.zeros_like(Aadj)
    for u, v in zip(Aedge[0], Aedge[1]):
        if u==v:
            Aadjn[u][v] = 1
        if u<v:
            if random.random() > edge_noise:
                Aadjn[u][v], Aadjn[v][u] = 1, 1
            else:
                while 1:
                    u, v = random.randint(0, Aadj.shape[0]-1), random.randint(0, Aadj.shape[0]-1)
                    if u != v and Aadj[u][v] == 0:
                        Aadjn[u][v], Aadjn[v][u] = 1, 1
                        break
    return Aadjn


def feature_truncation(Bfeat, ratio=0.):
    feat_dim = Bfeat.shape[1]
    trancate_featdim = int(feat_dim*(1-ratio)+0.01)
    left_ids = random.sample(range(feat_dim), trancate_featdim)
    random.shuffle(left_ids)
    print('left dims:', len(left_ids))
    return Bfeat[:, left_ids]


def feature_compression(Bfeat, ratio=0.):
    pca = PCA(n_components=1-ratio, svd_solver='full')
    # Bfeat = pca
    Bfeat = pca.fit_transform(Bfeat)
    print('left dims:', Bfeat.shape[1])
    return Bfeat


def feature_permutation(Bfeat, ratio=0.):
    feat_dim = Bfeat.shape[1]
    permutation_featdim = int(feat_dim*ratio+0.01)
    permutation_ids = random.sample(range(feat_dim), permutation_featdim)
    permutation_ids2 = copy.deepcopy(permutation_ids)
    random.shuffle(permutation_ids2)
    Bfeat[:, permutation_ids] = Bfeat[:, permutation_ids2]
    return Bfeat


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w
