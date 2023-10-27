import numpy as np
import time
from unbalancedgw.vanilla_ugw_solver import log_ugw_sinkhorn
import ot
from BAPG import *
from robust_gw import *
from collections import defaultdict
import pickle
import warnings
from partial_gw import pu_gw_emd
from GromovWassersteinFramework import *
import GromovWassersteinGraphToolkit as GwGt
from gromovWassersteinAveraging import *
import spectralGW as sgw
import networkx as nx
import torch.nn as nn
from utils_douban import *
from srGW import *
random.seed(123)
torch.random.manual_seed(123)
np.random.seed(123)

# import data
A, B, Afeat, Bfeat, gt = myload('douban', 0)
m,n = A.shape[0], B.shape[0]

# create initial point
Afeat=torch.tensor(Afeat).cuda()
Bfeat=torch.tensor(Bfeat).cuda()
sim = Afeat@Bfeat.T
G = nx.Graph(zip(A.nonzero()[0],A.nonzero()[1]))
G_noise = nx.Graph(zip(B.nonzero()[0],B.nonzero()[1]))


def NeuralSinkhorn(cost, p_s=None, p_t=None, trans=None, beta=0.1, outer_iter=20):
    if p_s is None:
        p_s = torch.ones([cost.shape[0],1],device=cost.device,dtype=cost.dtype)/cost.shape[0]
    if p_t is None:
        p_t = torch.ones([cost.shape[1],1],device=cost.device,dtype=cost.dtype)/cost.shape[1]
    if trans is None:
        trans = p_s @ p_t.T
    a = torch.ones([cost.shape[0],1],device=cost.device,dtype=cost.dtype)/cost.shape[0]
    cost_new = torch.exp(-cost / beta)
    for oi in range(outer_iter):
        b = p_t / (cost_new.T@a)
        a = p_s / (cost_new@b)
    trans = (a @ b.T) * cost_new
    return trans

#####Create Initial point using features via OT###########################
sim2=NeuralSinkhorn(1-sim,outer_iter=500,beta=0.1)
a1,a5,a10,a30 = my_check_align1(sim2.T, gt)

##### Robust GW ##########################################################
m, n = A.shape[0], B.shape[0]
p = np.ones([m, 1]).astype(np.float32) / m
q = np.ones([n, 1]).astype(np.float32) / n
Xinit = p @ q.T
start = time.time()
coup_rgw, obj_list_rgw, alpha, beta = robust_gw(Ds=A, Dt=B, a=p, b=q, PALM_maxiter=1000,
                                                rho1=0.1, rho2=0.1, eta=0.001, t1=0.1, t2=0.1, tau1=0.1, tau2=0.1,
                                                relative_error=1e-6, init_X = sim2.cpu().numpy(), init_alpha=p, init_beta=q)

end = time.time()
time_rgw = end - start
a1_rgw, a5_rgw, a10_rgw, a30_rgw = my_check_align1(coup_rgw.T, gt)

###### Unbalanced GW #################################################
for i in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
    a = np.ones(m) / m
    b = np.ones(n) / n
    A_double = A.astype('double')
    B_double = B.astype('double')
    CX, CY = torch.from_numpy(A_double), torch.from_numpy(B_double)
    P, Q = torch.from_numpy(a), torch.from_numpy(b)
    init_X_ugw = sim2
    rho = i
    eps = 0.001
    PI = None

    start = time.time()
    PI = log_ugw_sinkhorn(
        P,
        CX,
        Q,
        CY,
        init=init_X_ugw,
        eps=eps,
        rho=rho,
        rho2=rho,
        nits_plan=500,
        tol_plan=1e-12,
        nits_sinkhorn=500,
        tol_sinkhorn=1e-6,
    )
    end = time.time()
    coup_ugw = PI.cpu().data.numpy()

    time_ugw = end - start
    a1_ugw, a5_ugw, a10_ugw, a30_ugw = my_check_align1(coup_ugw.T, gt)

########## Partial GW #################################################################################################################
p = np.ones(m).astype(np.float32) / m
q = np.ones(n).astype(np.float32) / n
for i in range(0.1, 0.9, 0.1):
    start = time.time()
    coup_pgw = ot.partial.partial_gromov_wasserstein(C1=A, C2=B, p=p, q=q, m=0.85, G0=sim2.cpu().numpy(), log=False, numItermax=500,
                                                     tol=1e-6)
    # coup_pgw = pu_gw_emd(C1=A, C2=B, p=p.ravel(), q=q.ravel(), nb_dummies=2788, G0=init_X, log=False, max_iter=500)
    end = time.time()
    times_pgw= end - start
    a1_pgw, a5_pgw, a10_pgw, a30_pgw = my_check_align1(coup_pgw.T, gt)

######FW###########################################################################################################################

p = np.ones([m, 1]).astype(np.float32) / m
q = np.ones([n, 1]).astype(np.float32) / n
start = time.time()
coup_FW, log = ot.gromov.gromov_wasserstein(A, B, p.squeeze(), q.squeeze(), loss_fun='kl_loss', log=True, numItermax=500, G0=sim2.cpu().numpy())
end = time.time()

time_FW = end - start
a1_FW, a5_FW, a10_FW, a30_FW = my_check_align1(coup_FW.T, gt)

######BAPG###########################################################################################################################
start = time.time()
rho = 0.01
Xinit = p @ q.T
coup_bap, obj_list_bap = BAPG_numpy(A=A, B=B, a=p, b=q, X=sim2.cpu().numpy(), epoch=1000, eps=1e-6,
                                    rho=rho)
end = time.time()
time_BAPG = end - start
a1_BAPG, a5_BAPG, a10_BAPG, a30_BAPG = my_check_align(coup_bap.T, gt)

#######BPG##############################################################################################################################
def gw_torch(cost_s, cost_t, p_s=None, p_t=None, trans0=None, beta=1e-1, error_bound=1e-6,
                             outer_iter=500, inner_iter=1, gt=None):
    a = torch.ones_like(p_s)/p_s.shape[0]
    if trans0 is None:
        trans0 = p_s @ p_t.T
    for oi in range(outer_iter):
        cost = - 2 * (cost_s @ trans0 @ cost_t.T)
        kernel = torch.exp(-cost / beta) * trans0
        # a = torch.ones_like(p_s)/p_s.shape[0]
        for ii in range(inner_iter):
            b = p_t / (kernel.T@a)
            a_new = p_s / (kernel@b)
            relative_error = torch.sum(torch.abs(a_new - a)) / torch.sum(torch.abs(a))
            a = a_new
            if relative_error < 1e-10:
                break
        trans = (a @ b.T) * kernel
        relative_error = torch.sum(torch.abs(trans - trans0)) / torch.sum(torch.abs(trans0))
        if relative_error < error_bound:
            break
        trans0 = trans
    return trans

A_tensor = torch.tensor(A).float().cuda()
B_tensor = torch.tensor(B).float().cuda()
a = torch.ones([m,1]).cuda()/m
b = torch.ones([n,1]).cuda()/n
X = gw_torch(A_tensor, B_tensor, a, b, trans0=sim2.float(), beta=0.01, outer_iter=500, inner_iter=10)
res=X.cpu().T
a1_BPG,a5_BPG,a10_BPG,a30_BPG = my_check_align(res, gt)

######eBPG##############################################################################################################################
from ot.gromov import *
def entropic_gromov_wasserstein3(C1, C2, p, q, loss_fun, initX, epsilon,max_iter=1000, tol=1e-9, verbose=False, log=False):
    C1, C2, p, q = list_to_array(C1, C2, p, q)
    nx = get_backend(C1, C2, p, q)

    T = initX

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    cpt = 0
    err = 1

    if log:
        log = {'err': []}

    while (cpt < max_iter):

        Tprev = T

        # compute the gradient
        tens = gwggrad(constC, hC1, hC2, T)

        T = sinkhorn(p, q, tens, epsilon, method='sinkhorn')

        if cpt % 1 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(T - Tprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        # a1,a5,a10,a30 = my_check_align(T.T, ground_truth)
        # print(a1,a5,a10,a30)
        cpt += 1

    if log:
        log['gw_dist'] = gwloss(constC, hC1, hC2, T)
        return T, log
    else:
        return T


p = np.ones([m, 1]).astype(np.float32) / m
q = np.ones([n, 1]).astype(np.float32) / n

start = time.time()
coup_eBPG, _ = entropic_gromov_wasserstein3(A, B, a.squeeze(-1), b.squeeze(-1), initX=sim2.float(),
                                                     loss_fun='square_loss', epsilon=1e-2,
                                                     verbose=True, log=True, max_iter=500)
end = time.time()
time_eBPG = end - start
a1_eBPG, a5_eBPG, a10_eBPG, a30_eBPG = my_check_align(coup_eBPG.T, gt)

#########SpecGWL#######################################################################################################################
p = np.ones([m, 1]).astype(np.float32) / m
q = np.ones([n, 1]).astype(np.float32) / n
t = 10
start = time.time()
G_hk = sgw.undirected_normalized_heat_kernel(G, t)
G_hk_noise = sgw.undirected_normalized_heat_kernel(G_noise, t)
coup_hk, log_hk = ot.gromov.gromov_wasserstein(G_hk, G_hk_noise, p.squeeze(), q.squeeze(),
                                               loss_fun='square_loss', log=True, numItermax=500, G0=sim2.cpu().numpy())
end = time.time()
a1_hk, a5_hk, a10_hk, a30_hk = my_check_align1(coup_hk, gt)

##########srGW#####################################################################################################################
a = (np.ones(m) / m)
b = (np.ones(n) / n)
start = time.time()
A_double = A.astype('double')
B_double = B.astype('double')
CX, CY = torch.from_numpy(A_double).float(), torch.from_numpy(B_double).float()
P, Q = torch.from_numpy(a).float(), torch.from_numpy(b).float()
init_X = sim2.cpu().data.numpy().T
init_X_tensor = torch.from_numpy(init_X).float()
# mirror descent 
start = time.time()
coup_srGW_md, _ = md_semirelaxed_gromov_wasserstein(C1=CY, p=Q, C2=CX,gamma_entropy=10,eps=1e-12)
end = time.time()
coup_srGW_md = coup_srGW_md.cpu().data.numpy()
a1_eBPG, a5_eBPG, a10_eBPG, a30_eBPG = my_check_align1(coup_srGW_md, gt)
