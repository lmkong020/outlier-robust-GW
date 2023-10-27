import time
import ot
from BAPG import *
from robust_gw import *
from unbalancedgw.vanilla_ugw_solver import log_ugw_sinkhorn
from collections import defaultdict
import pickle
import warnings
from partial_gw import pu_gw_emd
from GromovWassersteinFramework import *
import GromovWassersteinGraphToolkit as GwGt
from gromovWassersteinAveraging import *
from scipy.sparse import csr_matrix, lil_matrix
import spectralGW as sgw
from srGW import *
import copy


def random_subgraph_new(G, overlap_ratio):
    nodes_num = G.number_of_nodes()
    sample_num = int(nodes_num * overlap_ratio)
    overlap_idx = list(range(sample_num))
    # overlap_idx = random.sample(range(nodes_num), sample_num)
    graph1_idx = copy.deepcopy(overlap_idx)
    graph2_idx = copy.deepcopy(overlap_idx)
    for i in range(nodes_num):
        if i not in overlap_idx:
            if random.random() < 0.5:
                graph1_idx.append(i)
            else:
                graph2_idx.append(i)
    # graph1_idx.sort()
    subG1 = G.subgraph(graph1_idx)
    # graph2_idx.sort()
    subG2 = G.subgraph(graph2_idx)
    overlap_idx0 = list(range(len(overlap_idx)))
    index = torch.tensor([overlap_idx0, overlap_idx0])
    # print(index)
    return subG1, subG2, index


def my_check_align(pred, ground_truth):
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
    return a1


warnings.filterwarnings("ignore")
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

database = 'proteins'  # proteins / reddit / enzymes / gaussian / powerlaw
overlap_ratio = 0.9
subgrpah_type = 'random'

graphs, new_graphs = [], []
ans = []

if database == 'proteins':
    print('------------------Node Matching on PROTIENS---------------')
    with open('data/PROTEINS/matching.pk', 'rb') as f:
        graphs, _ = pickle.load(f)

if database == 'reddit':
    print('------------------Node Matching on REDDIT---------------')
    with open('data/REDDIT-BINARY/matching.pk', 'rb') as f:
        graphs = pickle.load(f)[:500]

if database == 'enzymes':
    print('------------------Node Matching on ENZYMES---------------')
    with open('data/ENZYMES/matching.pk', 'rb') as f:
        graphs = pickle.load(f)

if database == 'synthetic':
    graphs, noise_graphs = [], []
    print('------------------Node Matching on Synthetic Database---------------')
    with open('data/Random/Graph1.pk', 'rb') as f:
        graph_pairs = pickle.load(f)
        for num_node in [500, 1000, 1500, 2000, 2500]:
            for noise_level in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                for G, G_noise in graph_pairs[(num_node, noise_level)]:
                    graphs.append(G)
                    noise_graphs.append(G_noise)

if database == 'powerlaw':
    graphs = []
    for nodes_num in [100, 200, 300, 400, 500]:
        for samples in range(10):
            g = nx.powerlaw_cluster_graph(nodes_num, 10, 0.01)
            graphs.append(g)

if database == 'gaussian':
    graphs = []
    for nodes_num in [100, 200, 300, 400, 500]:
        for samples in range(10):
            g = nx.gaussian_random_partition_graph(nodes_num, nodes_num / 10, 10, 0.25, 0.1)
            graphs.append(g)

new_graphs1, new_graphs2 = [], []
for G in graphs:
    subG1, subG2, idx = random_subgraph_new(G, overlap_ratio)
    new_graphs1.append(subG1)
    new_graphs2.append(subG2)
    ans.append(idx)

total_num_graphs = len(graphs)
print('total_num_graphs: ', total_num_graphs)
diff_times = []

for ttt in range(1):
    results, times = defaultdict(list), defaultdict(list)

    for j in range(0, total_num_graphs):  ## total_num_graphs
        print('graph id: ', j)
        G = new_graphs1[j]
        G_noise = new_graphs2[j]
        idx = ans[j]

        G_adj = nx.to_numpy_array(G).astype(np.float32)
        G_adj_noise = nx.to_numpy_array(G_noise).astype(np.float32)
        m, n = G_adj.shape[0], G_adj_noise.shape[0]
        p = np.ones([m, 1]).astype(np.float32) / m
        q = np.ones([n, 1]).astype(np.float32) / n
        Xinit = p @ q.T

        ######FW###########################################################################################################################
        p = np.ones([m, 1]).astype(np.float32) / m
        q = np.ones([n, 1]).astype(np.float32) / n
        # t = 10
        
        start = time.time()
        coup, log = ot.gromov.gromov_wasserstein(G_adj, G_adj_noise, p.squeeze(), q.squeeze(),
                                                 loss_fun='kl_loss', log=True, numItermax=500)
        end = time.time()
        
        times['FW'].append(end - start)
        results['FW'].append(my_check_align(coup, idx))

        ######BAPG###########################################################################################################################
        start = time.time()
        rho = 0.1
        coup_bap, obj_list_bap = BAPG_numpy(A=G_adj, B=G_adj_noise, a=p, b=q, X=Xinit, epoch=500, eps=1e-6,
                                            rho=rho)
        end = time.time()
        times['BAPGcpu'].append(end - start)
        results['BAPGcpu'].append(my_check_align(coup_bap, idx))

        # #######BPG##############################################################################################################################
        ot_hyperpara_adj = {'loss_type': 'L2',  # the key hyperparameters of GW distance
                            'ot_method': 'proximal',
                            'beta': 0.2,  #
                            'outer_iteration': 200,
                            'iter_bound': 1e-10,
                            'inner_iteration': 500,  # origin: 1, BPG:500
                            'sk_bound': 1e-5,  # origin: 1e-30, BPG:1e-5
                            'node_prior': 0,
                            'max_iter': 500,  # iteration and error bound for calculating barycenter
                            'cost_bound': 1e-16,
                            'update_p': False,  # optional updates of source distribution
                            'lr': 0,
                            'alpha': 0}
        start = time.time()
        coup_adj, d_gw, p_s = gromov_wasserstein_discrepancy(G_adj, G_adj_noise, p, q, ot_hyperpara_adj)
        end = time.time()
        times['BPG'].append(end - start)
        results['BPG'].append(my_check_align(coup_adj, idx))

        ######eBPG##############################################################################################################################
        p = np.ones([m, 1]).astype(np.float32) / m
        q = np.ones([n, 1]).astype(np.float32) / n
        start = time.time()
        coup_adj, _ = ot.gromov.entropic_gromov_wasserstein(G_adj, G_adj_noise, p.squeeze(-1), q.squeeze(-1),
                                                            loss_fun='square_loss', epsilon=1e-1,
                                                            verbose=False, log=True, max_iter=500)
        end = time.time()
        times['eBPG'].append(end - start)
        results['eBPG'].append(my_check_align(coup_adj, idx))

        #########SpecGWL#######################################################################################################################
        p = np.ones([m, 1]).astype(np.float32) / m
        q = np.ones([n, 1]).astype(np.float32) / n
        t = 10
        start = time.time()
        G_hk = sgw.undirected_normalized_heat_kernel(G, t)
        G_hk_noise = sgw.undirected_normalized_heat_kernel(G_noise, t)
        start2 = time.time()
        coup_hk, log_hk = ot.gromov.gromov_wasserstein(G_hk, G_hk_noise, p.squeeze(), q.squeeze(),
                                                       loss_fun='square_loss', log=True)
        end = time.time()
        
        times['SpecGWL'].append(end - start)
        diff_times.append(end - start2)
        results['SpecGWL'].append(my_check_align(coup_hk, idx))

        ###########srGW###############################################################################################################
        a = (np.ones(m) / m)
        b = (np.ones(n) / n)
        start = time.time()
        G_adj_double = G_adj.astype('double')
        G_adj_noise_double = G_adj_noise.astype('double')
        CX, CY = torch.from_numpy(G_adj_double).float(), torch.from_numpy(G_adj_noise_double).float()
        A, B = torch.from_numpy(a).float(), torch.from_numpy(b).float()
        # mirror descent
        start = time.time()
        for v in [0.1, 0.5, 1.0, 5.0, 10.0]:
            coup_srGW_md, _ = md_semirelaxed_gromov_wasserstein(C1=CX, p=A, C2=CY, gamma_entropy=v, eps=1e-6)
            end = time.time()
            coup_srGW_md = coup_srGW_md.cpu().data.numpy()
            name = 'srGW_' + str(v)
            times[name].append(end - start)
            results[name].append(my_check_align(coup_srGW_md, idx))

        ###########Robust GW###############################################################################################################
        p = np.ones([m, 1]).astype(np.float32) / m
        q = np.ones([n, 1]).astype(np.float32) / n
        start = time.time()
        coup_rgw, obj_list_rgw, alpha, beta = robust_gw(Ds=G_adj, Dt=G_adj_noise, a=p, b=q, PALM_maxiter=500,
                                                        rho1=0.05, rho2=0.05, eta=0.01,
                                                        t1=0.1, t2=0.1,
                                                        tau1=0.1, tau2=0.1, relative_error=1e-6)
        end = time.time()
        times['rgw'].append(end - start)
        results['rgw'].append(my_check_align(coup_rgw, idx))

        ###########Unbalanced GW###############################################################################################################
        a = np.ones(m) / m
        b = np.ones(n) / n
        G_adj_double = G_adj.astype('double')
        G_adj_noise_double = G_adj_noise.astype('double')
        CX, CY = torch.from_numpy(G_adj_double), torch.from_numpy(G_adj_noise_double)
        A, B = torch.from_numpy(a), torch.from_numpy(b)
        rho = 0.05
        eps = 0.001
        PI = None
        
        start = time.time()
        PI = log_ugw_sinkhorn(
            A,
            CX,
            B,
            CY,
            init=PI,
            eps=eps,
            rho=rho,
            rho2=rho,
            nits_plan=500,
            tol_plan=1e-6,
            nits_sinkhorn=500,
            tol_sinkhorn=1e-4,
        )
        end = time.time()
        coup_ugw = PI.cpu().data.numpy()
        times['ugw'].append(end - start)
        results['ugw'].append(my_check_align(coup_ugw, idx))

        ###########Partial GW###############################################################################################################
        p = np.ones([m, 1]) / m
        q = np.ones([n, 1]) / n
        ratio = m / n
        start = time.time()
        # coup_partial = pu_gw_emd(C1=G_adj_noise, C2=G_adj, p=q.ravel(), q=p.ravel(), nb_dummies=1, G0=None, log=False,
        #                          max_iter=1000)
        coup_partial = ot.partial.partial_gromov_wasserstein(C1=G_adj_noise, C2=G_adj, p=q.ravel(), q=p.ravel(),
                                                             m=0.9, log=False, numItermax=500)
        end = time.time()
        times['pgw'].append(end - start)
        results['pgw'].append(my_check_align(coup_partial, idx))

    print('---------------------------------Completed---------------------------------------')
    # with open('result2/align_523_{}_{}_{}.pk'.format(database,noise_level,ttt), 'wb') as f:
    #     pickle.dump(results, f)
    for method, result in results.items():
        print('Method: {} Mean: {:.4f}, Std: {:.4f}, Time: {:.4f}'.format(method,
                                                                          np.mean(results[method]),
                                                                          np.std(results[method]),
                                                                          np.sum(times[method])))
