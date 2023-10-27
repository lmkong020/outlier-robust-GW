import numpy as np
import networkx as nx
import random
import copy
import torch


def BAPG_torch(A, B, a, b, X=None, epoch=200, eps=1e-8, rho=1e-1, min_rho=1e-1, scaling=1.0, early_stop=True):
    if a is None:
        a = torch.ones([A.shape[0], 1]).float().cuda() / A.shape[0]
    if b is None:
        b = torch.ones([B.shape[0], 1]).float().cuda() / B.shape[0]
    if X is None:
        X = a @ b.T
    obj_list, acc_list, res_list = [], [], []
    for ii in range(epoch):
        # X_old = X
        rho = max(rho / scaling, min_rho)
        X = X + 1e-10
        X = torch.exp(A @ X @ B / rho) * X
        X = X * (a / (X @ torch.ones_like(b)))
        X = torch.exp(A @ X @ B / rho) * X
        X = X * (b.T / (X.T @ torch.ones_like(a)).T)
        if early_stop and ii > 0 and ii % 50 == 0:
            objective = -torch.trace(A @ X @ B @ X.T)
            # print(ii, objective)
            if len(obj_list) > 0 and (objective - obj_list[-1]) / obj_list[-1] < eps:
                print('iter:{}, smaller than eps'.format(ii))
                break
            obj_list.append(objective)
        # print(torch.norm(X_old-X))
        # if torch.norm(X_old-X) < 1e-6:
        #     print('iter:{}, norm smaller than eps'.format(ii))
        #     break
    return X, obj_list


def BAPG_numpy(A, B, a, b, X=None, epoch=200, eps=None, rho=1e-1):
    if a is None:
        a = np.ones([A.shape[0], 1], dtype=np.float32) / A.shape[0]
    if b is None:
        b = np.ones([B.shape[0], 1], dtype=np.float32) / B.shape[0]
    if X is None:
        X = a @ b.T
    obj_list, acc_list, res_list = [], [], []
    for ii in range(epoch):
        X_old = X
        X = X + 1e-10
        X = np.exp(A @ X @ B / rho) * X
        X = X * (a / (X @ np.ones_like(b)))
        X = np.exp(A @ X @ B / rho) * X
        X = X * (b.T / (X.T @ np.ones_like(a)).T)
        if eps is not None and np.linalg.norm(X_old - X) < eps:
            print('iter:{}, smaller than eps'.format(ii))
            break
        if ii > 0 and ii % 50 == 0:
            objective = -np.trace(A @ X @ B @ X.T)
            # print(ii, objective)
            if len(obj_list) > 0 and np.abs((objective - obj_list[-1]) / obj_list[-1]) < 1e-5:
                print('iter:{}, smaller than eps'.format(ii))
                break
            obj_list.append(objective)
    return X, obj_list


def add_noisy_edges(graph: nx.graph, noisy_level: float) -> nx.graph:
    nodes = list(graph.nodes)
    num_edges = len(graph.edges)
    num_noisy_edges = int(noisy_level * num_edges)
    graph_noisy = copy.deepcopy(graph)
    if num_noisy_edges > 0:
        i = 0
        while i < num_noisy_edges:
            src = random.choice(nodes)
            dst = random.choice(nodes)
            if (src, dst) not in graph_noisy.edges:
                graph_noisy.add_edge(src, dst)
                i += 1
    return graph_noisy


def add_noisy_nodes(graph: nx.graph, noisy_level: float) -> nx.graph:
    num_nodes = len(graph.nodes)
    num_noisy_nodes = int(noisy_level * num_nodes)

    num_edges = len(graph.edges)
    num_noisy_edges = int(noisy_level * num_edges / num_nodes + 1)

    graph_noisy = copy.deepcopy(graph)
    if num_noisy_nodes > 0:
        for i in range(num_noisy_nodes):
            graph_noisy.add_node(int(i + num_nodes))
            j = 0
            while j < num_noisy_edges:
                src = random.choice(list(range(i + num_nodes)))
                if (src, int(i + num_nodes)) not in graph_noisy.edges:
                    graph_noisy.add_edge(src, int(i + num_nodes))
                    j += 1
    return graph_noisy


def add_noisy_nodes(graph: nx.graph, noisy_level: float) -> nx.graph:
    num_nodes = len(graph.nodes)
    num_noisy_nodes = int(noisy_level * num_nodes)

    num_edges = len(graph.edges)
    num_noisy_edges = int(noisy_level * num_edges / num_nodes + 1)

    graph_noisy = copy.deepcopy(graph)
    if num_noisy_nodes > 0:
        for i in range(num_noisy_nodes):
            graph_noisy.add_node(int(i + num_nodes))
            j = 0
            while j < num_noisy_edges:
                src = random.choice(list(range(i + num_nodes)))
                if (src, int(i + num_nodes)) not in graph_noisy.edges:
                    graph_noisy.add_edge(src, int(i + num_nodes))
                    j += 1
    return graph_noisy


def node_correctness(coup, perm_inv):
    coup_max = coup.argmax(1)
    perm_inv_max = perm_inv.argmax(1)
    acc = np.sum(coup_max == perm_inv_max) / len(coup_max)
    return acc


def node_correctness2(coup, ans):
    coup_max = coup.argmax(1)
    acc = np.sum(coup_max == ans) / len(coup_max)
    return acc


def random_subgraph(G, ratio):
    nodes_num = G.number_of_nodes()
    sample_num = int(nodes_num * ratio)
    sample_idx = random.sample(range(nodes_num), sample_num)
    sample_idx.sort()
    subG = G.subgraph(sample_idx)
    return subG, sample_idx


def connected_subgraph(G, ratio):
    nodes_num = G.number_of_nodes()
    sample_num = int(nodes_num * ratio)
    best_node = nx.to_numpy_array(G).astype(np.float32).sum(0).argmax()
    bfs_idx = list(nx.bfs_tree(G, best_node).nodes())[:sample_num]
    bfs_idx.sort()
    subG = G.subgraph(bfs_idx)
    return subG, bfs_idx
