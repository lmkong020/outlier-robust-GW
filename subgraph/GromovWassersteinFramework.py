"""
A Gromov-Wasserstein Learning Framework for Graph Analysis
Basic functionalities include:
1) Gromov-Wasserstein discrepancy (for graph partition)
2) Gromov-Wasserstein barycenter (for graph matching)
"""
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.special import softmax
from typing import List, Dict, Tuple


def extract_graph_info(graph, weights = None):
    idx2node = {}
    for i in range(len(graph.nodes)):
        idx2node[i] = i

    probs = np.zeros((len(graph.nodes), 1))
    adj = lil_matrix((len(graph.nodes), len(graph.nodes)))
    for edge in graph.edges:
        src = edge[0]
        dst = edge[1]
        if weights is None:
            adj[src, dst] += 1
            probs[src, 0] += 1
            probs[dst, 0] += 1
        else:
            adj[src, dst] += weights[src, dst]
            probs[src, 0] += weights[src, dst]
            probs[dst, 0] += weights[src, dst]

    return probs, csr_matrix(adj), idx2node

def extract_graph_info2(graph, weights = None):
    idx2node = {}
    for i in range(len(graph.nodes)):
        idx2node[i] = i
    nodes = list(graph.nodes)
    probs = np.zeros((len(graph.nodes), 1))
    adj = lil_matrix((len(graph.nodes), len(graph.nodes)))
    for edge in graph.edges:
        src = nodes.index(edge[0])
        dst = nodes.index(edge[1])
        if weights is None:
            adj[src, dst] += 1
            probs[src, 0] += 1
            probs[dst, 0] += 1
        else:
            adj[src, dst] += weights[src, dst]
            probs[src, 0] += weights[src, dst]
            probs[dst, 0] += weights[src, dst]

    return probs, csr_matrix(adj), idx2node

def node_distribution_similarity(p_s: np.ndarray, p_t: np.ndarray, values: list=None) -> np.ndarray:
    """
    Calculate the node distribution similarity matrix
    Args:
        p_s: (n_s, 1) array representing the distribution of source node
        p_t: (n_t, 1) array representing the distribution of target node
    Returns:
        cost_st: (n_s, n_t) the cost matrix between node probability
    """
    # index_s = np.argsort(p_s[:, 0]) / p_s.shape[0]
    # index_s = np.reshape(index_s, p_s.shape)
    # index_t = np.argsort(p_t[:, 0]) / p_t.shape[0]
    # index_t = np.reshape(index_t, p_t.shape)
    # cost_st = (np.repeat(index_s, p_t.shape[0], axis=1) - np.repeat(index_t, p_s.shape[0], axis=1).T) ** 2\
    #             - 2 * index_s @ index_t.T
    if values is None:
        cost_st = (np.repeat(p_s, p_t.shape[0], axis=1) -
                   np.repeat(p_t, p_s.shape[0], axis=1).T) ** 2  # - 2 * p_s @ p_t.T
    else:
        cost_st = (np.repeat(values[0] * p_s, p_t.shape[0], axis=1) -
                   np.repeat(values[1] * p_t, p_s.shape[0], axis=1).T) ** 2  # - 2 * p_s @ p_t.T
    return cost_st


def softmax_grad(x: np.ndarray) -> np.ndarray:
    """
    The gradient of softmax function
    Args:
        x: (N, 1) or (N, ) array representing a distribution generated by softmax function
    Returns:
        grad_x: (N, N) array, the Jacobian matrix representing the gradient of softmax
    """
    s = x.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def update_distribution(a: np.ndarray, p_s0: np.ndarray, theta0: np.ndarray,
                        beta: float, lr: float, weight: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update distribution via gradient descent
    Args:
        a: (n_s, 1) dual vector
        p_s0: (n_s, 1) current distribution
        theta0: (n_s, 1) current parameters of the distribution
        beta: the weight of first term
        lr: the learning rate
        weight: the weight of second term (regularizer)
    Returns:
        p_s: (n_s, 1) array of updated distribution
        theta: (n_s, 1) array of updated parameters
    """
    # update source distribution
    # grad_ps = beta * (np.log(a) - np.matmul((np.matmul(np.log(a), all1.transpose()) / kernel), all1))
    grad_ps = beta * np.log(a)
    if weight > 0:
        grad_ps -= (weight * (np.log(p_s0) + 1))
    grad_theta = np.matmul(softmax_grad(p_s0), grad_ps)
    # normalization
    grad_theta -= np.mean(grad_theta)
    grad_theta /= (1e-10 + np.sum(grad_theta ** 2) ** 0.5)
    theta = theta0 - lr * grad_theta
    p_s = softmax(theta)
    return p_s, theta


def sinkhorn_knopp_iteration(cost: np.ndarray, p_s: np.ndarray = None, p_t: np.ndarray = None,
                             a: np.ndarray = None, trans0: np.ndarray = None,
                             beta: float = 1e-1, error_bound: float = 1e-3,
                             max_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sinkhorn-Knopp iteration algorithm
    When initial optimal transport "trans0" is not available, the function solves
        min_{trans in Pi(p_s, p_t)} <cost, trans> + beta * <log(trans), trans>
    When initial optimal transport "trans0" is given, the function solves:
        min_{trans in Pi(p_s, p_t)} <cost, trans> + beta * KL(trans || trans0)
    Args:
        cost: (n_s, n_t) array representing distance between nodes
        p_s: (n_s, 1) array representing the distribution of source nodes
        p_t: (n_t, 1) array representing the distribution of target nodes
        a: (n_s, 1) array representing the dual variable
        trans0: (n_s, n_t) initial array of optimal transport
        beta: the weight of entropic regularizer
        error_bound: the error bound to check convergence
        max_iter: the maximum number of iterations
    Returns:
        trans: optimal transport
        a: updated dual variable
    """
    if p_s is None:
        p_s = np.ones((cost.shape[0], 1)) / cost.shape[0]

    if p_t is None:
        p_t = np.ones((cost.shape[1], 1)) / cost.shape[1]

    if a is None:
        a = np.ones((cost.shape[0], 1)) / cost.shape[0]

    # cost /= np.max(cost)
    if trans0 is not None:
        kernel = np.exp(-cost / beta) * trans0
    else:
        kernel = np.exp(-cost / beta)

    relative_error = np.inf
    b = []
    i = 0
    # print(a)
    while relative_error > error_bound and i < max_iter:
        b = p_t / (np.matmul(kernel.T, a))
        a_new = p_s / np.matmul(kernel, b)
        relative_error = np.sum(np.abs(a_new - a)) / np.sum(np.abs(a))
        a = a_new
        i += 1
    trans = np.matmul(a, b.T) * kernel
    # print('sinkhorn iteration = {}'.format(i))
    return trans, a


def node_cost_st(cost_s: csr_matrix, cost_t: csr_matrix,
                 p_s: np.ndarray, p_t: np.ndarray, loss_type: str = 'L2', prior: float = None) -> np.ndarray:
    """
    Calculate invariant cost between the nodes in different graphs based on learned optimal transport
    Args:
        cost_s: (n_s, n_s) array, the cost matrix of source graph
        cost_t: (n_t, n_t) array, the cost matrix of target graph
        p_s: (n_s, 1) array, the distribution of source nodes
        p_t: (n_t, 1) array, the distribution of target nodes
        loss_type: 'L2' the Euclidean loss type for Gromov-Wasserstein discrepancy
                   'KL' the KL-divergence loss type for Gromov-Wasserstein discrepancy
        prior: whether use node distribution similarity matrix as a prior
    Returns:
        cost_st: (n_s, n_t) array, the estimated invariant cost between the nodes in two graphs
    """
    n_s = cost_s.shape[0]
    n_t = cost_t.shape[0]
    if loss_type == 'L2':
        # f1(a) = a^2, f2(b) = b^2, h1(a) = a, h2(b) = 2b
        # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
        # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T

        # f1_st = np.repeat(np.matmul(cost_s ** 2, p_s), n_t, axis=1)
        # f2_st = np.repeat(np.matmul(p_t.T, (cost_t ** 2).T), n_s, axis=0)
        f1_st = np.repeat((cost_s ** 2) @ p_s, n_t, axis=1)
        f2_st = np.repeat(((cost_t ** 2) @ p_t).T, n_s, axis=0)
    else:
        # f1(a) = a*log(a) - a, f2(b) = b, h1(a) = a, h2(b) = log(b)
        # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
        # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T

        f1_st = np.repeat(np.matmul(cost_s * np.log(cost_s + 1e-15) - cost_s, p_s), n_t, axis=1)
        # f2_st = np.repeat(np.matmul(p_t.T, cost_t.T), n_s, axis=0)
        f2_st = np.repeat((cost_t @ p_t).T, n_s, axis=0)

    cost_st = f1_st + f2_st
    if prior is not None:
        cost_st += (prior * node_distribution_similarity(p_s, p_t))

    return cost_st


def node_cost(cost_s: csr_matrix, cost_t: csr_matrix, trans: np.ndarray,
              cost_st: np.ndarray, loss_type: str = 'L2') -> np.ndarray:
    """
    Calculate the cost between the nodes in different graphs based on learned optimal transport
    Args:
        cost_s: (n_s, n_s) array, the cost matrix of source graph
        cost_t: (n_t, n_t) array, the cost matrix of target graph
        trans: (n_s, n_t) array, the learned optimal transport between two graphs
        cost_st: (n_s, n_t) array, the estimated invariant cost between the nodes in two graphs
        loss_type: 'L2' the Euclidean loss type for Gromov-Wasserstein discrepancy
                   'KL' the KL-divergence loss type for Gromov-Wasserstein discrepancy
    Returns:
        cost: (n_s, n_t) array, the estimated cost between the nodes in two graphs
    """
    if loss_type == 'L2':
        # f1(a) = a^2, f2(b) = b^2, h1(a) = a, h2(b) = 2b
        # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
        # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T

        # cost = cost_st - 2 * np.matmul(np.matmul(cost_s, trans), cost_t.T)
        cost = cost_st - 2 * (cost_s @ trans @ cost_t.T)
    else:
        # f1(a) = a*log(a) - a, f2(b) = b, h1(a) = a, h2(b) = log(b)
        # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
        # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T

        # cost = cost_st - np.matmul(np.matmul(cost_s, trans), (np.log(cost_t + 1e-15)).T)
        cost = cost_st - np.matmul(cost_s @ trans, (np.log(cost_t + 1e-15)).T)
    return cost


def gromov_wasserstein_average(transports: Dict, costs: Dict,
                               p_center: np.ndarray, weights: Dict, loss_type: str) -> np.ndarray:
    """
    Averaging of cost matrix
    Args:
        transports: a dictionary, whose keys are graph ids and values are (n_s, n_c) np.ndarray of optimal transports
        costs: a dictionary, whose keys are graph ids and values are (n_s, n_s) np.ndarray of cost matrices
        p_center: (n_c, 1) np.ndarray of barycenter's distribution
        weights: a dictionary, whose keys are graph ids and values are float number of weight
        loss_type: 'L2' the Euclidean loss type for Gromov-Wasserstein discrepancy
                   'KL' the KL-divergence loss type for Gromov-Wasserstein discrepancy
    Returns:
        barycenter: (N, N) np.ndarray, the barycenter of cost matrix
    """
    barycenter = 0
    if loss_type == 'L2':
        for n in costs.keys():
            cost = costs[n]
            trans = transports[n]
            # barycenter += weights[n] * np.matmul(np.matmul(trans.T, cost), trans)
            barycenter += weights[n] * (trans.T @ (cost @ trans))
        barycenter /= np.matmul(p_center, p_center.T)
    else:
        for n in costs.keys():
            cost = costs[n]
            trans = transports[n]
            barycenter += weights[n] * np.matmul(np.matmul(trans.T, np.log(cost + 1e-15)), trans)
        barycenter /= np.matmul(p_center, p_center.T)
        barycenter = np.exp(barycenter)
    return barycenter


def gromov_wasserstein_discrepancy(cost_s: csr_matrix, cost_t: csr_matrix,
                                   p_s: np.ndarray, p_t: np.ndarray,
                                   ot_hyperpara: Dict, trans0=None) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Calculate Gromov-Wasserstein discrepancy with optionally-updated source probability
    Args:
        cost_s: (n_s, n_s) np.ndarray of source cost matrix
        cost_t: (n_t, n_t) np.ndarray of target cost matrix
        p_s: (n_s, 1) np.ndarray, the predefined source distribution
        p_t: (n_t, 1) np.ndarray, the predefined target distribution
        ot_hyperpara: dictionary of hyperparameter
        trans0: optional (n_s, n_t) array, the initial transport
    Returns:
        trans0: (n_s, n_t) array, the optimal transport
        d_gw: a float representing Gromov-Wasserstein discrepancy
        p_s: (n_s, 1) array, the optimal source distribution
    """

    n_s = cost_s.shape[0]
    if ot_hyperpara['update_p']:
        theta = np.zeros((n_s, 1))
        p_s = softmax(theta)
    else:
        theta = np.zeros((n_s, 1))

    if trans0 is None:
        trans0 = np.matmul(p_s, p_t.T)

    a = np.ones((n_s, 1)) / n_s

    t = 0
    relative_error = np.inf
    # calculate invariant cost matrix
    cost_st = node_cost_st(cost_s, cost_t, p_s, p_t,
                           loss_type=ot_hyperpara['loss_type'], prior=ot_hyperpara['node_prior'])
    cost_st = np.zeros_like(cost_st)
    while relative_error > ot_hyperpara['iter_bound'] and t < ot_hyperpara['outer_iteration']:
        # update optimal transport via Sinkhorn iteration method
        cost = node_cost(cost_s, cost_t, trans0, cost_st, ot_hyperpara['loss_type'])
        if ot_hyperpara['ot_method'] == 'proximal':
            trans, a = sinkhorn_knopp_iteration(cost=cost,
                                                p_s=p_s,
                                                p_t=p_t,
                                                a=a,
                                                trans0=trans0,
                                                beta=ot_hyperpara['beta'],
                                                error_bound=ot_hyperpara['sk_bound'],
                                                max_iter=ot_hyperpara['inner_iteration'])
        else:
            trans, a = sinkhorn_knopp_iteration(cost=cost,
                                                p_s=p_s,
                                                p_t=p_t,
                                                a=a,
                                                trans0=None,
                                                beta=ot_hyperpara['beta'],
                                                error_bound=ot_hyperpara['sk_bound'],
                                                max_iter=ot_hyperpara['inner_iteration'])
        relative_error = np.sum(np.abs(trans - trans0)) / np.sum(np.abs(trans0))
        trans0 = trans
        t += 1

        # optionally, update source distribution
        if ot_hyperpara['update_p']:
            p_s, theta = update_distribution(a, p_s, theta,
                                             ot_hyperpara['beta'], ot_hyperpara['lr'], ot_hyperpara['alpha'])
    # print('proximal iteration = {}'.format(t))
    cost = node_cost(cost_s, cost_t, trans0, cost_st, ot_hyperpara['loss_type'])
    d_gw = (cost * trans0).sum()
    return trans0, d_gw, p_s



def gromov_wasserstein_barycenter(costs: Dict, p_s: Dict, p_center: np.ndarray,
                                  ot_hyperpara: Dict, weights: Dict = None) -> Tuple[np.ndarray, Dict, List]:
    """
    Multi-graph matching based on one-step Gromov-Wasserstein barycenter learning.
    Args:
        costs: a dictionary, whose keys are graph ids and values are (n_s, n_s) cost matrices of different graphs
        p_s: a dictionary, whose keys are graph ids and values ara (n_s, 1) distributions of nodes of different graphs
        p_center: (n_c, 1) array, the distribution of barycenter's nodes
        ot_hyperpara: the dictionary of hyperparameters to train the Gromov-Wasserstein barycenter.
        weights: a dictionary, whose keys are graph ids and values are the weights of the graphs
    Returns:
        barycenter: (n_c, n_c) the cost matrix corresponding to the barycenter graph
        transports: a dictionary whose keys are graph ids and values are (n_s, n_c) optimal transports
        d_gw_sum: the sum of Gromov-Wasserstein discrepancy over iterations
    """
    # initialization
    num = len(costs)
    transports = {}
    for n in costs.keys():
        transports[n] = np.matmul(p_s[n], p_center.T)

    if weights is None:
        weights = {}
        for n in costs.keys():
            weights[n] = 1 / num

    # barycenter0 = np.random.rand(p_center.shape[0], p_center.shape[0])
    barycenter0 = csr_matrix(np.diag(p_center[:, 0]))

    d_gw_sum = []
    i = 0
    relative_error = np.inf
    while relative_error > ot_hyperpara['cost_bound'] and i < ot_hyperpara['max_iter']:
        # update optimal transport
        d_gw = {}
        for n in costs.keys():
            transports[n], d_gw[n], p_s[n] = gromov_wasserstein_discrepancy(costs[n], barycenter0,
                                                                            p_s[n], p_center,
                                                                            ot_hyperpara, transports[n])
        # averaging cost matrix
        barycenter = gromov_wasserstein_average(transports, costs, p_center, weights, ot_hyperpara['loss_type'])
        relative_error = np.sum(np.abs(barycenter - barycenter0)) / np.sum(np.abs(barycenter0))
        i += 1
        barycenter0 = barycenter
        d_gw_sum.append(d_gw)
    # print('barycenter iteration = {}'.format(i))
    return barycenter0, transports, d_gw_sum