import numpy as np
from copy import copy
import torch
import ot


def robust_gw(Ds, Dt, a, b, PALM_maxiter, rho1, rho2, eta, t1, t2, tau1, tau2, relative_error, init_X=None,
              init_alpha=None, init_beta=None):
    n = len(Ds)
    m = len(Dt)
    e = np.ones((m, 1))
    f = np.ones((n, 1))
    if init_X is None:
        X = np.ones((n, m)) / n / m
    else:
        X = init_X
    if init_alpha is None:
        alpha = np.ones((n, 1)) / n
    else:
        alpha = init_alpha
    if init_beta is None:
        beta = np.ones((m, 1)) / m
    else:
        beta = init_beta
    max_iter = PALM_maxiter
    obj = np.matrix.trace(tensor_product(Ds, Dt, X) @ np.transpose(X)) + tau1 * kl_divergence(X @ e,
                                                                                              alpha) + tau2 * kl_divergence(
        X.T @ f, beta)
    objective_list = [obj]
    for i in range(max_iter):
        obj_old = obj
        X_old = X + 1e-10
        alpha_old = alpha
        beta_old = beta
        X, inner_iter = sinkhorn_uot(Ds, Dt, X_old, alpha, beta, eta, tau1, tau2)
        if rho1 == 0:
            alpha = a
        else:
            alpha = update_alpha(X, alpha_old, a, rho1, t1)
        if rho2 == 0:
            beta = b
        else:
            beta = update_beta(X, beta_old, b, rho2, t2)
        res = np.max(np.abs(X - X_old))
        if res < relative_error:
            print('iter:{}, smaller than eps'.format(i))
            break
        if i > 0 and i % 50 == 0:
            objective = np.matrix.trace(tensor_product(Ds, Dt, X) @ np.transpose(X)) + tau1 * kl_divergence(X @ e,
                                                                                                            alpha) + tau2 * kl_divergence(
                X.T @ f, beta)
            # print(ii, objective)
            if len(objective_list) > 0 and np.abs((objective - objective_list[-1]) / objective_list[-1]) < 1e-5:
                print('iter:{}, smaller than eps'.format(i))
                break
            objective_list.append(objective)
    return X, objective_list, alpha, beta


def tensor_product(Ds, Dt, T):
    n = len(Ds)
    m = len(Dt)
    f1 = lambda a: pow(a, 2)
    f2 = lambda b: pow(b, 2)
    h1 = lambda a: a
    h2 = lambda b: 2 * b
    e = np.ones((m, 1))
    f = np.ones((n, 1))
    et = np.ones((1, m))
    ft = np.ones((1, n))
    return f1(Ds) @ (T @ e) @ et + f @ (ft @ T) @ np.transpose(f2(Dt)) - h1(Ds) @ T @ np.transpose(h2(Dt))


def sinkhorn_uot(Ds, Dt, X_old, alpha, beta, eta, tau1, tau2):
    n = len(Ds)
    m = len(Dt)
    C = tensor_product(Ds, Dt, X_old)
    u = np.zeros((n, 1))
    v = np.zeros((m, 1))
    logX = eta * np.log(X_old)
    C = C - logX
    max_iter = 500

    for i in range(max_iter):
        u_old = u
        v_old = v
        B = np.exp((u + v.T - C) / eta)

        # Sinkhorn update
        if i % 2 == 0:
            Ba = B.sum(axis=1).reshape(-1, 1)
            u = (u / eta + np.log(alpha) - np.log(Ba)) * (tau1 * eta / (eta + tau1))
        else:
            Bb = B.sum(axis=0).reshape(-1, 1)
            v = (v / eta + np.log(beta) - np.log(Bb)) * (tau2 * eta / (eta + tau2))

        err = np.linalg.norm(u - u_old) + np.linalg.norm(v - v_old)
        if err < 1e-4:
            break
    return B, i


def compute_B(C, u, v, eta):
    return np.exp((u + v.T - C) / eta)


def update_alpha(X, alpha_old, mu, rho1, t1):
    X1 = np.sum(X, axis=1)
    m = len(X1)
    X1 = X1.reshape((m, 1))
    s = np.sum(X1)
    t = t1
    alpha0 = (X1 + t * alpha_old) / (s + t)
    if sum(np.multiply(mu, np.log(np.divide(mu, alpha0)))) - rho1 <= 0:
        return alpha0
    else:
        init = 0
        w = newton_method_alpha(s, X1, t, alpha_old, mu, rho1, init)
        alpha = (X1 + t * alpha_old + w * mu) / (s + t + w)
        return alpha


def update_beta(X, beta_old, nu, rho2, t2):
    X2 = np.transpose(np.sum(X, axis=0))
    n = len(X2)
    X2 = X2.reshape((n, 1))
    s = sum(X2)
    t = t2
    beta0 = (X2 + t * beta_old) / (s + t)
    if sum(np.multiply(nu, np.log(np.divide(nu, beta0)))) - rho2 <= 0:
        return beta0
    else:
        init = 0
        w = newton_method_beta(s, X2, t, beta_old, nu, rho2, init)
        beta = (X2 + t * beta_old + w * nu) / (s + t + w)
        return beta


def newton_method_alpha(s, X1, t, alpha_old, mu, rho1, init):
    curr = init
    fix1 = X1 + t * alpha_old
    fix2 = s + t
    mu_square = np.power(mu, 2)
    for i in range(500):
        alpha = (fix1 + curr * mu) / (fix2 + curr)
        func = np.sum(np.multiply(mu, np.log(np.divide(mu, alpha)))) - rho1
        func_diff = np.sum(mu / (fix2 + curr)) - np.sum(np.divide(mu_square, (fix1 + curr * mu)))
        curr = curr - func / func_diff
        error = abs(func)
        if error < 1e-5:
            return curr


def newton_method_beta(s, X2, t, beta_old, nu, rho2, init):
    curr = init
    fix1 = X2 + t * beta_old
    fix2 = s + t
    nu_square = np.power(nu, 2)
    for i in range(100):
        beta = (fix1 + curr * nu) / (fix2 + curr)
        func = np.sum(np.multiply(nu, np.log(np.divide(nu, beta)))) - rho2
        func_diff = np.sum(nu / (fix2 + curr)) - np.sum(np.divide(nu_square, (fix1 + curr * nu)))
        curr = curr - func / func_diff
        error = abs(func)
        if error < 1e-5:
            return curr


def kl_divergence(a, b):
    return np.sum(np.multiply(a, np.log(np.divide(a, b))) - a + b)
