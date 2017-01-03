import matplotlib.pyplot as plt
import numpy as np

import tectosaur.quadrature as quad
import tectosaur.geometry as geometry
from tectosaur.interpolate import cheb, cheb_wts, barycentric_evalnd, to_interval
from tectosaur.limit import limit, richardson_limit

from tectosaur_tables.gpu_integrator import coincident_integral, adjacent_integral

# tol = 0.0001
# rho_order = 100
# starting_eps = 0.01
# n_eps = 7
# K = "U"

# tol = 1e-4
# rho_order = 100
# starting_eps = 1e-6
# n_eps = 3
# K = "T"
# K = "A"

K = "H"
tol = 1e-7
rho_order = 80
theta_order = 80
starting_eps = 1e-4
n_eps = 4

n_A = 8
n_B = 8
n_pr = 8

n_gpus = 1

# play parameters
# K = "H"
# tol = 1e-2
# rho_order = 30
# theta_order = 30
# starting_eps = 1e-1
# n_eps = 3
#
# n_A = 2
# n_B = 2
# n_pr = 2

filename = (
    '%s_%i_%f_%i_%f_%i_%i_%i_coincidenttable.npy' %
    (K, rho_order, starting_eps, n_eps, tol, n_A, n_B, n_pr)
)
print(filename)

all_eps = starting_eps * 2.0 ** -np.arange(n_eps)

def eval(i, pt):
    for j in range(10):
        print(i)
    Ahat,Bhat,prhat = pt
    # From symmetry and enforcing that the edge (0,0)-(1,0) is always longest,
    # A,B can be limited to (0,0.5)x(0,1).
    A = to_interval(0.0, 0.5, Ahat)
    B = to_interval(0.0, 1.0, Bhat)
    pr = to_interval(0.0, 0.5, prhat)

    tri = [[0,0,0],[1,0,0],[A,B,0.0]]

    integrals = []
    for eps in all_eps:
        print('running: ' + str((pt, eps)))
        res = coincident_integral(tol, K, tri, eps, 1.0, pr, rho_order, theta_order)
        print(res[0])
        integrals.append(res)
    integrals = np.array(integrals)
    lim = take_limits(integrals, True)
    return lim

def take_limits(integrals, remove_divergence):
    out = np.empty((81, 2))
    for i in range(81):
        out[i] = limit(all_eps, integrals[:, i], remove_divergence)
    return out

def test_f(results, eval_fnc, pts, wts):
    P = np.random.rand(pts.shape[1]) * 2 - 1.0
    correct = eval_fnc(0, P)[:,0]
    for i in range(81):
        interp = barycentric_evalnd(pts, wts, results[:,i,0], np.array([P]))[0]
        print("testing:  " + str(i) + "     " + str(
            (correct[i], interp, np.abs((correct[i] - interp) / correct[i]), correct[i] - interp)
        ))

def build_tables(eval_fnc, pts, wts):
    results = np.array([eval_fnc(i, p) for i, p in enumerate(pts.tolist())])
    np.save(filename, results)
    np.random.seed(15)
    for i in range(3):
        test_f(results, eval_fnc, pts, wts)

if __name__ == '__main__':
    Ahats = cheb(-1, 1, n_A)
    Bhats = cheb(-1, 1, n_B)
    prhats = cheb(-1, 1, n_pr)
    Ah,Bh,Nh = np.meshgrid(Ahats, Bhats,prhats)
    pts = np.array([Ah.ravel(),Bh.ravel(), Nh.ravel()]).T

    Awts = cheb_wts(-1,1,n_A)
    Bwts = cheb_wts(-1,1,n_B)
    prwts = cheb_wts(-1,1,n_pr)
    wts = np.outer(Awts,np.outer(Bwts,prwts)).ravel()

    build_tables(eval, pts, wts)
