import multiprocessing
import matplotlib.pyplot as plt
import numpy as np

import tectosaur.quadrature as quad
import tectosaur.geometry as geometry
from tectosaur.interpolate import cheb, cheb_wts, to_interval, barycentric_evalnd
from tectosaur.limit import limit, richardson_limit

from tectosaur_tables.gpu_integrator import coincident_integral, adjacent_integral
from coincident import build_tables

# H parameters
K = "H"
rho_order = 100
starting_eps = 0.0001
n_eps = 4
tol = 0.0001
n_pr = 8
n_theta = 8

# play parameters
K = "H"
rho_order = 40
theta_order = 28
starting_eps = 1e-4
n_eps = 2
tol = 1e-6
n_pr = 2
n_theta = 2

filename = (
    '%s_%i_%f_%i_%f_%i_%i_adjacenttable.npy' %
    (K, rho_order, starting_eps, n_eps, tol, n_theta, n_pr)
)
print(filename)

all_eps = starting_eps * 2.0 ** -np.arange(n_eps)
rho_gauss = quad.gaussxw(rho_order)

min_angle = 20
rho = 0.5 * np.tan(np.deg2rad(min_angle))

def eval(pt):
    thetahat, prhat = pt
    theta = to_interval(0, np.pi, thetahat)
    pr = to_interval(0.0, 0.5, prhat)
    print("(theta, pr) = " + str((theta, pr)))
    Y = rho * np.cos(theta)
    Z = rho * np.sin(theta)

    tri1 = [[0,0,0],[1,0,0],[0.5,rho,0]]
    tri2 = [[1,0,0],[0,0,0],[0.5,Y,Z]]
    integrals = []
    for eps in all_eps:
        print('running: ' + str((pt, eps)))
        rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
        res = adjacent_integral(tol, K, tri1, tri2, eps, 1.0, pr, rho_order, theta_order)
        print(res[0])
        integrals.append(res)
    return integrals

if __name__ == '__main__':
    thetahats = cheb(-1, 1, n_theta)
    prhats = cheb(-1, 1, n_pr)
    Th,Nh = np.meshgrid(thetahats,prhats)
    pts = np.array([Th.ravel(), Nh.ravel()]).T

    thetawts = cheb_wts(-1, 1, n_theta)
    prwts = cheb_wts(-1, 1, n_pr)
    wts = np.outer(thetawts, prwts).ravel()

    build_tables(eval, pts, wts)
