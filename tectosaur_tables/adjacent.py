import matplotlib.pyplot as plt
import numpy as np

from tectosaur.interpolate import to_interval
from tectosaur.table_lookup import adjacent_interp_pts_wts

from tectosaur_tables.fixed_integrator import adjacent_fixed
from tectosaur_tables.build_tables import fixed_quad, TableParams, take_limits, get_eps
from tectosaur_tables.better_limit import limit


def make_adjacent_params(K, tol, low_nq, check_quad, adaptive_quad,
        n_rho, n_theta, starting_eps, n_eps, include_log, n_phi, n_pr):
    pts, wts = adjacent_interp_pts_wts(n_phi, n_pr)
    p = TableParams(
        K, tol, low_nq, check_quad, adaptive_quad, n_rho, n_theta,
        starting_eps, n_eps, include_log, (n_phi, n_pr), pts, wts
    )
    p.filename = (
        '%s_%i_%f_%i_%f_%i_%i_adjacenttable.npy' %
        (K, n_rho, starting_eps, n_eps, tol, n_phi, n_pr)
    )
    return p

def eval_tri_integral(obs_tri, src_tri, pr, p):
    epsvs = get_eps(p.n_eps, p.starting_eps)

    integrals = []
    last_orders = None
    for eps in epsvs:
        print('running: ' + str((np.arccos(src_tri[2][1] / p.psi), pr, eps)))
        I = lambda n_outer, n_rho, n_theta: adjacent_fixed(
            n_outer, p.K, obs_tri, src_tri, eps, 1.0, pr, n_rho, n_theta
        )
        res, last_orders = fixed_quad(I, p, last_orders)
        integrals.append(res)
    integrals = np.array(integrals)
    lim = take_limits(epsvs, integrals, 1, p.starting_eps)[0,0]
    print(lim)
    results[(p.starting_eps, p.n_eps)] = lim

    return take_limits(epsvs, integrals, 1, p.starting_eps)

results = dict()
def eval_integral(i, pt, p):
    for j in range(3):
        print("")
    print("starting integral: " + str(i))
    for j in range(3):
        print("")

    phihat, prhat = pt
    phi = to_interval(np.pi * 20 / 180., np.pi, phihat) # From 10 degrees to 180 degrees.
    pr = to_interval(0.0, 0.5, prhat)
    print("(phi, pr) = " + str((phi, pr)))
    Y = p.psi * np.cos(phi)
    Z = p.psi * np.sin(phi)

    obs_tri = [[0,0,0],[1,0,0],[0.5,p.psi,0]]
    src_tri = [[1,0,0],[0,0,0],[0.5,Y,Z]]
    return eval_tri_integral(obs_tri, src_tri, pr, p)

