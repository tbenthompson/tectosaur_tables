import matplotlib.pyplot as plt
import numpy as np

from tectosaur.interpolate import to_interval
from tectosaur.table_lookup import coincident_interp_pts_wts

from tectosaur_tables.fixed_integrator import coincident_fixed
from tectosaur_tables.build_tables import build_tables, fixed_quad,\
    TableParams, take_limits, get_eps

def make_coincident_params(K, tol, low_nq, check_quad, adaptive_quad, n_rho, n_theta,
        starting_eps, n_eps, include_log, n_A, n_B, n_pr):

    pts, wts = coincident_interp_pts_wts(n_A, n_B, n_pr)
    p = TableParams(
        K, tol, low_nq, check_quad, adaptive_quad, n_rho, n_theta,
        starting_eps, n_eps, include_log, (n_A, n_B, n_pr), pts, wts
    )
    p.filename = (
        '%s_%i_%f_%i_%f_%i_%i_%i_coincidenttable.npy' %
        (K, n_rho, starting_eps, n_eps, tol, n_A, n_B, n_pr)
    )
    return p

def eval_tri_integral(tri, pr, p):
    epsvs = get_eps(p.n_eps, p.starting_eps)

    integrals = []
    last_orders = None
    for eps in epsvs:
        print('running: ' + str((tri[2][0], tri[2][1], pr, eps)))
        I = lambda n_outer, n_rho, n_theta: coincident_fixed(
            n_outer, p.K, tri, eps, 1.0, pr, n_rho, n_theta
        )
        res, last_orders = fixed_quad(I, p, last_orders)
        integrals.append(res)
    integrals = np.array(integrals)
    n_log_terms = 1 if p.include_log else 0
    lim = take_limits(epsvs, integrals, n_log_terms, p.starting_eps)
    print(lim[0,0])
    results[(p.starting_eps, p.n_eps)] = lim

    return take_limits(epsvs, integrals, n_log_terms, p.starting_eps)

results = dict()
max_lim_err = 0
def eval_integral(i, pt, p):
    global max_lim_err

    for j in range(3):
        print("")
    print("starting integral: " + str(i))
    for j in range(3):
        print("")

    Ahat,Bhat,prhat = pt
    # From symmetry and enforcing that the edge (0,0)-(1,0) is always longest,
    # A,B can be limited to (0,0.5)x(0,1).
    A = to_interval(p.minlegalA, 0.5, Ahat)
    B = to_interval(p.minlegalB, p.maxlegalB, Bhat)
    pr = to_interval(0.0, 0.5, prhat)

    tri = [[0,0,0],[1,0,0],[A,B,0.0]]

    return eval_tri_integral(tri, pr, p)
