import matplotlib.pyplot as plt
import numpy as np

from tectosaur.interpolate import to_interval
from tectosaur.table_lookup import adjacent_interp_pts_wts

from tectosaur_tables.fixed_integrator import adjacent_fixed
from build_tables import build_tables, fixed_quad, TableParams
from tectosaur_tables.better_limit import limit


def make_adjacent_params(K, tol, low_nq, check_quad_error, adaptive_quad_error,
        n_rho, n_theta, starting_eps, n_eps, n_phi, n_pr, eps_step = 2.0):
    pts, wts = adjacent_interp_pts_wts(n_phi, n_pr)
    p = TableParams(
        K, tol, low_nq, check_quad_error, adaptive_quad_error, n_rho, n_theta,
        starting_eps, n_eps, (n_phi, n_pr), pts, wts,
        eps_step = eps_step
    )
    p.filename = (
        '%s_%i_%f_%i_%f_%i_%i_adjacenttable.npy' %
        (K, n_rho, starting_eps, n_eps, tol, n_phi, n_pr)
    )
    return p

max_lim_err = 0
def eval(i, pt, p):
    global max_lim_err
    for j in range(3):
        print("")
    print("starting integral: " + str(i))
    for j in range(3):
        print("")

    phihat, prhat = pt
    phi = to_interval(0, np.pi, phihat)
    pr = to_interval(0.0, 0.5, prhat)
    print("(phi, pr) = " + str((phi, pr)))
    Y = p.psi * np.cos(phi)
    Z = p.psi * np.sin(phi)

    tri1 = [[0,0,0],[1,0,0],[0.5,p.psi,0]]
    tri2 = [[1,0,0],[0,0,0],[0.5,Y,Z]]
    integrals = []
    Ls = []
    next_eps = p.starting_eps
    all_eps = []
    lim_err = 1000
    last_orders = None
    while lim_err > p.lim_tol:
        all_eps.append(next_eps)
        print('running: ' + str((pt, next_eps)))
        I = lambda n_outer, n_rho, n_theta: adjacent_fixed(
            n_outer, p.K, tri1, tri2, next_eps, 1.0, pr, n_rho, n_theta
        )
        res, last_orders = fixed_quad(I, p, last_orders)
        integrals.append(res)
        if len(integrals) > 1:
            lim = take_limits(all_eps, np.array(integrals), 1, p.starting_eps)[0,0]
            print("running limit: " + str(lim))
            if len(Ls) != 0:
                lim_err = np.abs((Ls[-1] - lim) / lim)
                print("lim err: " + str(lim_err))
            Ls.append(lim)
        next_eps /= p.eps_step
    max_lim_err = max(max_lim_err, lim_err)
    print("max lim err: " + str(max_lim_err))

    return take_limits(all_eps, np.array(integrals), 1)

def take_limits(all_eps, integrals, log_terms, max):
    out = np.empty((81, 2))
    for i in range(81):
        out[i] = limit(all_eps, integrals[:, i], log_terms, max)
    return out

results = dict()
def new_eval(i, pt, p):
    global max_lim_err, done
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

    tri1 = [[0,0,0],[1,0,0],[0.5,p.psi,0]]
    tri2 = [[1,0,0],[0,0,0],[0.5,Y,Z]]
    integrals = []

    from tectosaur.interpolate import cheb
    max = p.starting_eps
    n_pts = p.lim_tol

    # 0.1, 8    = -0.122455087187
    # 0.1, 10   = -0.123070835319
    # 0.1, 11   = -0.123181014707
    # 0.1, 12   = -0.123266167718
    # 0.1, 12T  = -0.123382314723
    # 0.1, 13T  = -0.123412152943
    # 0.1, 14T  = -0.123470818587
    # 0.1, 14TT = -0.123563030696
    # 0.1, 15TT = -0.123386545758
    # 0.1, 16TT = -0.123515235471
    # 0.1, 17TT = -0.123524329593

    epsvs = cheb(0, max, n_pts)
    epsvs.append((epsvs[-1] + epsvs[-2]) / 2)
    epsvs.append((epsvs[-1] + epsvs[-2]) / 2)
    epsvs.sort()
    epsvs = epsvs[::-1]

    integrals = []
    last_orders = None
    for eps in epsvs:
        print('running: ' + str((pt, eps)))
        I = lambda n_outer, n_rho, n_theta: adjacent_fixed(
            n_outer, p.K, tri1, tri2, eps, 1.0, pr, n_rho, n_theta
        )
        res, last_orders = fixed_quad(I, p, last_orders)
        integrals.append(res)
    integrals = np.array(integrals)
    lim = take_limits(epsvs, integrals, 1, max)[0,0]
    print(lim)
    results[(max, n_pts)] = lim

    return take_limits(epsvs, integrals, 1, max)


def final_table():
    p = make_adjacent_params('H', 1e-7, 50, True, True, 50, 50, 0.01, 200, 14, 6, eps_step = 1.5)
    p.n_test_tris = 100
    build_tables(new_eval, p)
    plt.savefig('adjacent_H_final_table_err.pdf')


if __name__ == '__main__':
    final_table()

    # def aitken(seq, its = 1):
    #     assert(its > 0)
    #     if its > 1:
    #         S = aitken(seq, its - 1)
    #     else:
    #         S = seq
    #     return S[2:] - ((S[2:] - S[1:-1]) ** 2 / ((S[2:] - S[1:-1]) - (S[1:-1] - S[:-2])))

    # vs = np.array([-0.11527428819709082, -0.11619966401248603, -0.11638986792026991, -0.11646070487737595, -0.1164949890546126, -0.11651476591045216, -0.116526746767, -0.116534185967, -0.116539735132, -0.116543628948])
    # print(aitken(vs))
    # print(aitken(vs, 2))
    # print(aitken(vs, 3))
    # print(aitken(vs, 4))
    # import sys;sys.exit()

    # start = [0.01]#[0.2, 0.1, 0.05, 0.025, 0.01]
    # n_interp = [200]#[6, 8, 10, 13, 16, 20]


    # # with err = 1e-7, result = array([-0.11653372, -0.11654198, -0.11654784, -0.11655055, -0.11655294])
    # # with err = 1e-8, result = array([-0.11653357, -0.11654182, -0.11654708, -0.11655049, -0.11655285])

    # for S in start:
    #     for N in n_interp:
    #         p = make_adjacent_params('H', 1e-7, 50, True, True, 50, 50, S, N, 1, 1, eps_step = 1.5)
    #         build_tables(new_eval, p, run_test = False)
    #         # plt.savefig('adjacent_H_final_table_err.pdf')
    # print(results)
    # print(results)
    # import pickle
    # with open('adjacent_grid_search.pkl', 'wb') as f:
    #     pickle.dump(results, f)

    # p = make_adjacent_params('H', 1e-6, 150, True, 100, 100, 1e-1 / 4, 10, 1, 1)
    # p = make_adjacent_params('H', 1e-6, 100, True, 150, 100, 1e-1 / 4, 10, 1, 1)

    # These parameters give 1e-9 error for the individual integrals
    # as tested by increasing each quadrature order and then comparing the values
    # p = make_adjacent_params('H', 1e-6, 200, False, 200, 200, 8e-1, 23, 1, 1, eps_step = np.sqrt(2))
    # p.n_test_tris = 0
    # build_tables(eval, p)
    # plt.show()

    # n_phi = 20
    # for n_pr in range(2, 12):
    #     p = make_adjacent_params('H', 1e-3, 40, True, 50, 50, 1e-1, 2, n_phi, n_pr)
    #     p.n_test_tris = 10
    #     build_tables(eval, p)
    #     plt.savefig('adj_' + str(n_phi) + '_' + str(n_pr) + '.pdf')
    #     # plt.show()
    # n_pr = 12
    # for n_phi in range(2, 30, 3):
    #     p = make_adjacent_params('H', 1e-3, 40, True, 50, 50, 1e-1, 2, n_phi, n_pr)
    #     p.n_test_tris = 10
    #     build_tables(eval, p)
    #     plt.savefig('adj_' + str(n_phi) + '_' + str(n_pr) + '.pdf')
    # plt.show()

# H parameters
# K = "H"
# tol = 1e-7
# rho_order = 70
# theta_order = 70
# starting_eps = 1e-5
# n_eps = 4
# n_pr = 8
# n_theta = 8

# play parameters
# K = "H"
# rho_order = 40
# theta_order = 28
# starting_eps = 1e-4
# n_eps = 2
# tol = 1e-6
# n_pr = 2
# n_theta = 2
