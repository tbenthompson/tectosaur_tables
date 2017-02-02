import matplotlib.pyplot as plt
import numpy as np

from tectosaur.interpolate import to_interval
from tectosaur.table_lookup import adjacent_interp_pts_wts

from tectosaur_tables.fixed_integrator import adjacent_fixed
from build_tables import build_tables, safe_fixed_quad, TableParams, take_limits


def make_adjacent_params(K, tol, low_nq, check_quad_error, n_rho, n_theta,
        starting_eps, n_eps, n_phi, n_pr):
    pts, wts = adjacent_interp_pts_wts(n_phi, n_pr)
    p = TableParams(
        K, tol, low_nq, check_quad_error, n_rho, n_theta,
        starting_eps, n_eps, (n_phi, n_pr), pts, wts
    )
    p.filename = (
        '%s_%i_%f_%i_%f_%i_%i_adjacenttable.npy' %
        (K, n_rho, starting_eps, n_eps, tol, n_phi, n_pr)
    )
    return p

def eval(i, pt, p):
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
    for eps in p.all_eps:
        print('running: ' + str((pt, eps)))
        I = lambda nq: adjacent_fixed(nq, p.K, tri1, tri2, eps, 1.0, pr, p.n_rho, p.n_theta)
        res = safe_fixed_quad(I, p)
        integrals.append(res)
        if len(integrals) > 1:
            lim = take_limits(np.array(integrals), True, p.all_eps[:len(integrals)])[0,0]
            print("running limit: " + str(lim))
            if len(integrals) > 2:
                err = np.abs((old_lim - lim) / lim)
                print("lim err: " + str(err))
            old_lim = lim

    return np.array(integrals)

if __name__ == '__main__':
    p = make_adjacent_params('H', 1e-6, 450, True, 250, 250, 1e-1 / 32, 8, 1, 1)
    p.n_test_tris = 0
    build_tables(eval, p)
    plt.show()


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
