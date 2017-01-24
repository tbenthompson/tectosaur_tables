import matplotlib.pyplot as plt
import numpy as np

from tectosaur.interpolate import to_interval
from tectosaur.table_lookup import coincident_interp_pts_wts

from tectosaur_tables.fixed_integrator import coincident_fixed
from build_tables import build_tables, safe_fixed_quad, TableParams

def make_coincident_params(K, tol, low_nq, check_quad_error, n_rho, n_theta,
        starting_eps, n_eps, n_A, n_B, n_pr):

    pts, wts = coincident_interp_pts_wts(n_A, n_B, n_pr)
    p = TableParams(
        K, tol, low_nq, check_quad_error, n_rho, n_theta,
        starting_eps, n_eps, (n_A, n_B, n_pr), pts, wts
    )
    p.filename = (
        '%s_%i_%f_%i_%f_%i_%i_%i_coincidenttable.npy' %
        (K, n_rho, starting_eps, n_eps, tol, n_A, n_B, n_pr)
    )
    return p

def eval(i, pt, p):
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

    integrals = []
    for eps in p.all_eps:
        print('running: ' + str((pt, eps)))
        I = lambda nq: coincident_fixed(nq, p.K, tri, eps, 1.0, pr, p.n_rho, p.n_theta)
        res = safe_fixed_quad(I, p)
        integrals.append(res)

    return np.array(integrals)

if __name__ == '__main__':
    # p = CoincidentParams("U", 1e-8, 80, 80, 80, 1e-4, 4, 8, 8, 8)
    # p = CoincidentParams("T", 1e-8, 80, 80, 80, 1e-4, 4, 8, 8, 8)
    # p = CoincidentParams("H", 1e-8, 80, 80, 80, 1e-4, 4, 12, 17, 9)

    p = make_coincident_params("H", 1e-3, 40, False, 50, 50, 1e-1, 2, 1, 1, 1)
    p.n_test_tris = 1
    build_tables(eval, p)
    plt.show()

    # for n_A, n_B, n_pr in [(25, 15, 12), (15, 25, 12), (25, 12, 12), (12, 25, 12)]:
    #     p = CoincidentParams("H", 1e-3, 40, False, 50, 50, 1e-1, 2, n_A, n_B, n_pr)
    #     build_tables(eval, p)
    #     plt.savefig(str(p.n_A) + '_' + str(p.n_B) + '_' + str(p.n_pr) + '.pdf')
    # plt.show()
