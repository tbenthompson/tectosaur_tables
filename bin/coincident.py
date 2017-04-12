import matplotlib.pyplot as plt
import numpy as np

from tectosaur.interpolate import to_interval
from tectosaur.table_lookup import coincident_interp_pts_wts

from tectosaur_tables.fixed_integrator import coincident_fixed
from build_tables import build_tables, fixed_quad, TableParams, take_limits, get_eps

def make_coincident_params(K, tol, low_nq, check_quad, adaptive_quad, n_rho, n_theta,
        starting_eps, n_eps, n_A, n_B, n_pr):

    pts, wts = coincident_interp_pts_wts(n_A, n_B, n_pr)
    p = TableParams(
        K, tol, low_nq, check_quad, adaptive_quad, n_rho, n_theta,
        starting_eps, n_eps, (n_A, n_B, n_pr), pts, wts
    )
    p.filename = (
        '%s_%i_%f_%i_%f_%i_%i_%i_coincidenttable.npy' %
        (K, n_rho, starting_eps, n_eps, tol, n_A, n_B, n_pr)
    )
    return p


results = dict()
max_lim_err = 0
def eval(i, pt, p):
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

    epsvs = get_eps(p.n_eps, p.starting_eps)

    integrals = []
    last_orders = None
    for eps in epsvs:
        print('running: ' + str((pt, eps)))
        I = lambda n_outer, n_rho, n_theta: coincident_fixed(
            n_outer, p.K, tri, eps, 1.0, pr, n_rho, n_theta
        )
        res, last_orders = fixed_quad(I, p, last_orders)
        integrals.append(res)
    integrals = np.array(integrals)
    lim = take_limits(epsvs, integrals, 1, p.starting_eps)
    print(lim[0,0])
    results[(p.starting_eps, p.n_eps)] = lim

    return take_limits(epsvs, integrals, 1, p.starting_eps)

def final_table():
    p = make_coincident_params("H", 1e-6, 200, False, 100, 180, 1e-1 / 32, 6, 12, 17, 9)
    p.n_test_tris = 100
    build_tables(eval, p)
    plt.savefig('final_table_err.pdf')

if __name__ == '__main__':
    # final_table()

    p = make_coincident_params("H", 1e-5, 25, True, True, 25, 25, 0.01, 10, 12, 17, 9)
    p.n_test_tris = 0
    build_tables(eval, p)

    # p = CoincidentParams("U", 1e-8, 80, 80, 80, 1e-4, 4, 8, 8, 8)
    # p = CoincidentParams("T", 1e-8, 80, 80, 80, 1e-4, 4, 8, 8, 8)
    # p = CoincidentParams("H", 1e-8, 80, 80, 80, 1e-4, 4, 12, 17, 9)

    # steps = 6
    # size = np.exp(5 * np.log(2) / (steps - 1))
    # print(size)
    # p = make_coincident_params("H", 1e-6, 150, False, 76, 75, 1e-1 / 32, steps, 1, 1, 1, size)
    # p.n_test_tris = 0
    # build_tables(eval, p)
    # plt.show()

    # for n_A, n_B, n_pr in [(25, 15, 12), (15, 25, 12), (25, 12, 12), (12, 25, 12)]:
    #     p = CoincidentParams("H", 1e-3, 40, False, 50, 50, 1e-1, 2, n_A, n_B, n_pr)
    #     build_tables(eval, p)
    #     plt.savefig(str(p.n_A) + '_' + str(p.n_B) + '_' + str(p.n_pr) + '.pdf')
    # plt.show()
