import matplotlib.pyplot as plt
import numpy as np

from tectosaur.interpolate import to_interval
from tectosaur.table_lookup import coincident_interp_pts_wts

from tectosaur_tables.fixed_integrator import coincident_fixed
from build_tables import build_tables, safe_fixed_quad, TableParams, take_limits

def make_coincident_params(K, tol, low_nq, check_quad_error, n_rho, n_theta,
        starting_eps, n_eps, n_A, n_B, n_pr, eps_step = 2.0):

    pts, wts = coincident_interp_pts_wts(n_A, n_B, n_pr)
    p = TableParams(
        K, tol, low_nq, check_quad_error, n_rho, n_theta,
        starting_eps, n_eps, (n_A, n_B, n_pr), pts, wts,
        eps_step = eps_step
    )
    p.filename = (
        '%s_%i_%f_%i_%f_%i_%i_%i_coincidenttable.npy' %
        (K, n_rho, starting_eps, n_eps, tol, n_A, n_B, n_pr)
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

    Ahat,Bhat,prhat = pt
    # From symmetry and enforcing that the edge (0,0)-(1,0) is always longest,
    # A,B can be limited to (0,0.5)x(0,1).
    A = to_interval(p.minlegalA, 0.5, Ahat)
    B = to_interval(p.minlegalB, p.maxlegalB, Bhat)
    pr = to_interval(0.0, 0.5, prhat)

    tri = [[0,0,0],[1,0,0],[A,B,0.0]]

    integrals = []
    old_lim = 0
    for eps in p.all_eps:
        print('running: ' + str((pt, eps)))
        I = lambda nq: coincident_fixed(nq, p.K, tri, eps, 1.0, pr, p.n_rho, p.n_theta)
        res = safe_fixed_quad(I, p)
        integrals.append(res)
        if len(integrals) > 1:
            lim = take_limits(np.array(integrals), True, p.all_eps[:len(integrals)])[0,0]
            print("running limit: " + str(lim))
            if len(integrals) > 2:
                lim_err = np.abs((old_lim - lim) / lim)
                print("lim err: " + str(lim_err))
            old_lim = lim

    max_lim_err = max(lim_err, max_lim_err)
    print("runing max lim err: " + str(max_lim_err))

    return np.array(integrals)

def final_table():
    p = make_coincident_params("H", 1e-6, 200, False, 100, 180, 1e-1 / 32, 6, 12, 17, 9)
    p.n_test_tris = 100
    build_tables(eval, p)
    plt.savefig('final_table_err.pdf')

if __name__ == '__main__':
    final_table()

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
