import matplotlib.pyplot as plt
import numpy as np

import tectosaur.quadrature as quad
import tectosaur.geometry as geometry
from tectosaur.interpolate import cheb, cheb_wts, barycentric_evalnd, to_interval
from tectosaur.limit import limit, richardson_limit
from tectosaur.util.timer import Timer
from tectosaur.table_lookup import coincident_interp_pts_wts

from tectosaur_tables.gpu_integrator import coincident_integral, adjacent_integral
from tectosaur_tables.fixed_integrator import coincident_fixed


class BuildParams:
    def __init__(self, K, tol, low_nq, check_quad_error, n_rho, n_theta,
            starting_eps, n_eps, n_A, n_B, n_pr):

        self.K = K
        self.tol = tol
        self.low_nq = low_nq
        self.check_quad_error = check_quad_error
        self.n_rho = n_rho
        self.n_theta = n_theta
        self.starting_eps = starting_eps
        self.n_eps = n_eps
        self.n_A = n_A
        self.n_B = n_B
        self.n_pr = n_pr

        self.filename = (
            '%s_%i_%f_%i_%f_%i_%i_%i_coincidenttable.npy' %
            (K, n_rho, starting_eps, n_eps, tol, n_A, n_B, n_pr)
        )
        print(self.filename)
        self.all_eps = starting_eps * 2.0 ** -np.arange(n_eps)


        # FOR A MIN ANGLE OF 10 DEGREES
        self.minlegalA = 0.0160559624778
        self.minlegalB = 0.0881595061826
        self.maxlegalB = 0.873575060826

        self.high_nq = low_nq * 2

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
    t = Timer()
    for eps in p.all_eps:
        print('running: ' + str((pt, eps)))
        I = lambda nq: coincident_fixed(nq, p.K, tri, eps, 1.0, pr, p.n_rho, p.n_theta)
        res = I(p.low_nq)
        if p.check_quad_error:
            res_hi = I(p.high_nq)
            rel_err = np.abs((res_hi[0] - res[0]) / res_hi[0])
            print(rel_err)
            assert(rel_err < p.tol)
            res = res_hi
        print(res[0])
        t.report(str((pt, eps)))
        integrals.append(res)
        if len(integrals) > 1:
            print("limit: " + str(take_limits(np.array(integrals), True, p.all_eps[:len(integrals)])[0,:]))
    integrals = np.array(integrals)
    lim = take_limits(integrals, True, p.all_eps)
    return lim

def take_limits(integrals, remove_divergence, all_eps):
    out = np.empty((81, 2))
    for i in range(81):
        out[i] = limit(all_eps, integrals[:, i], remove_divergence)
    return out

def test_f(results, eval_fnc, pts, wts, p):
    rand_pt = np.random.rand(pts.shape[1]) * 2 - 1.0
    correct = eval_fnc(0, rand_pt, p)[:,0]
    for i in range(1):
        interp = barycentric_evalnd(pts, wts, results[:,i,0], np.array([rand_pt]))[0]
        rel_err = np.abs((correct[i] - interp) / correct[i])
        abs_diff = correct[i] - interp
        print('testing: ' + str(i) + ' ' + str((correct[i], interp, rel_err, abs_diff)))
    return rel_err

def build_tables(eval_fnc, p, run_test = True):
    pts, wts = coincident_interp_pts_wts(p.n_A, p.n_B, p.n_pr)

    results = np.array([eval_fnc(i, pt, p) for i, pt in enumerate(pts.tolist())])
    np.save(p.filename, results)
    results = np.load(p.filename)
    if run_test:
        np.random.seed(15)
        all = []
        for i in range(30):
            all.append(test_f(results, eval_fnc, pts, wts, p))
        plt.figure()
        plt.hist(np.log10(np.abs(all)))
        plt.title(str(p.n_A) + '_' + str(p.n_B) + '_' + str(p.n_pr))

if __name__ == '__main__':
    # p = BuildParams("U", 1e-8, 80, 80, 80, 1e-4, 4, 8, 8, 8)
    # p = BuildParams("T", 1e-8, 80, 80, 80, 1e-4, 4, 8, 8, 8)
    # p = BuildParams("H", 1e-8, 80, 80, 80, 1e-4, 4, 8, 8, 8)

    # for n_interp in [10, 12]:
    #     p = BuildParams("H", 1e-3, 30, 50, 50, 1e-1, 2, n_interp, n_interp, n_interp)
    #     build_tables(eval, p)

    # p = BuildParams("H", 1e-1, 250, False, 100, 100, 10 ** -1, 12, 1, 1, 1)
    for n_A, n_B, n_pr in [(25, 25, 5)]:
        p = BuildParams("H", 1e-3, 40, False, 50, 50, 1e-1, 2, n_A, n_B, n_pr)
        build_tables(eval, p)
    plt.show()
