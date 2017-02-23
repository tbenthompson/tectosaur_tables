import numpy as np
import matplotlib.pyplot as plt

from tectosaur.interpolate import barycentric_evalnd
from tectosaur.limit import limit

class TableParams:
    def __init__(self, K, tol, low_nq, check_quad_error, adaptive_quad_error, n_rho, n_theta,
            starting_eps, lim_tol, interp_params, pts, wts, eps_step = 2.0):

        self.K = K
        self.tol = tol
        self.low_nq = low_nq
        self.check_quad_error = check_quad_error
        self.adaptive_quad_error = adaptive_quad_error
        self.n_rho = n_rho
        self.n_theta = n_theta
        self.starting_eps = starting_eps
        self.lim_tol = lim_tol
        self.eps_step = eps_step
        self.interp_params = interp_params
        self.pts = pts
        self.wts = wts

        # FOR A MIN ANGLE OF 10 DEGREES
        self.min_angle = 10
        self.psi = 0.5 * np.tan(np.deg2rad(self.min_angle))
        self.minlegalA = 0.0160559624778
        self.minlegalB = 0.0881595061826
        self.maxlegalB = 0.873575060826

        self.n_test_tris = 100

def fixed_quad(I, p, orders = None):
    if orders is None:
        orders = [p.low_nq, p.n_rho, p.n_theta]
    if p.check_quad_error:
        return safe_fixed_quad(I, orders, p.tol, p.adaptive_quad_error)
    else:
        return I(*orders), orders

def safe_fixed_quad(I, orders, tol = 0, adaptive = False):
    res = I(*orders)
    for i in range(len(orders)):
        new_orders = orders.copy()
        new_orders[i] += 1
        res_hi = I(*new_orders)
        rel_err = np.abs((res_hi[0] - res[0]) / res_hi[0])
        print("quad error estimate (" + ','.join([str(o) for o in new_orders]) + "): " + str(rel_err))
        if not adaptive:
            assert(rel_err < tol)
        else:
            if rel_err > tol:
                new_orders[i] = orders[i] * 2
                return safe_fixed_quad(I, new_orders, tol, adaptive)
    return res, orders

def take_limits(integrals, log_terms, all_eps):
    out = np.empty((81, 2))
    for i in range(81):
        out[i] = limit(all_eps, integrals[:, i], log_terms)
    return out

def test_f(results, eval_fnc, p):
    rand_pt = np.random.rand(p.pts.shape[1]) * 2 - 1.0
    correct = eval_fnc(0, rand_pt, p)
    for i in range(1):
        interp = barycentric_evalnd(p.pts, p.wts, results[:,i,0], np.array([rand_pt]))[0]
        rel_err = np.abs((correct[i] - interp) / correct[i])
        abs_diff = correct[i] - interp
        print('testing: ' + str(i) + ' ' + str((correct[i], interp, rel_err, abs_diff)))
    return rel_err

def build_tables(eval_fnc, p, run_test = True):
    results = []
    for i, pt in enumerate(p.pts.tolist()):
        results.append(eval_fnc(i, pt, p))
        print("sample output: " + str(results[-1][0]))
    results = np.array(results)

    np.save(p.filename, results)

    results = np.load(p.filename)
    if run_test:
        np.random.seed(15)
        all = []
        for i in range(p.n_test_tris):
            all.append(test_f(results, eval_fnc, p))
        plt.figure()
        plt.hist(np.log10(np.abs(all)))
        plt.title(' '.join(map(str, p.interp_params)))
