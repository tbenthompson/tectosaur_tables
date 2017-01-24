import numpy as np
import matplotlib.pyplot as plt

from tectosaur.interpolate import barycentric_evalnd
from tectosaur.limit import limit

class TableParams:
    def __init__(self, K, tol, low_nq, check_quad_error, n_rho, n_theta,
            starting_eps, n_eps, interp_params, pts, wts):

        self.K = K
        self.tol = tol
        self.low_nq = low_nq
        self.check_quad_error = check_quad_error
        self.n_rho = n_rho
        self.n_theta = n_theta
        self.starting_eps = starting_eps
        self.n_eps = n_eps
        self.interp_params = interp_params
        self.pts = pts
        self.wts = wts

        self.all_eps = starting_eps * 2.0 ** -np.arange(n_eps)

        # FOR A MIN ANGLE OF 10 DEGREES
        self.min_angle = 10
        self.psi = 0.5 * np.tan(np.deg2rad(self.min_angle))
        self.minlegalA = 0.0160559624778
        self.minlegalB = 0.0881595061826
        self.maxlegalB = 0.873575060826

        self.high_nq = low_nq * 2
        self.n_test_tris = 100

def safe_fixed_quad(I, p):
    res = I(p.low_nq)
    if p.check_quad_error:
        res_hi = I(p.high_nq)
        rel_err = np.abs((res_hi[0] - res[0]) / res_hi[0])
        print("quad error overestimate: " + str(rel_err))
        assert(rel_err < p.tol)
        res = res_hi
    return res

def take_limits(integrals, remove_divergence, all_eps):
    out = np.empty((81, 2))
    for i in range(81):
        out[i] = limit(all_eps, integrals[:, i], remove_divergence)
    return out

def test_f(results, eval_fnc, p):
    rand_pt = np.random.rand(p.pts.shape[1]) * 2 - 1.0
    correct = take_limits(eval_fnc(0, rand_pt, p), True, p.all_eps)[:,0]
    for i in range(1):
        interp = barycentric_evalnd(p.pts, p.wts, results[:,i,0], np.array([rand_pt]))[0]
        rel_err = np.abs((correct[i] - interp) / correct[i])
        abs_diff = correct[i] - interp
        print('testing: ' + str(i) + ' ' + str((correct[i], interp, rel_err, abs_diff)))
    return rel_err

def build_tables(eval_fnc, p, run_test = True):

    # results = []
    # for i, pt in enumerate(p.pts.tolist()):
    #     results.append(take_limits(eval_fnc(i, pt, p), True, p.all_eps))
    #     print("sample output: " + str(results[-1][0]))
    # results = np.array(results)

    # np.save(p.filename, results)

    results = np.load(p.filename)
    if run_test:
        np.random.seed(15)
        all = []
        for i in range(p.n_test_tris):
            all.append(test_f(results, eval_fnc, p))
        plt.figure()
        plt.hist(np.log10(np.abs(all)))
        plt.title(' '.join(map(str, p.interp_params)))
