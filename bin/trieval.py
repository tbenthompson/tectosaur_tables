import numpy as np
from tectosaur.interpolate import cheb
from tectosaur_tables.fixed_integrator import adjacent_fixed
from tectosaur_tables.gpu_integrator import adjacent_integral
from build_tables import fixed_quad
from tectosaur_tables.better_limit import limit


class Param:
    def __init__(self):
        self.check_quad_error = True
        self.adaptive_quad_error = True
        self.low_nq = 50
        self.n_rho = 50
        self.n_theta = 50
        self.tol = 1e-3

tri1 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.18198511713320004, 0.0]]
tri2 = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, -0.05911988056116087, 0.17211456237174882]]
pr = 0.42677669500000004
K = 'H'


eps_max = 0.01
log_terms = 1
epsvs = cheb(0, eps_max, 20)

last_orders = None
integrals = []
p = Param()
for eps in epsvs:
    print(eps)
    I = lambda n_outer, n_rho, n_theta: adjacent_fixed(
        n_outer, K, tri1, tri2, eps, 1.0, pr, n_rho, n_theta
    )
    res, last_orders = fixed_quad(I, p, last_orders)
    # res = adjacent_integral(p.tol, K, tri1, tri2, eps, 1.0, pr, p.n_rho, p.n_theta)
    integrals.append(res)
integrals = np.array(integrals)
print(limit(epsvs, integrals[:, 0], log_terms, eps_max))
# print(res.reshape((3,3,3,3)))
