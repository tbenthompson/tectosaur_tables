import numpy as np

from tectosaur.interpolate import to_interval
from tectosaur.test_decorators import golden_master, slow

from tectosaur_tables.gpu_integrator import *
from tectosaur_tables.fixed_integrator import *

def test_fixed_coincident():
    # A,B,pr = 0.25,0.25,0.25
    # A = 0.094372304633
    # B = 0.233326931864
    # pr = 0.25
    A,B,pr = 0.0,0.1,0.11110744174509951
    for B in np.linspace(0.1, 1.0, 10):
        print("starting: " + str(B))
        # compare for a range of values of (a,b,nu)
        args = ['H', [[0, 0, 0], [1.0, 0, 0], [A, B, 0]], 0.0001, 1.0, pr, 80, 80]

        adaptive_res = coincident_integral(1e-6, *args)

        old = None
        diff = 0
        for i in range(4):
            nq = 25 * (2 ** i)
            new = coincident_fixed(nq, *args)
            if old is not None:
                diff = np.abs(old - new)[0]
                print(adaptive_res[0], new[0], adaptive_res[0] - new[0], diff)
            old = new

        np.testing.assert_almost_equal(adaptive_res, new)
        assert(diff < 1e-5)
