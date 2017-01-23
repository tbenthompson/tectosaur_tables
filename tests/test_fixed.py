import numpy as np

from tectosaur.interpolate import to_interval
from tectosaur.test_decorators import golden_master, slow

from tectosaur_tables.gpu_integrator import *
from tectosaur_tables.fixed_integrator import *

def convergence(fnc, args, start, n_steps, correct):
    old = None
    diff = 0
    for i in range(n_steps):
        nq = start * (2 ** i)
        new = fnc(nq, *args)
        if old is not None:
            diff = np.abs(old - new)[0]
            print(correct, new[0], correct - new[0], diff)
        old = new
    return old, diff

def test_fixed_coincident():
    A,B,pr = 0.0,0.1,0.11110744174509951
    args = ['H', [[0, 0, 0], [1.0, 0, 0], [A, B, 0]], 0.0001, 1.0, pr, 80, 80]

    adaptive_res = coincident_integral(1e-6, *args)
    fixed_res, diff = convergence(coincident_fixed, args, 25, 4, adaptive_res[0])

    np.testing.assert_almost_equal(adaptive_res, fixed_res)
    assert(diff < 1e-5)

def test_fixed_adjacent():
    args = [
        'H', [[0, 0, 0], [1.2, 0, 0], [0.3, 1.1, 0]],
        [[1.2, 0, 0], [0, 0, 0], [0.3, -1.1, 0]],
        0.001, 1.0, 0.25, 50, 50
    ]
    adaptive_res = adjacent_integral(1e-4, *args)

    fixed_res, diff = convergence(adjacent_fixed, args, 25, 4, adaptive_res[0])
    np.testing.assert_almost_equal(adaptive_res, fixed_res)
    assert(diff < 1e-5)
