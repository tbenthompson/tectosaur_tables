import numpy as np
from tectosaur_tables.better_limit import *
from tectosaur.interpolate import cheb

def test_easy():
    f = lambda x: 1.2 + x ** 2
    xs = np.array(cheb(0, 1, 5))
    fvs = f(xs)
    out = limit(xs, fvs, 0, 1)
    np.testing.assert_almost_equal(out, [1.2, 0])

def test_logarithm():
    f = lambda x: 1.2 + x ** 2 + np.log(x)
    xs = np.array(cheb(0, 0.2, 5))
    fvs = f(xs)
    out = limit(xs, fvs, 1, 0.2)
    np.testing.assert_almost_equal(out, [1.2, 1.0])
