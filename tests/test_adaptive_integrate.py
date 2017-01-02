import numpy as np
import scipy.integrate

from tectosaur.test_decorators import golden_master, slow

from tectosaur_tables.gpu_integrator import coincident_integral, adjacent_integral
import cppimport
adaptive_integrate = cppimport.imp('tectosaur_tables.adaptive_integrate').adaptive_integrate

def profile_integration():
    coincident_integral(
        1e-6, 'H',
        [[0, 0, 0], [1.2, 0, 0], [0.3, 1.1, 0]],
        0.001, 1.0, 0.25, 50, 50
    )

@slow
@golden_master
def test_coincident_integral():
    return coincident_integral(
        0.001, 'H',
        [[0, 0, 0], [1.2, 0, 0], [0.3, 1.1, 0]],
        0.01, 1.0, 0.25, 50, 50
    )

@slow
@golden_master
def test_adjacent_integral():
    return adjacent_integral(
        0.001, 'H',
        [[0, 0, 0], [1.2, 0, 0], [0.3, 1.1, 0]],
        [[1.2, 0, 0], [0, 0, 0], [0.3, -1.1, 0]],
        0.01, 1.0, 0.25, 50, 50
    )

def f1d(x):
    return np.sin(x)

def test_adaptive_1d():
    res = adaptive_integrate.integrate(f1d, [0], [1], 0.01)
    np.testing.assert_almost_equal(res[0], 1 - np.cos(1))

def f2d(x):
    return np.array([np.sin(x[:,0])*x[:,1]]).T

def test_adaptive_2d():
    res = adaptive_integrate.integrate(f2d, [0,0], [1,1], 0.01)
    np.testing.assert_almost_equal(res[0], np.sin(0.5) ** 2)

def vec_out(x):
    return np.array([x[:,0] + 1, x[:,0] - 1]).T

def test_adaptive_vector_out():
    res = adaptive_integrate.integrate(vec_out, [0], [1], 0.01)
    np.testing.assert_almost_equal(res[0], [1.5, -0.5])

if __name__ == '__main__':
    profile_integration()
