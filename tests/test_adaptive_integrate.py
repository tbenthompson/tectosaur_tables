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

# TODO: This test can be removed when the old DenseIntegralOp formation routines are removed.
import tectosaur.geometry as geometry
from tectosaur.dense_integral_op import DenseIntegralOp
@slow
def test_coincident():
    K = 'H'
    eps = 0.08
    pts = np.array([[0,0,0],[1,0,0],[0.4,0.3,0]])
    eps_scale = np.sqrt(np.linalg.norm(geometry.tri_normal(pts)))
    tris = np.array([[0,1,2]])

    op = DenseIntegralOp(
        [eps, eps / 2], 17, 10, 13, 10, 10, 3.0,
        K, 1.0, 0.25, pts, tris, remove_sing = True
    )

    tol = 0.001
    res = coincident_integral(
        tol, K, pts[tris[0]].tolist(), 100, 0.001, eps, 2, eps_scale,
        1.0, 0.25, include_log = True
    )
    np.testing.assert_almost_equal(res, op.mat.reshape(81), 3)

# TODO: This test can be removed when the old DenseIntegralOp formation routines are removed.
@slow
def test_edge_adj():
    K = 'H'

    eps = 0.08
    pts = np.array([[0,0,0],[1,0,0],[0.5,0.5,0],[0,1,0],[0,-1,0],[0.5,-0.5,0]])
    tris = np.array([[0,1,2],[1,0,4]])
    eps_scale = np.sqrt(np.linalg.norm(geometry.tri_normal(pts[tris[0]])))
    op = DenseIntegralOp(
        [eps, eps / 2], 10, 15, 10, 10, 10, 3.0, K, 1.0, 0.25, pts, tris,
        remove_sing = True
    )

    res = adj_limit(
        K, pts[tris[0]].tolist(), pts[tris[1]].tolist(), 100, 0.001,
        eps, 2, eps_scale, 1.0, 0.25, include_log = True
    )
    np.testing.assert_almost_equal(res, op.mat[:9,9:].reshape(81), 4)

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
