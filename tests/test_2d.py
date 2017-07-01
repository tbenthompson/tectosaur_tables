import numpy as np

from tectosaur_tables.better_limit import limit
from tectosaur.util.geometry import element_pt

import cppimport
adaptive_integrate = cppimport.imp('tectosaur_tables.adaptive_integrate').adaptive_integrate

# Plan for 2d integrals:
# -- coincident:
#
# -- adjacent:
#       cut out a pair of standard-sized segments from the adjacent portion.
#       lookup standard size from doubling table
#       also interpolate on intersection angle (and poisson ratio)

def unscaled_seg_normal(corners):
    return np.array([-corners[1][1] + corners[0][1], corners[1][0] - corners[0][0]])

def el_jacobian(corners):
    return np.sqrt((corners[1][0] - corners[0][0]) ** 2 + (corners[1][1] - corners[0][1]) ** 2)

def normalize(vec):
    return vec / np.linalg.norm(vec)

def linear_basis_seg(xhat):
    return [1 - xhat, xhat]

def make_2d_integrator(obs_el, src_el, eps, K):
    obs_el = np.array(obs_el)
    src_el = np.array(src_el)
    obs_nx, obs_ny = unscaled_seg_normal(obs_el)
    src_nx, src_ny = unscaled_seg_normal(src_el)
    obs_L = el_jacobian(obs_el)
    src_L = el_jacobian(src_el)
    eps_offset_x, eps_offset_y = normalize([obs_nx, obs_ny]) * eps
    def f(x):
        obs_xhat = x[:,0]
        src_xhat = x[:,1]

        obs_basis = linear_basis_seg(obs_xhat)
        obs_pt = element_pt(obs_basis, obs_el)
        xx = obs_pt[0] - eps_offset_x
        xy = obs_pt[1] - eps_offset_y

        src_basis = linear_basis_seg(src_xhat)
        src_pt = element_pt(src_basis, src_el)
        yx = src_pt[0]
        yy = src_pt[1]

        rx = xx - yx
        ry = xy - yy
        r2 = rx ** 2 + ry ** 2

        dotnm = obs_nx * src_nx + obs_ny * src_ny
        dotrn = obs_nx * rx + obs_ny * ry
        dotrm = src_nx * rx + src_ny * ry

        if K == 'S':
            # Single layer
            kernel = np.log(np.sqrt(r2)) / (2 * np.pi)
        elif K == 'D':
            # Double layer
            kernel = dotrm / (2 * np.pi * r2)
        elif K == 'H':
            # Hypersingular
            kernel = ((dotnm / r2) - (2 * dotrn * dotrm) / (r2 * r2)) / (2 * np.pi)
        else:
            kernel = (np.log(K) / (2 * np.pi)) * np.ones_like(r2)

        out = np.empty((4, kernel.shape[0]))
        for i in range(2):
            for j in range(2):
                out[i * 2 + j] = obs_L * src_L * kernel * obs_basis[i] * src_basis[j]
        return out.T
    return f

def laplace_integral(el, K, eps = 0.1):
    tol = 1e-5
    est = adaptive_integrate.integrate(
        make_2d_integrator(el, el, eps, K), [0, 0], [1, 1], tol
    )[0]
    return np.array(est)

def laplace_limit(el, K, epsvs, n_log_terms):
    vs = np.array([laplace_integral(el, K, eps) for eps in epsvs])
    return np.array([
        limit(epsvs, vs[:, i], n_log_terms, np.max(epsvs))[0]
        for i in range(vs.shape[1])
    ])

def laplace_tester(el, K, correct):
    np.testing.assert_almost_equal(laplace_integral(el, K), correct)

def test_2d_laplace_simplest():
    laplace_tester(
        [[0, 0], [1, 0]], 'H',
        [-0.291354746782188, -0.0759048754363768, -0.075904875436377, -0.291354746782187]
    )

def test_2d_laplace_double_length():
    laplace_tester(
        [[0, 0], [2, 0]], 'H',
        4 * np.array([-0.398599297267922, -0.0783850214314389, -0.0783850214314391, -0.398599297267922])
    )

def test_2d_laplace_rotated():
    laplace_tester(
        [[0, 0], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]], 'H',
        [-0.291354746782188, -0.0759048754363768, -0.075904875436377, -0.291354746782187]
    )

def test_2d_laplace_translated():
    laplace_tester(
        [[1, 1], [1, 2]], 'H',
        [-0.291354746782188, -0.0759048754363768, -0.075904875436377, -0.291354746782187]
    )

def test_2d_single_layer_scale():
    scale = np.random.rand()
    eps = 0.1

    el1 = np.array([[0,0],[1,0]])
    el2 = scale * el1

    for K in ['S', 'D', 'H']:
        a = laplace_integral(el1, K, eps)
        b = laplace_integral(el2, K, eps * scale)
        if K == 'S':
            c = laplace_integral(el1, scale)
            e = scale ** 2 * (a + c)
        elif K == 'D':
            e = scale ** 2 * a
        elif K == 'H':
            e = scale ** 2 * a
        np.testing.assert_almost_equal(b, e)
