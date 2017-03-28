import numpy as np

from tectosaur.limit import limit
from tectosaur.geometry import element_pt

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

def normalize(vec):
    return vec / np.linalg.norm(vec)

def linear_basis_seg(xhat):
    return [1 - xhat, xhat]

def make_2d_integrator(obs_el, src_el, eps):
    obs_el = np.array(obs_el)
    src_el = np.array(src_el)
    obs_nx, obs_ny = unscaled_seg_normal(obs_el)
    src_nx, src_ny = unscaled_seg_normal(src_el)
    eps_offset_x, eps_offset_y = normalize([obs_nx, obs_ny]) * eps
    def f(x):
        obs_xhat = x[:,0]
        src_xhat = x[:,1]

        obs_basis = linear_basis_seg(obs_xhat)
        obs_pt = element_pt(obs_basis, obs_el)
        xx = obs_pt[0] + eps_offset_x
        xy = obs_pt[1] + eps_offset_y

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

        # Single layer
        kernel = np.log(np.sqrt(r2)) / (2 * np.pi)
        # Double layer
        # kernel = dotrm / (2 * np.pi * r2)
        # Hypersingular
        # kernel = ((-dotnm / r2) + (2 * dotrn * dotrm) / (r2 * r2)) / (2 * np.pi)

        out = np.empty((4, kernel.shape[0]))
        for i in range(2):
            for j in range(2):
                out[i * 2 + j] = obs_basis[i] * src_basis[j] * kernel
        return out.T
    return f

def laplace_integral(el, eps = 0.1):
    tol = 1e-4
    est = adaptive_integrate.integrate(
        make_2d_integrator(el, el, eps), [0, 0], [1, 1], tol
    )[0]
    return est

def laplace_tester(el, correct):
    np.testing.assert_almost_equal(laplace_integral(el), correct)

def test_2d_laplace_simplest():
    laplace_tester(
        [[0, 0], [1, 0]],
        [0.291354746782188, 0.0759048754363768, 0.075904875436377, 0.291354746782187]
    )

def test_2d_laplace_double_length():
    laplace_tester(
        [[0, 0], [2, 0]],
        [0.398599297267922, 0.0783850214314389, 0.0783850214314391, 0.398599297267922]
    )

def test_2d_laplace_rotated():
    laplace_tester(
        [[0, 0], [1, 1]],
        [0.34455624305342, 0.0774673541686122, 0.0774673541686119, 0.344556243053421]
    )

def test_2d_laplace_translated():
    laplace_tester(
        [[1, 1], [1, 2]],
        [0.291354746782187, 0.0759048754363769, 0.0759048754363769, 0.291354746782187]
    )

def test_2d_rotate():
    epsvs = 0.05 * (2.0 ** (-np.arange(6)))
    print(epsvs)
    theta = 0.0
    for scale in (2.0 ** np.arange(-1, 2)):
        vals = []
        for eps in epsvs:
            vals.append(laplace_integral(
                [[0, 0], [np.cos(theta) * scale, np.sin(theta) * scale]],
                eps = eps * scale
            ))
        # print(epsvs, vals)
        lim = limit(epsvs, vals, 1, False)
        print('')
        print(lim)
    # for theta in np.linspace(0, 2 * np.pi, 100)[:-1]:
    #     print(laplace_integral([[0, 0], [np.cos(theta), np.sin(theta)]]))

