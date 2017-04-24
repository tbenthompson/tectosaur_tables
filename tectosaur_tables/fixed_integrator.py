import numpy as np

import tectosaur.quadrature as quad
from tectosaur_tables.gpu_integrator import make_gpu_integrator

def outer_quad(eps, n_outer_sing, n_outer_smooth, chunk):
    ad_order = 3
    q1 = quad.aimi_diligenti(quad.gaussxw(n_outer_smooth), ad_order, ad_order)
    q2 = quad.aimi_diligenti(quad.gaussxw(n_outer_smooth), ad_order, ad_order)
    pts = []
    wts = []
    qx = quad.map_to(q1, [0, 1])
    for ox, wx in zip(*qx):
        qy = quad.map_to(q2, [0, 1])
        for oy, wy in zip(*qy):
            pts.append([ox, oy])
            wts.append(wx * wy)

    return np.array(pts), np.array(wts)

def general_fixed(outer_order, type, K, obs_tri, src_tri,
        eps, sm, pr, rho_order, theta_order, flip_obsn):
    rho_gauss = quad.gaussxw(rho_order)
    rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
    theta_q = quad.gaussxw(theta_order)
    n_chunks = 3 if type == 'coincident' else 2
    res = np.zeros(81)
    for chunk in range(n_chunks):
        q_rule = outer_quad(eps, outer_order, outer_order, chunk)
        integrator = make_gpu_integrator(
            type, K, obs_tri, src_tri, eps, sm, pr, rho_q, theta_q, flip_obsn, chunk
        )
        chunk_res_pts = integrator(q_rule[0])
        chunk_res = np.sum(chunk_res_pts * q_rule[1][:,np.newaxis], axis = 0)
        res += chunk_res
    return np.array(res)

def coincident_fixed(outer_order, K, tri, eps, sm, pr, rho_order, theta_order):
    return general_fixed(
        outer_order, 'coincident', K, tri, tri, eps, sm, pr, rho_order, theta_order, False
    )

def adjacent_fixed(outer_order, K, obs_tri, src_tri, eps,
        sm, pr, rho_order, theta_order, flip_obsn):
    return general_fixed(
        outer_order, 'adjacent', K, obs_tri, src_tri,
        eps, sm, pr, rho_order, theta_order, flip_obsn
    )

