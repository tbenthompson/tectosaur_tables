import os
import numpy as np
import tectosaur.util.gpu as gpu
import tectosaur.quadrature as quad
from tectosaur.util.timer import Timer

import cppimport
adaptive_integrate = cppimport.imp('tectosaur_tables.adaptive_integrate').adaptive_integrate

float_type = np.float64
def make_gpu_integrator(type, K, obs_tri, src_tri, eps, sm, pr, rho_q, theta_q, flip_obsn, chunk):
    this_dir = os.path.dirname(os.path.realpath(__file__))

    module = gpu.load_gpu('kernels.cl', tmpl_dir = this_dir, tmpl_args = dict())
    fnc = getattr(module, type + '_integrals' + K)
    gpu_rho_qx = gpu.to_gpu(rho_q[0].flatten(), float_type)
    gpu_rho_qw = gpu.to_gpu(rho_q[1].flatten(), float_type)
    gpu_theta_qx = gpu.to_gpu(theta_q[0].flatten(), float_type)
    gpu_theta_qw = gpu.to_gpu(theta_q[1].flatten(), float_type)
    gpu_obs_tri = gpu.to_gpu(np.array(obs_tri).flatten(), float_type)
    gpu_src_tri = gpu.to_gpu(np.array(src_tri).flatten(), float_type)

    def integrand(x):
        # import matplotlib.pyplot as plt
        # plt.plot(x[:,0], x[:,1], '.')
        # plt.xlim([-0.1,1.1])
        # plt.ylim([-0.1,1.1])
        # plt.show()
        t = Timer(silent = True)
        n_x = x.shape[0]
        integrand.total_n_x += n_x
        # print(integrand.total_n_x)
        out = np.zeros((n_x,81))
        t.report("start")

        def call_integrator(start_idx, end_idx):
            n_items = end_idx - start_idx
            gpu_pts = gpu.to_gpu(x[start_idx:end_idx,:], float_type)
            gpu_result = gpu.empty_gpu((n_items, 81), float_type)
            fnc(
                gpu.gpu_queue, (n_items,), None,
                gpu_result.data, np.int32(chunk), gpu_pts.data,
                gpu_obs_tri.data, gpu_src_tri.data,
                np.int32(rho_q[0].shape[0]), gpu_rho_qx.data, gpu_rho_qw.data,
                np.int32(theta_q[0].shape[0]), gpu_theta_qx.data, gpu_theta_qw.data,
                float_type(eps), float_type(sm), float_type(pr),
                np.int32(flip_obsn)
            )
            out[start_idx:end_idx] = gpu_result.get()

        call_size = 2 ** 14
        next_call_start = 0
        next_call_end = call_size
        while next_call_end < n_x + call_size:
            this_call_end = min(next_call_end, n_x)
            call_integrator(next_call_start, this_call_end)
            next_call_start += call_size
            next_call_end += call_size
        t.report("call gpu")
        return out
    integrand.total_n_x = 0
    return integrand

def general_integral(tol, type, K, obs_tri, src_tri, eps, sm, pr, rho_order, theta_order):
    rho_gauss = quad.gaussxw(rho_order)
    rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
    theta_q = quad.gaussxw(theta_order)
    n_chunks = 3 if type == 'coincident' else 2
    res = np.zeros(81)
    for chunk in range(n_chunks):
        chunk_res = adaptive_integrate.integrate(
            make_gpu_integrator(
                type, K, obs_tri, src_tri, eps, sm, pr, rho_q, theta_q, chunk
            ),
            [0,0], [1,1], tol
        )
        # print(chunk_res[0][0])
        res += chunk_res[0]
    return np.array(res)

def coincident_integral(tol, K, tri, eps, sm, pr, rho_order, theta_order):
    return general_integral(tol, 'coincident', K, tri, tri, eps, sm, pr, rho_order, theta_order)

def adjacent_integral(tol, K, obs_tri, src_tri, eps, sm, pr, rho_order, theta_order):
    return general_integral(tol, 'adjacent', K, obs_tri, src_tri, eps, sm, pr, rho_order, theta_order)
