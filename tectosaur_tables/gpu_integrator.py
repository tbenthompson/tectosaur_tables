import numpy as np
import tectosaur.util.gpu as gpu
import tectosaur.quadrature as quad

import cppimport
adaptive_integrate = cppimport.imp('tectosaur_tables.adaptive_integrate').adaptive_integrate

def make_gpu_integrator(type, K, obs_tri, src_tri, eps, sm, pr, rho_q, theta_q, chunk):
    module = gpu.ocl_load_gpu('tectosaur_tables/kernels.cl')
    fnc = getattr(module, type + '_integrals' + K)
    gpu_rqx, gpu_rqw = gpu.quad_to_gpu(rho_q)
    gpu_tqx, gpu_tqw = gpu.quad_to_gpu(theta_q)
    gpu_obs_tri = gpu.to_gpu(np.array(obs_tri).flatten())
    gpu_src_tri = gpu.to_gpu(np.array(src_tri).flatten())

    def integrand(x):
        n_x = x.shape[0]
        integrand.total_n_x += n_x
        # print(integrand.total_n_x)
        out = np.zeros((n_x,81))

        def call_integrator(start_idx, end_idx):
            n_items = end_idx - start_idx
            gpu_pts = gpu.to_gpu(x[start_idx:end_idx,:])
            gpu_result = gpu.empty_gpu((n_items, 81))
            fnc(
                gpu.ocl_gpu_queue, (n_items,), None,
                gpu_result.data, np.int32(chunk), gpu_pts.data,
                np.int32(rho_q[0].shape[0]), gpu_rqx.data, gpu_rqw.data,
                np.int32(theta_q[0].shape[0]), gpu_tqx.data, gpu_tqw.data,
                gpu_obs_tri.data, gpu_src_tri.data,
                gpu.float_type(eps), gpu.float_type(sm), gpu.float_type(pr)
            );
            out[start_idx:end_idx] = gpu_result.get()

        call_size = 2 ** 12
        next_call_start = 0
        next_call_end = call_size
        while next_call_end < n_x + call_size:
            this_call_end = min(next_call_end, n_x)
            call_integrator(next_call_start, this_call_end)
            next_call_start += call_size
            next_call_end += call_size
        return out
    integrand.total_n_x = 0
    return integrand

def general_integral(tol, type, K, obs_tri, src_tri, eps, sm, pr, rho_order, theta_order):
    rho_gauss = quad.gaussxw(50)
    rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
    theta_q = quad.gaussxw(50)
    n_chunks = 3 if type == 'coincident' else 2
    res = np.zeros(81)
    for chunk in range(n_chunks):
        chunk_res = adaptive_integrate.integrate(
            make_gpu_integrator(
                type, K, obs_tri, src_tri, eps, sm, pr, rho_q, theta_q, chunk
            ),
            [0,0], [1,1], tol
        )
        res += chunk_res[0]
    return np.array(res)

def coincident_integral(tol, K, tri, eps, sm, pr, rho_order, theta_order):
    return general_integral(tol, 'coincident', K, tri, tri, eps, sm, pr, rho_order, theta_order)

def adjacent_integral(tol, K, obs_tri, src_tri, eps, sm, pr, rho_order, theta_order):
    return general_integral(tol, 'adjacent', K, obs_tri, src_tri, eps, sm, pr, rho_order, theta_order)
