<%!
from tectosaur import source_dir
%>
<%
def dn(dim):
    return ['x', 'y', 'z'][dim]
import numpy as np
from tectosaur_tables.gpu_integrator import float_type as np_float_type
if np_float_type == np.float32:
    float_type = 'float'
else:
    float_type = 'double'
%>
#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define Real ${float_type}

<%namespace name="prim" file="../integral_primitives.cl"/>

<%def name="co_theta_low(chunk)">\
% if chunk == 0:
M_PI - atan2(1 - obsyhat, obsxhat);
% elif chunk == 1:
M_PI + atan2(obsyhat, obsxhat);
% elif chunk == 2:
-atan2(obsyhat, 1 - obsxhat);
% endif
</%def>

<%def name="co_theta_high(chunk)">\
% if chunk == 0:
M_PI + atan2(obsyhat, obsxhat);
% elif chunk == 1:
2 * M_PI - atan2(obsyhat, 1 - obsxhat);
% elif chunk == 2:
M_PI - atan2(1 - obsyhat, obsxhat);
% endif
</%def>

<%def name="co_rhohigh(chunk)">\
% if chunk == 0:
-obsxhat / cos(theta);
% elif chunk == 1:
-obsyhat / sin(theta);
% elif chunk == 2:
(1 - obsyhat - obsxhat) / (cos(theta) + sin(theta));
% endif
</%def>

<%def name="adj_theta_low(chunk)">\
% if chunk == 0:
0;
% elif chunk == 1:
M_PI - atan2(1, 1 - obsxhat);
% else:
0;
% endif
</%def>

<%def name="adj_theta_high(chunk)">\
% if chunk == 0:
M_PI - atan2(1, 1 - obsxhat);
% elif chunk == 1:
M_PI;
% else:
0;
% endif
</%def>

<%def name="adj_rhohigh(chunk)">\
% if chunk == 0:
obsxhat / (costheta + sintheta);
% elif chunk == 1:
-(1 - obsxhat) / costheta;
% else:
0;
% endif
</%def>

<%def name="func_def(type, k_name)">
__kernel
void ${type}_integrals${k_name}(__global Real* result, int chunk, 
    __global Real* pts, 
    __global Real* in_obs_tri, __global Real* in_src_tri,
    Real eps, Real G, Real nu)
</%def>

<%def name="zero_output()">
    Real sum[81];
    Real kahanC[81];
    for (int i = 0; i < 81; i++) {
        sum[i] = 0;
        kahanC[i] = 0;
    }
</%def>

<%def name="integral_setup(obs_tri_name, src_tri_name)">
    const int cell_idx = get_global_id(0);

    ${prim.constants()}
    Real obs_tri[3][3];
    Real src_tri[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            obs_tri[i][j] = ${obs_tri_name}[i * 3 + j];
            src_tri[i][j] = ${src_tri_name}[i * 3 + j];
        }
    }
    ${prim.tri_info("obs", "n")}
    ${prim.tri_info("src", "l")}
</%def>

<%def name="eval_hatvars()">
Real obsxhat = pts[cell_idx * 2];
Real obsyhat = pts[cell_idx * 2 + 1] * (1 - obsxhat);
</%def>

<%def name="rho_quad_eval()">
Real rhohat = (rho_qx[ri] + 1) / 2.0;
Real rho = rhohat * rhohigh;
Real jacobian = rho_qw[ri] * rho * rhohigh * outer_jacobian;
</%def>

<%def name="setup_kernel_inputs()">
    % for which, ptname in [("obs", "x_no_offset_"), ("src", "y")]:
        ${prim.basis(which + "")}
        ${prim.pts_from_basis(
            ptname, which + "",
            lambda b, d: which + "_tri[" + str(b) + "][" + str(d) + "]", 3
        )}
    % endfor

    % for dim in range(3):
        Real x${dn(dim)} = x_no_offset_${dn(dim)} - eps * n${dn(dim)};
    % endfor

    Real Dx = yx - xx;
    Real Dy = yy - xy; 
    Real Dz = yz - xz;
    Real r2 = Dx * Dx + Dy * Dy + Dz * Dz;
</%def>

<%def name="add_to_sum()">
    % for d_obs in range(3):
        % for d_src in range(3):
            {
                Real kernel_val = jacobian * K${d_obs}${d_src};
                % for b_obs in range(3):
                    % for b_src in range(3):
                        {
                            int idx = ${prim.temp_result_idx(d_obs, d_src, b_obs, b_src)};
                            Real add_to_sum = kernel_val * obsb${b_obs} * srcb${b_src};
                            Real y = add_to_sum - kahanC[idx];
                            Real t = sum[idx] + y;
                            kahanC[idx] = (t - sum[idx]) - y;
                            sum[idx] = t;
                        }
                    % endfor
                % endfor
            }
        % endfor
    % endfor
</%def>

<%def name="coincident_integrals(k_name)">
${func_def("coincident", k_name)}
{
    ${zero_output()}
    ${integral_setup("in_obs_tri", "in_src_tri")}
    ${eval_hatvars()}

    for (int oti = 0; oti < ${theta_q[0].shape[0]}; oti++) {
        Real thetahat = (theta_qx[oti] + 1) / 2;

        Real thetalow;
        Real thetahigh;
        if (chunk == 0) {
            thetalow = ${co_theta_low(0)}
            thetahigh = ${co_theta_high(0)}
        } else if (chunk == 1) {
            thetalow = ${co_theta_low(1)}
            thetahigh = ${co_theta_high(1)}
        } else {
            thetalow = ${co_theta_low(2)}
            thetahigh = ${co_theta_high(2)}
        }
        Real theta = (1 - thetahat) * thetalow + thetahat * thetahigh;

        Real outer_jacobian = 
            0.5 * theta_qw[oti] * (1 - obsxhat) * (thetahigh - thetalow);
        Real costheta = cos(theta);
        Real sintheta = sin(theta);

        Real rhohigh;
        if (chunk == 0) {
            rhohigh = ${co_rhohigh(0)}
        } else if (chunk == 1) {
            rhohigh = ${co_rhohigh(1)}
        } else {
            rhohigh = ${co_rhohigh(2)}
        }

        for (int ri = 0; ri < ${rho_q[0].shape[0]}; ri++) {
            ${rho_quad_eval()}

            Real srcxhat = obsxhat + rho * costheta;
            Real srcyhat = obsyhat + rho * sintheta;

            ${setup_kernel_inputs()}
            ${prim.tensor_kernels(k_name)}
            ${add_to_sum()}
        }
    }
    Real const_jacobian = 0.5 * obs_jacobian * src_jacobian;
    for (int i = 0; i < 81; i++) {
        result[cell_idx * 81 + i] = const_jacobian * (sum[i] + kahanC[i]);
    }
}
</%def>

<%def name="adjacent_integrals(k_name)">
${func_def("adjacent", k_name)}
{
    ${zero_output()}
    ${integral_setup("in_obs_tri", "in_src_tri")}
    ${eval_hatvars()}

    for (int oti = 0; oti < ${theta_q[0].shape[0]}; oti++) {
        Real thetahat = (theta_qx[oti] + 1) / 2;

        Real thetalow;
        Real thetahigh;
        if (chunk == 0) {
            thetalow = ${adj_theta_low(0)}
            thetahigh = ${adj_theta_high(0)}
        } else {
            thetalow = ${adj_theta_low(1)}
            thetahigh = ${adj_theta_high(1)}
        }
        Real theta = (1 - thetahat) * thetalow + thetahat * thetahigh;

        Real outer_jacobian = 
            0.5 * theta_qw[oti] * (1 - obsxhat) * (thetahigh - thetalow);
        Real costheta = cos(theta);
        Real sintheta = sin(theta);

        Real rhohigh;
        if (chunk == 0) {
            rhohigh = ${adj_rhohigh(0)}
        } else {
            rhohigh = ${adj_rhohigh(1)}
        }

        for (int ri = 0; ri < ${rho_q[0].shape[0]}; ri++) {
            ${rho_quad_eval()}

            Real srcxhat = rho * costheta + (1 - obsxhat);
            Real srcyhat = rho * sintheta;

            ${setup_kernel_inputs()}
            ${prim.tensor_kernels(k_name)}
            ${add_to_sum()}
        }
    }

    Real const_jacobian = 0.5 * obs_jacobian * src_jacobian;
    for (int i = 0; i < 81; i++) {
        result[cell_idx * 81 + i] = const_jacobian * sum[i];
    }
}
</%def>

__constant Real rho_qx[${rho_q[0].shape[0]}] = {
    % for x in rho_q[0][:-1]:
        ${x},
    % endfor
    ${rho_q[0][-1]}
};

__constant Real rho_qw[${rho_q[0].shape[0]}] = {
    % for w in rho_q[1][:-1]:
    ${w},
    % endfor
    ${rho_q[1][-1]}
};

__constant Real theta_qx[${theta_q[0].shape[0]}] = {
    % for x in theta_q[0][:-1]:
    ${x},
    % endfor
    ${theta_q[0][-1]}
};

__constant Real theta_qw[${theta_q[0].shape[0]}] = {
    % for w in theta_q[1][:-1]:
    ${w},
    % endfor
    ${theta_q[1][-1]}
};

${prim.geometry_fncs()}

% for k_name in ['U', 'T', 'A', 'H']:
${coincident_integrals(k_name)}
${adjacent_integrals(k_name)}
% endfor
