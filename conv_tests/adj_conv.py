import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

from tectosaur.limit import *

# Thought about using an interpolation method that handles endpoint singularities.
# But, generally, these methods are intended for evaluating the function *away* from
# those singularities, whereas I am interested in evaluating the function *at* the
# singularity, with the singularity removed.


def ratinterp(x, f, order):
    rhs = f
    lhs_1 = np.array([x ** i for i in range(order)]).T
    lhs_2 = -lhs_1[:,1:] * f[:, np.newaxis]
    lhs = np.hstack((lhs_1, lhs_2))
    # print(np.linalg.cond(lhs))
    coeffs = np.linalg.lstsq(lhs, rhs)
    As = coeffs[0][:order]
    Bs = coeffs[0][order:]
    return As, Bs

def rateval(x, coeffs):
    pows = np.array([x ** i for i in range(coeffs[0].shape[0])])
    numer = np.sum(pows * coeffs[0][:, np.newaxis], axis = 0)
    denom = 1 + np.sum(pows[1:,:] * coeffs[1][:, np.newaxis], axis = 0)
    return numer / denom

def ratinterp_with_log(x, f, order):
    pows = np.array([x ** i for i in range(order)])

    log_start_coeff = limit_coeffs(x, f, 1)[0][0]
    p0 = [log_start_coeff] + [0] * (order * 2 - 1)
    def obj(param):
        rat_numer = np.sum(pows * param[1:(order + 1)][:, np.newaxis], axis = 0)
        rat_denom = 1 + np.sum(pows[1:] * param[(order + 1):][:, np.newaxis], axis = 0)
        lhs = rat_numer / rat_denom + param[0] * np.log(x)
        diff = lhs - f
        return np.sum(diff ** 2)

    res = scipy.optimize.minimize(obj, p0)
    log_coeff = res.x[0]
    print(log_start_coeff, log_coeff)
    return log_coeff, res.x[1:(order + 1)], res.x[(order + 1):]

def rateval_with_log(x, coeffs):
    pows = np.array([x ** i for i in range(coeffs[1].shape[0])])
    numer = np.sum(pows * coeffs[1][:, np.newaxis], axis = 0)
    denom = 1 + np.sum(pows[1:,:] * coeffs[2][:, np.newaxis], axis = 0)
    return numer / denom + coeffs[0] * np.log(x)



data = [4.808509588401806e-07, 2.859074232873816e-06, 1.1925641395873547e-05, 3.988211942814987e-05, 0.00011449083571240847, 0.0002934196140107116, 0.0006827105644366299, 0.0014456945537162772, 0.00278183480335222, 0.004875127130823552, 0.007844460727703402, 0.011726055058313504, 0.016486447672913615, 0.022047431313778138, 0.02830902562671784, 0.03516577759095902, 0.04251670750169806, 0.050270601813516946, 0.05834825447763689, 0.0666828360941662, 0.07521919274840261, 0.08391260074644313, 0.09272731288525342]
eps = [0.8, 0.565685424949238, 0.39999999999999997, 0.28284271247461895, 0.19999999999999996, 0.14142135623730945, 0.09999999999999996, 0.07071067811865472, 0.049999999999999975, 0.035355339059327355, 0.024999999999999984, 0.017677669529663678, 0.01249999999999999, 0.008838834764831837, 0.006249999999999994, 0.004419417382415918, 0.0031249999999999967, 0.0022097086912079584, 0.0015624999999999981, 0.001104854345603979, 0.000781249999999999, 0.0005524271728019895, 0.0003906249999999994]
data = np.array(data)
eps = np.array(eps)
def main():
    f = lambda x: (0.2 - x / 10.0) / (1.0 + x / 10.0) + 3.0 * np.log(x)
    fvs = f(eps)
    ratinterp_with_log(eps, fvs, 5)

    first = 10
    last = 18
    step = 1
    test_first = first
    test_last = last + 4
    for i in range(1, 6):
        I = ratinterp_with_log(eps[first:last], data[first:last], i)
        I_data = rateval_with_log(eps[test_first:test_last], I)
        print(
            "rational interp order=" + str(i) + ": " +
            str(np.max(np.abs(data[test_first:test_last] - I_data)))
        )

# def main():
#     lim = -0.11656244
#     factor = -0.02652583
#     # data = data - factor * np.log(eps)
#
#     first = 10
#     last = 18
#     step = 1
#     test_first = first
#     test_last = last + 4
#     print(8e-1 / (np.sqrt(2) ** 10), eps[last - 1])
#
#     prev_val = None
#     for i in range(last - first - 1):
#         I_old = limit_coeffs(eps[first:last:step], data[first:last:step], i)
#         I_old_data = limit_interp_eval(eps[test_first:test_last], *I_old)
#         I_limit = I_old[0][i]
#         # print(I_limit)
#         if prev_val is not None:
#             print(I_limit, np.abs((I_limit - prev_val) / I_limit))
#         prev_val = I_limit
#         print("poly+log" + str(i) + " interp: " + str(np.max(np.abs(data[test_first:test_last] - I_old_data))))
#
#     for i in range(2, 12):
#         I = ratinterp(np.log10(eps)[first:last:step], data[first:last:step], i)
#         I_data = rateval(np.log10(eps[test_first:test_last]), I)
#         print(
#             "rational interp order=" + str(i) + ": " +
#             str(np.max(np.abs(data[test_first:test_last] - I_data)))
#         )
        # plt.plot(np.log10(eps[test_first:test_last]), data[test_first:test_last], 'b-')
        # plt.plot(np.log10(eps[test_first:test_last]), I_data, 'r-')
        # plt.plot(np.log10(eps[test_first:test_last]), I_old_data, 'k-')
        # plt.show()

    # plt.plot(eps, data)
    # plt.show()

    # plt.figure(dpi = 300)
    # plt.plot(np.log10(eps), data - factor * np.log(eps), 'b-')
    # plt.plot(np.log10(eps), lim + (eps ** 0.5) * 0.20, 'r-')
    # plt.plot(np.log10(eps), lim + (eps ** 0.3) * 0.12, 'g-')
    # plt.plot(np.log10(eps), lim + (eps ** 0.6) * 0.12, 'k-')
    # plt.show()

    # plt.plot(np.log10(eps), data)
    # plt.show()

    # plt.plot(np.log10(eps), data - factor * np.log(eps))
    # plt.show()



if __name__ == '__main__':
    main()

# def main():
#     starting_eps = 0.1
#     eps_step = 2.0
#     data = [
#         0.00068271,0.00278183,0.00784446,0.01648645,0.02830903,
#         0.04251671,0.05834825,0.07521919,0.09272731,0.11061386,
#         0.12871985,0.14695075,0.16525172,0.18359154,0.20195269,
#         0.22032546,0.23870452,0.25708696,0.27547122
#     ]
#     eps = starting_eps * eps_step ** -np.arange(len(data))
#
#     all_res1 = []
#     all_res2 = []
#     for i in range(2, 19):
#         res1 = limit(eps[:i], data[:i], True)
#         print(res1)
#         all_res1.append(res1[0])
#
#         res2 = limit2(eps[:i], data[:i], i // 2)
#         print(res2)
#         all_res2.append(res2[0])
#     all_res1 = np.array(all_res1)
#     all_res2 = np.array(all_res2)
#     err1 = np.abs((all_res1[1:] - all_res1[:-1]) / all_res1[1:])
#     err2 = np.abs((all_res2[1:] - all_res2[:-1]) / all_res2[1:])
#
#     print(err1)
#     print(err2)
#     print(err1[1:] / err1[:-1])
#     print(err2[1:] / err2[:-1])
#
    # lim_seq = []
    # for i in range(2, len(data)):
    #     new_lim = limit(eps[:i], data[:i], True)

    #     print("val: " + str(data[i]))
    #     print("new lim: " + str(new_lim))
    #     if len(lim_seq) > 0:
    #         print("err: " + str(np.abs((new_lim[0] - lim_seq[-1]) / new_lim[0])))
    #     print("")

    #     lim_seq.append(new_lim[0])
    # lim_seq = np.array(lim_seq)


    # print(lim_seq)
    # best = -0.116526095502#aitken(lim_seq)[-1]
    # print(best)
    # print(aitken(lim_seq) - best)

#
#     log_factor = limit(eps, data, True)[1]
#     plt.plot(np.log10(eps), data - np.log(eps) * log_factor)
#     plt.show()
