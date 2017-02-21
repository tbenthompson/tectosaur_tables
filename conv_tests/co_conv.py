import sys
import numpy as np
import matplotlib.pyplot as plt

from tectosaur.limit import *

# data = np.array([2.07219073e-07,1.63962263e-06,1.25589286e-05,8.50385877e-05,3.74305515e-04,1.02408106e-04,-6.33866485e-03,-2.71863864e-02,-6.39659259e-02,-1.12261117e-01,-1.67251270e-01,-2.25760269e-01,-2.86043187e-01,-3.47207810e-01,-4.08809514e-01,-4.70628176e-01,-5.32554728e-01,-5.94534989e-01,-6.56541802e-01,-7.18560176e-01])
#
# eps = 16.0 * (2.0 ** -np.arange(20))
#
# start = 5
# for n_eps in range(3, len(data) - start - 2):
#     end = start + n_eps
#     print(eps[end - 1])
#     print(str(n_eps) + ": " + str(limit(eps[start:end], data[start:end], n_eps // 2 + 1)))

# start = 7
# for end in range(15, 19):
#     print(str(end) + ": " + str(limit(eps[start:end], data[start:end], 1)))
#
# lim_est = []
# start = 7
# for n_log_terms in range(1, 8):
#     for i in range(start + n_log_terms + 1, 16):
#         lim_est.append(limit(eps[start:i], data[start:i], n_log_terms))
#     print(lim_est[-1])
# print(eps[start:i])
# lim_est = np.array(lim_est)
# print(lim_est)
# for i in range(1, 5):
#     print(aitken(lim_est[:,0], i))
# sys.exit()

# plt.figure(dpi = 300)
# plt.plot(np.log10(eps), data - np.log(eps) * 0.089522614159784741)
# plt.show()


eps = np.array([0.05, 0.03546099290780142, 0.025149640360142857, 0.017836624368895642, 0.012650088204890528, 0.008971693762333708, 0.006362903377541637, 0.004512697430880594, 0.0032004946318302087, 0.002269854348815751])
data = np.array([-0.078515313769355083, -0.10283208289455281, -0.12895463657028489, -0.15640320212095549, -0.1848094647934197, -0.21390023658213603, -0.24347720231110509, -0.27339829411613747, -0.30356253590168081, -0.33389847187017907])
all = [limit(eps[:i], data[:i], 5) for i in range(6, 11)]
print(all)
print(limit(eps, data, 1))
print(aitken(np.array(all)[:,0], 1))
