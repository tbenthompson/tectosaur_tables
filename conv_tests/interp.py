import scipy.special
import numpy as np
import matplotlib.pyplot as plt
from tectosaur.interpolate import cheb

# data = [0.0006827105644366296, 0.0010053619592424034, 0.0014456945537162757, 0.0020295631534195035, 0.002781834803352217, 0.0037244876921632137, 0.004875127130823547, 0.006246133353873914, 0.007844460727703393, 0.009671962654106145, 0.011726055058313487, 0.014000539146491946, 0.01648644767291359, 0.0191728297181584, 0.02204743131377812, 0.025097258384754736, 0.028309025626717815, 0.03166950338356974, 0.035165777590959, 0.03878543787218614, 0.04251670750169802, 0.04634852707388779, 0.0502706018135169, 0.05427342074838063, 0.05834825447763686, 0.062487137009825826, 0.06668283609416616, 0.0709288155966472, 0.07521919274840257, 0.07954869249518562, 0.08391260074644308, 0.08830671792571852, 0.09272731288525335, 0.09717107883321792, 0.10163509537829399, 0.10611677806974795, 0.11061386459556148, 0.11512435342862974, 0.11964651389583734, 0.12417877900192174]
# eps = [0.1, 0.08408964152537146, 0.07071067811865477, 0.059460355750136064, 0.05000000000000002, 0.04204482076268574, 0.03535533905932738, 0.029730177875068032, 0.02500000000000001, 0.021022410381342872, 0.017677669529663695, 0.01486508893753402, 0.012500000000000006, 0.010511205190671438, 0.00883883476483185, 0.007432544468767011, 0.006250000000000003, 0.00525560259533572, 0.004419417382415925, 0.003716272234383506, 0.0031250000000000023, 0.00262780129766786, 0.0022097086912079627, 0.001858136117191753, 0.0015625000000000014, 0.0013139006488339302, 0.0011048543456039816, 0.0009290680585958768, 0.0007812500000000007, 0.0006569503244169652, 0.0005524271728019908, 0.00046453402929793843, 0.00039062500000000046, 0.00032847516220848265, 0.00027621358640099545, 0.00023226701464896922, 0.00019531250000000023, 0.00016423758110424135, 0.00013810679320049775, 0.00011613350732448464]

def make_terms(n_terms, log_terms, max):
    poly_terms = n_terms - log_terms
    terms = []
    for i in range(log_terms):
        terms.append(lambda e, i=i: e ** i * np.log(e))
    for i in range(poly_terms):
        terms.append(lambda e, i=i: scipy.special.chebyt(i)((2.0 / max) * e - 1))
    return terms

def main():
    max = 0.1
    for n_pts in range(2, 20):
        eps = cheb(0, max, n_pts)
        # eps.append((eps[-1] + eps[-2]) / 2)
        # eps.append((eps[-1] + eps[-2]) / 2)
        # eps.append((eps[-1] + eps[-2]) / 2)
        # eps.append((eps[-1] + eps[-2]) / 2)
        # eps.append(0.005)
        # eps.append(0.0025)
        # eps.append(0.0025 / 2)
        # eps.append(0.0025 / 4)
        # eps.append(0.0025 / 8)
        # eps.append(0.0025 / 8)
        print(eps)
        terms = make_terms(len(eps), 1, max)
        mat = [[t(e) for t in terms] for e in eps]
        print(np.linalg.cond(mat))



if __name__ == '__main__':
    main()