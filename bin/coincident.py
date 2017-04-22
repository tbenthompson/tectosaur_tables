import numpy as np
import matplotlib.pyplot as plt
from tectosaur_tables.build_tables import build_and_test_tables
from tectosaur_tables.coincident import make_coincident_params, eval_integral, eval_tri_integral

def final_table(K, max_eps, n_eps, n_A, n_B, n_pr, remove_log):
    p = make_coincident_params(
        K, 1e-7, 25, True, True, 25, 25, max_eps, n_eps, remove_log, n_A, n_B, n_pr
    )
    p.n_test_tris = 100
    build_and_test_tables(eval_integral, p)
    plt.savefig('coincident_' + str(K) + '_final_table_err.pdf')

def interp_order_test(K, remove_log, n_A, n_B, n_pr):
    p = make_coincident_params(
        K, 1e-7, 25, True, True, 25, 25, 0.1, 1, remove_log, n_A, n_B, n_pr
    )
    p.n_test_tris = 100
    build_and_test_tables(eval_integral, p)
    plt.savefig('co_interp_test_' + str(n_A) + '_' + str(n_B) + '_' + str(n_pr) + '.pdf')

def explore_interp_orders(K, remove_log):
    n_A = [8, 12, 16]
    n_B = [13, 17, 20]
    n_pr = [5, 7, 10]
    for i in range(len(n_A) - 1):
        interp_order_test(K, remove_log, n_A[i], n_B[-1], n_pr[-1])
    for i in range(len(n_B) - 1):
        interp_order_test(K, remove_log, n_A[-1], n_B[i], n_pr[-1])
    for i in range(len(n_pr) - 1):
        interp_order_test(K, remove_log, n_A[-1], n_B[-1], n_pr[i])
    interp_order_test(K, remove_log, n_A[-1], n_B[-1], n_pr[-1])

def eps_order_test(K, remove_log, starting_eps, n_eps):
    p = make_coincident_params(
        K, 1e-7, 25, True, True, 25, 25, starting_eps, n_eps, remove_log, 1, 1, 1
    )
    A = 0.3
    B = 0.4
    pr = 0.25
    tri = [[0,0,0],[1,0,0],[A,B,0]]
    out = eval_tri_integral(tri, pr, p)
    return out

def explore_eps_orders(K, remove_log):
    # n_eps = [16,32,64,128,256]
    n_eps = [2,3]
    max_eps = 1e-7

    results = []
    for N in n_eps:
        results.append(eps_order_test(K, remove_log, max_eps, N))
        print("eps test done")
        print(results[-1])
    print('Tests complete...')
    results = np.array(results)
    a = results[:,0,0]
    print(np.abs((a[1:] - a[:-1]) / a[1:]))
    np.save('co_' + K + '_eps_convergence.npy', results)

if __name__ == '__main__':
    final_table('T', 1e-7, 3, 12, 13, 7, False)
    # explore_interp_orders('T', False)
    # explore_eps_orders('T', False)
    # p = make_coincident_params(
    #     "T", 1e-8, 25, True, True, 25, 25, 0.01, 40, False, 1, 1, 1
    # )
    # # p.n_test_tris = 0
    # # build_and_test_tables(eval_integral, p)


    # from tectosaur_tables.coincident import eval_tri_integral
    # tri = np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0]])
    # pr = 0.25
    # out = eval_tri_integral(tri, pr, p)
    # print(out)
