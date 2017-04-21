import numpy as np
from tectosaur_tables.adjacent import make_adjacent_params, eval_tri_integral,\
    eval_integral
from tectosaur_tables.build_tables import build_and_test_tables
import matplotlib.pyplot as plt

def final_table(K, n_eps, n_phi, n_pr, remove_log):
    p = make_adjacent_params(K, 1e-7, 50, True, True, 50, 50, 0.01, n_eps, remove_log, n_phi, n_pr)
    p.n_test_tris = 100
    build_and_test_tables(eval_integral, p)
    plt.savefig('adjacent_' + str(K) + '_final_table_err.pdf')

def interp_order_test(K, remove_log, n_phi, n_pr):
    p = make_adjacent_params(K, 1e-7, 25, True, True, 25, 25, 0.1, 1, remove_log, n_phi, n_pr)
    p.n_test_tris = 100
    build_and_test_tables(eval_integral, p)
    plt.savefig('adj_interp_test_' + str(n_phi) + '_' + str(n_pr) + '.pdf')

def explore_interp_orders(K, remove_log):
    n_phi = np.arange(5, 15)
    n_pr = np.arange(4, 11)
    n_phi = [1]
    n_pr = [1]
    for i in range(len(n_phi) - 1):
        interp_order_test(K, remove_log, n_phi[i], n_pr[-1])
    for i in range(len(n_pr) - 1):
        interp_order_test(K, remove_log, n_phi[-1], n_pr[i])
    interp_order_test(K, remove_log, n_phi[-1], n_pr[-1])

def eps_order_test(K, remove_log, starting_eps, n_eps):
    p = make_adjacent_params(K, 1e-7, 25, True, True, 25, 25, starting_eps, n_eps, remove_log, 1, 1)
    pr = 0.25
    phi = 0.5 * np.pi
    Y = p.psi * np.cos(phi)
    Z = p.psi * np.sin(phi)
    obs_tri = [[0,0,0],[1,0,0],[0.5,p.psi,0]]
    src_tri = [[1,0,0],[0,0,0],[0.5,Y,Z]]
    out = eval_tri_integral(obs_tri, src_tri, pr, p)
    return out

def explore_eps_orders(K, remove_log):
    n_eps = [16,32,64,128,256]
    results = []
    for N in n_eps:
        results.append(eps_order_test(K, remove_log, 0.01, N))
        print("eps test done")
        print(results[-1])
    print('Tests complete...')
    import ipdb;ipdb.set_trace()
    # print(str(zip(n_eps, results)))

if __name__ == '__main__':
    # final_table('H', 200, 14, 6, True)
    final_table('T', 256, 12, 7, False)
    # explore_interp_orders('T', False)
    # explore_eps_orders('T', False)



def compare_with_old():
    p = make_adjacent_params(
        "T", 1e-4, 25, True, True, 25, 25, 0.01, 10, False, 1, 1
    )
    import tectosaur.mesh as mesh
    from tectosaur.dense_integral_op import DenseIntegralOp
    # for i in range(10):
    #     phi = (0.2 + np.random.rand(1)[0] * 2.3) * np.pi / 2.0
    phi = 0.51
    Y = np.cos(phi)
    Z = np.sin(phi)
    m = (
        np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, Y, Z]
        ]),
        np.array([
            [0, 1, 2],
            [1, 0, 3]
        ], dtype = np.int64)
    )
    print(m[0][m[1]].tolist()[0])
    print(m[0][m[1]].tolist()[1])
    eps = 0.08 * (2.0 ** -np.arange(0, 8))
    iop = DenseIntegralOp(
        eps, 1, 25, 1, 1, 1, 4.0,
        'T', 1.0, 0.25, m[0], m[1],
        use_tables = False,
        remove_sing = False
    )

    obs_tri = np.array(m[0][m[1][0,:]])
    src_tri = np.array(m[0][m[1][1,:]])
    pr = 0.25
    out = eval_tri_integral(obs_tri, src_tri, pr, p)

    A = iop.mat[:9,9:].reshape((3,3,3,3))
    B = out[:,0].reshape((3,3,3,3))
    diff = A - B
    print(diff)
    check = np.all(np.abs(diff) < 1e-4)
    if not check:
        import ipdb; ipdb.set_trace()
