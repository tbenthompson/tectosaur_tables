import numpy as np
from tectosaur_tables.adjacent import make_adjacent_params, eval_tri_integral,\
    eval_integral
from tectosaur_tables.build_tables import build_and_test_tables
import matplotlib.pyplot as plt

def final_tableH():
    p = make_adjacent_params('H', 1e-7, 50, True, True, 50, 50, 0.01, 200, 14, 6)
    p.n_test_tris = 100
    build_and_test_tables(eval_integral, p)
    plt.savefig('adjacent_H_final_table_err.pdf')

def interp_order_test(K, remove_log, n_phi, n_pr):
    p = make_adjacent_params(K, 1e-7, 25, True, True, 25, 25, 0.1, 1, remove_log, n_phi, n_pr)
    p.n_test_tris = 10
    build_and_test_tables(eval_integral, p)
    plt.savefig('adj_interp_test_' + str(n_phi) + '_' + str(n_pr) + '.pdf')

if __name__ == '__main__':
    # final_table()

    n_phi = [5, 8, 11, 14]
    n_pr = [4, 6, 8, 10]
    # n_phi = [1]
    # n_pr = [1]
    for i in range(len(n_phi - 1)):
        interp_order_test('T', False, n_phi[i], n_pr[-1])
    for i in range(len(n_pr - 1)):
        interp_order_test('T', False, n_phi[-1], n_pr[i])
    interp_order_test('T', False, n_phi[-1], n_pr[-1])


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
