import numpy as np
from tectosaur_tables.adjacent import make_adjacent_params, eval_tri_integral
from tectosaur_tables.build_tables import build_and_test_tables

def final_table():
    p = make_adjacent_params('H', 1e-7, 50, True, True, 50, 50, 0.01, 200, 14, 6)
    p.n_test_tris = 100
    build_and_test_tables(new_eval, p)
    plt.savefig('adjacent_H_final_table_err.pdf')

if __name__ == '__main__':
    # final_table()

    p = make_adjacent_params(
        "T", 1e-4, 25, True, True, 25, 25, 0.01, 10, False, 1, 1
    )

    import tectosaur.mesh as mesh
    from tectosaur.dense_integral_op import DenseIntegralOp

    phi = 2.5 * np.pi / 2
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
    eps = 0.08 * (2.0 ** -np.arange(0, 12))
    iop = DenseIntegralOp(
        eps, 25, 15, 6, 3, 6, 4.0,
        'T', 1.0, 0.25, m[0], m[1],
        use_tables = False,
        remove_sing = False
    )
    A = iop.mat[:9,9:].reshape((3,3,3,3))

    obs_tri = np.array(m[0][m[1][0,:]])
    src_tri = np.array(m[0][m[1][1,:]])
    pr = 0.25
    out = eval_tri_integral(obs_tri, src_tri, pr, p)
    B = out[:,0].reshape((3,3,3,3))
    import ipdb; ipdb.set_trace()

