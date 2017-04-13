import numpy as np
from tectosaur_tables.build_tables import build_and_test_tables
from tectosaur_tables.coincident import make_coincident_params, eval_integral

def final_table():
    p = make_coincident_params(
        "H", 1e-6, 200, False, 100, 180, 1e-1 / 32, 6, 12, 17, 9
    )
    p.n_test_tris = 100
    build_tables(eval_integral, p)
    plt.savefig('final_table_err.pdf')

if __name__ == '__main__':
    p = make_coincident_params(
        "T", 1e-8, 25, True, True, 25, 25, 0.01, 40, False, 1, 1, 1
    )
    # p.n_test_tris = 0
    # build_and_test_tables(eval_integral, p)


    from tectosaur_tables.coincident import eval_tri_integral
    tri = np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0]])
    pr = 0.25
    out = eval_tri_integral(tri, pr, p)
    print(out)
