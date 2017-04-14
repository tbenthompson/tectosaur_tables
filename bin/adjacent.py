from tectosaur_tables.adjacent import make_adjacent_params
from tectosaur_tables.build_tables import build_and_test_tables

def final_table():
    p = make_adjacent_params('H', 1e-7, 50, True, True, 50, 50, 0.01, 200, 14, 6)
    p.n_test_tris = 100
    build_and_test_tables(new_eval, p)
    plt.savefig('adjacent_H_final_table_err.pdf')

def test_13th():
    p = make_adjacent_params('H', 1e-7, 50, True, True, 50, 50, 0.01, 200, 14, 6)
    new_eval(13, p.pts[13], p)

if __name__ == '__main__':
    final_table()

    # def aitken(seq, its = 1):
    #     assert(its > 0)
    #     if its > 1:
    #         S = aitken(seq, its - 1)
    #     else:
    #         S = seq
    #     return S[2:] - ((S[2:] - S[1:-1]) ** 2 / ((S[2:] - S[1:-1]) - (S[1:-1] - S[:-2])))

    # vs = np.array([-0.11527428819709082, -0.11619966401248603, -0.11638986792026991, -0.11646070487737595, -0.1164949890546126, -0.11651476591045216, -0.116526746767, -0.116534185967, -0.116539735132, -0.116543628948])
    # print(aitken(vs))
    # print(aitken(vs, 2))
    # print(aitken(vs, 3))
    # print(aitken(vs, 4))
    # import sys;sys.exit()

    # start = [0.01]#[0.2, 0.1, 0.05, 0.025, 0.01]
    # n_interp = [200]#[6, 8, 10, 13, 16, 20]


    # # with err = 1e-7, result = array([-0.11653372, -0.11654198, -0.11654784, -0.11655055, -0.11655294])
    # # with err = 1e-8, result = array([-0.11653357, -0.11654182, -0.11654708, -0.11655049, -0.11655285])

    # for S in start:
    #     for N in n_interp:
    #         p = make_adjacent_params('H', 1e-7, 50, True, True, 50, 50, S, N, 1, 1, eps_step = 1.5)
    #         build_tables(new_eval, p, run_test = False)
    #         # plt.savefig('adjacent_H_final_table_err.pdf')
    # print(results)
    # print(results)
    # import pickle
    # with open('adjacent_grid_search.pkl', 'wb') as f:
    #     pickle.dump(results, f)

    # p = make_adjacent_params('H', 1e-6, 150, True, 100, 100, 1e-1 / 4, 10, 1, 1)
    # p = make_adjacent_params('H', 1e-6, 100, True, 150, 100, 1e-1 / 4, 10, 1, 1)

    # These parameters give 1e-9 error for the individual integrals
    # as tested by increasing each quadrature order and then comparing the values
    # p = make_adjacent_params('H', 1e-6, 200, False, 200, 200, 8e-1, 23, 1, 1, eps_step = np.sqrt(2))
    # p.n_test_tris = 0
    # build_tables(eval, p)
    # plt.show()

    # n_phi = 20
    # for n_pr in range(2, 12):
    #     p = make_adjacent_params('H', 1e-3, 40, True, 50, 50, 1e-1, 2, n_phi, n_pr)
    #     p.n_test_tris = 10
    #     build_tables(eval, p)
    #     plt.savefig('adj_' + str(n_phi) + '_' + str(n_pr) + '.pdf')
    #     # plt.show()
    # n_pr = 12
    # for n_phi in range(2, 30, 3):
    #     p = make_adjacent_params('H', 1e-3, 40, True, 50, 50, 1e-1, 2, n_phi, n_pr)
    #     p.n_test_tris = 10
    #     build_tables(eval, p)
    #     plt.savefig('adj_' + str(n_phi) + '_' + str(n_pr) + '.pdf')
    # plt.show()

# H parameters
# K = "H"
# tol = 1e-7
# rho_order = 70
# theta_order = 70
# starting_eps = 1e-5
# n_eps = 4
# n_pr = 8
# n_theta = 8

# play parameters
# K = "H"
# rho_order = 40
# theta_order = 28
# starting_eps = 1e-4
# n_eps = 2
# tol = 1e-6
# n_pr = 2
# n_theta = 2
