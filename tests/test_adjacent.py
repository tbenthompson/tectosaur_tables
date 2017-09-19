from tectosaur_tables.adjacent import eval_integral, make_adjacent_params, eval_tri_integral
from tectosaur.util.test_decorators import golden_master

@golden_master
def test_adjacentH():
    p = make_adjacent_params(
        "H", 1e-3, 25, True, True, 25, 25, 0.05, 5, True, 14, 6
    )
    return eval_integral(0, p.pts[0,:].tolist(), p)

def flipping_experiment():
    p = make_adjacent_params("elasticT3", 1e-3, 25, True, True, 25, 25, 1e-2, 3, True, 1, 1)
    results = eval_tri_integral(
        [[0,0,0],[1,0,0],[0.5,0.5,0]],
        [[1,0,0],[0,0,0],[0.5,-0.5,0]],
        0.25, p
    )
    print(results[:,0].reshape((3,3,3,3)))

if __name__ == "__main__":
    flipping_experiment()

