from tectosaur_tables.adjacent import eval_integral, make_adjacent_params
from tectosaur.test_decorators import golden_master

@golden_master
def test_adjacentH():
    p = make_adjacent_params(
        "H", 1e-3, 25, True, True, 25, 25, 0.05, 5, True, 14, 6
    )
    return eval_integral(0, p.pts[0,:].tolist(), p)
