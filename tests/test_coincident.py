from tectosaur_tables.coincident import eval_integral, make_coincident_params
from tectosaur.test_decorators import golden_master

@golden_master
def test_coincidentH():
    p = make_coincident_params(
        "H", 1e-6, 25, True, True, 25, 25, 0.01, 10, True, 12, 17, 9
    )
    return eval_integral(0, p.pts[0,:].tolist(), p)
