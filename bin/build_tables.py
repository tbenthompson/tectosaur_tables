def take_limits(integrals, remove_divergence, all_eps):
    out = np.empty((81, 2))
    for i in range(81):
        out[i] = limit(all_eps, integrals[:, i], remove_divergence)
    return out

def test_f(results, eval_fnc, pts, wts, p):
    rand_pt = np.random.rand(pts.shape[1]) * 2 - 1.0
    correct = eval_fnc(0, rand_pt, p)[:,0]
    for i in range(1):
        interp = barycentric_evalnd(pts, wts, results[:,i,0], np.array([rand_pt]))[0]
        rel_err = np.abs((correct[i] - interp) / correct[i])
        abs_diff = correct[i] - interp
        print('testing: ' + str(i) + ' ' + str((correct[i], interp, rel_err, abs_diff)))
    return rel_err

def build_tables(eval_fnc, p, run_test = True):
    pts, wts = coincident_interp_pts_wts(p.n_A, p.n_B, p.n_pr)

    results = np.array([eval_fnc(i, pt, p) for i, pt in enumerate(pts.tolist())])
    np.save(p.filename, results)
    results = np.load(p.filename)
    if run_test:
        np.random.seed(15)
        all = []
        for i in range(30):
            all.append(test_f(results, eval_fnc, pts, wts, p))
        plt.figure()
        plt.hist(np.log10(np.abs(all)))
        plt.title(str(p.n_A) + '_' + str(p.n_B) + '_' + str(p.n_pr))
