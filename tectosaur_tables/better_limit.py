import numpy as np
import scipy.special

def make_poly_term(i, max):
    poly = scipy.special.chebyt(i, True)
    def poly_term(e):
        # return e ** i
        arg = (2.0 / max) * e - 1
        return poly(arg)
    return poly_term

def make_terms(n_terms, log_terms, max):
    poly_terms = n_terms - log_terms
    terms = []
    for i in range(log_terms):
        terms.append(lambda e, i=i: e ** i * np.log(e))
    for i in range(poly_terms):
        terms.append(make_poly_term(i, max))
    return terms

def limit_coeffs(eps_vals, f_vals, log_terms, max):
    terms = make_terms(len(eps_vals), log_terms, max)
    mat = [[t(e) for t in terms] for e in eps_vals]
    assert(np.linalg.cond(mat) < 10000)
    coeffs = np.linalg.lstsq(mat, f_vals)
    return coeffs[0], terms

def limit(eps_vals, f_vals, log_terms, max):
    coeffs, terms = limit_coeffs(eps_vals, f_vals, log_terms, max)
    res = np.sum(coeffs[log_terms:] * [t(0) for t in terms[log_terms:]])
    if log_terms > 0:
        return res, coeffs[0]
    else:
        return res, 0

