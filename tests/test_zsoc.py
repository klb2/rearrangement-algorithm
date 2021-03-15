import numpy as np
import scratch
from scipy import stats

@np.vectorize
def quant_neg_exp(p):
    #if p == 0:
    #    return -np.inf
    #return np.log(p)
    return -stats.expon.ppf(1-p)
qf = [stats.expon.ppf, stats.expon.ppf, quant_neg_exp, quant_neg_exp]

num_steps = 1000

prob_under = np.arange(num_steps)/num_steps
prob_over = np.arange(1, num_steps+1)/num_steps

x_mat = scratch.create_matrix_from_quantile(qf, prob_over, level=0.)
#_x1 = np.array([.1, .2, .5, 1, 2, 5])
#_y1 = -np.array([.1, .2, .5, 1, 2, 3])
#x_mat = np.vstack((_x1, _x1, _y1, _y1)).T
print(x_mat)
optim_func = min
x_ra_up = scratch.basic_rearrange(x_mat, tol=0, tol_type="absolute", lookback=len(qf),
                          max_ra=0, optim_func=optim_func)
bound_up = optim_func(np.sum(x_ra_up, axis=1))
print(x_ra_up)
print(bound_up)

#alpha = 0
#results_up_under, results_up_over = scratch.bounds_var(alpha, qf, method="upper", num_steps=50)
#results_low_under, results_low_over = scratch.bounds_var(alpha, qf, method="lower", num_steps=50)
#print("Results")
#print(results_low_under[0], results_low_over[0])
#print(results_up_under[0], results_up_over[0])
