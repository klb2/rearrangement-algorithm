import numpy as np
from scipy import stats


#def sum_func(x):
#    """Sum function that can be used in the RA as a supermodular function"""
#    return np.sum(x, axis=1)
#
#def max_func(x):
#    """Max function that can be used in the RA as a supermodular function"""
#    return np.max(x, axis=1)


def basic_rearrange(x_mat, tol, tol_type, lookback, max_ra, optim_func,
                    supermod_func=np.sum, is_sorted=True, verbose=False,
                    *args, **kwargs):
    num_samples, num_var = np.shape(x_mat)
    x_mat = np.vstack([np.random.permutation(_col) for _col in x_mat.T]).T  #random permutation
    row_sums = supermod_func(x_mat, axis=1)

    iteration = 0
    col_idx = 0
    opt_rs_history = []
    while True:
        iteration = iteration + 1

        #### FOR SUM
        ##for col_idx in range(num_var):
        #_column = x_mat[:, col_idx]
        #rs_mj = row_sums - _column
        #_rank_idx = stats.rankdata(rs_mj, method='ordinal')-1
        #rearrange_col = np.sort(_column)[::-1][_rank_idx]
        #x_mat[:, col_idx] = rearrange_col
        #row_sums = rs_mj + rearrange_col
        #####
        
        _column = x_mat[:, col_idx]
        _x_wo_column = np.delete(x_mat, col_idx, axis=1) # https://stackoverflow.com/q/21022542
        rs_mj = supermod_func(_x_wo_column, axis=1)
        _rank_idx = stats.rankdata(rs_mj, method='ordinal')-1
        rearrange_col = np.sort(_column)[::-1][_rank_idx]
        x_mat[:, col_idx] = rearrange_col
        row_sums = supermod_func(x_mat, axis=1)

        opt_rs_new = optim_func(row_sums)
        opt_rs_history.append(opt_rs_new)

        if iteration > lookback:
            opt_rs_lookback_ago = opt_rs_history[iteration-lookback-1]
            if tol_type == "absolute":
                _tol = abs(opt_rs_new - opt_rs_lookback_ago)
            elif tol_type == "relative":
                _tol = abs((opt_rs_new - opt_rs_lookback_ago)/opt_rs_lookback_ago)

            #tol_reached = np.allclose(x_mat, x_old) if tol == 0 else _tol <= tol
            tol_reached = _tol <= tol

            if tol_reached or iteration == max_ra:
                break
        #else:
        #opt_rs_old = opt_rs_new
        #x_old = np.copy(x_mat)

        col_idx = np.mod(col_idx + 1, num_var)
#    bound = optim_func(row_sums)
#    print(iteration)
#    print(bound)
#    print(opt_rs_history)
    #print(x_mat)
    return x_mat


def create_matrix_from_quantile(quant, prob, level=1.):
    x_mat = np.array([qf(prob) for qf in quant]).T
    num_samples, num_var = np.shape(x_mat)
    for _col_idx in range(num_var):
        if np.isinf(x_mat[0, _col_idx]):
            x_mat[0, _col_idx] = quant[_col_idx](level/(2*num_samples))
        if np.isinf(x_mat[-1, _col_idx]):
            x_mat[-1, _col_idx] = quant[_col_idx](level+(1-level)*(1-1/(2*num_samples)))
    return x_mat


def rearrange_algorithm(level: float, quant, num_steps: int=10, abstol: float=0,
                        lookback: int=0, max_ra: int=0, method: str="lower",
                        sample: bool=True, supermod_func=np.sum):
    """Computing the lower/upper bounds for the best and worst VaR

    This function performs the RA and calculates the lower and upper bounds on
    the worst or best case VaR for a given confidence level.

    Parameters
    ----------
    level : float
        Confidence level between 0 and 1.

    quant : list
        List of marginal quantile functions

    num_steps : int
        Number of discretization points

    abstol : float
        Absolute convergence tolerance

    lookback : int
        Number of column rearrangements to look back for deciding about
        convergence. Must be a number in {1, ..., max_ra-1}.
        If set to zero, it defaults to len(quant).

    max_ra : int
        Number of column rearrangements. If zero, it defaults to infinitely
        many.

    method : str
        Risk measure that is approximated. Valid options are:
        * `lower` or `best.VaR`: for best VaR
        * `upper` or `worst.VaR`: for worst VaR

    sample : bool
        Indication whether each column of the two working matrices is randomly
        permuted before the rearrangements begin


    Returns
    -------
    TODO
    """
    if lookback == 0:
        lookback = len(quant)
    method = method.lower()

    #if not 0 < level < 1:
    #    raise ValueError("Level needs to be between zero and one!")
    if abstol < 0:
        raise ValueError("Absolute tolerance needs to be non-negative!")
    if num_steps < 2:
        raise ValueError("Number of discretization points needs to be at least 2")

    if method in ['worst.VaR', 'upper']:
        prob_under = level + (1-level)*np.arange(num_steps)/num_steps
        prob_over = level + (1-level)*np.arange(1, num_steps+1)/num_steps
        optim_func = min
    elif method in ['best.VaR', 'lower']:
        prob_under = level*np.arange(num_steps)/num_steps
        prob_over = level*np.arange(1, num_steps+1)/num_steps
        optim_func = max
    else:
        raise NotImplementedError("Only best and worst VaR are supported right now.")


    # Determine underline{X}^*
    x_mat_under = create_matrix_from_quantile(quant, prob_under, level)
    x_ra_low = basic_rearrange(x_mat_under, tol=abstol, tol_type="absolute",
            lookback=lookback, max_ra=max_ra, optim_func=optim_func,
            supermod_func=supermod_func)
    bound_low = optim_func(supermod_func(x_ra_low, axis=1))

    # Determine overline{X}^*
    x_mat_over = create_matrix_from_quantile(quant, prob_over, level)
    x_ra_up = basic_rearrange(x_mat_over, tol=abstol, tol_type="absolute",
            lookback=lookback, max_ra=max_ra, optim_func=optim_func,
            supermod_func=supermod_func)
    bound_up = optim_func(supermod_func(x_ra_up, axis=1))
    return (bound_low, x_ra_low), (bound_up, x_ra_up)


def bounds_expectation_supermod(quant, num_steps: int=10, abstol: float=0,
                                lookback: int=0, max_ra: int=0, supermod_func=np.sum,
                                method: str="lower", sample: bool=True):
    if lookback == 0:
        lookback = len(quant)
    method = method.lower()

    #if not 0 < level < 1:
    #    raise ValueError("Level needs to be between zero and one!")
    if abstol < 0:
        raise ValueError("Absolute tolerance needs to be non-negative!")
    if num_steps < 2:
        raise ValueError("Number of discretization points needs to be at least 2")

    prob_under = np.arange(num_steps)/num_steps
    prob_over = (np.arange(num_steps)+1)/num_steps
    optim_func = min

    x_mat_under = create_matrix_from_quantile(quant, prob_under)
    x_mat_over = create_matrix_from_quantile(quant, prob_over)
    if method in ["lower"]:
        x_ra_low = basic_rearrange(x_mat_under, tol=abstol, tol_type="absolute",
                                lookback=lookback, max_ra=max_ra, optim_func=optim_func)
        x_ra_up = basic_rearrange(x_mat_over, tol=abstol, tol_type="absolute",
                                lookback=lookback, max_ra=max_ra, optim_func=optim_func)
    elif method in ["upper"]:
        x_ra_low = np.sort(x_mat_under, axis=0)
        x_ra_up = np.sort(x_mat_over, axis=0)
    bound_low = np.mean(supermod_func(x_ra_low, axis=1))
    bound_up = np.mean(supermod_func(x_ra_up, axis=1))
    return (bound_low, x_ra_low), (bound_up, x_ra_up)
