import numpy as np
from scipy import stats


def basic_rearrange(quant, prob, tol, tol_type, lookback, max_ra, optim_func, level,
                    is_sorted=True, verbose=False, *args, **kwargs):
    x_mat = np.array([qf(prob) for qf in quant]).T
    num_samples, num_var = np.shape(x_mat)
    for _col_idx in range(num_var):
        if np.isinf(x_mat[0, _col_idx]):
            x_mat[0, _col_idx] = quant[_col_idx](level/(2*num_samples))
        if np.isinf(x_mat[-1, _col_idx]):
            x_mat[-1, _col_idx] = quant[_col_idx](level+(1-level)*(1-1/(2*num_samples)))

    x_mat = np.vstack([np.random.permutation(_col) for _col in x_mat.T]).T  #random permutation
    row_sums = np.sum(x_mat, axis=1)  # TODO: to change
    #opt_rs_old = optim_func(row_sums)
    #x_old = np.copy(x_mat)

    iteration = 0
    col_idx = 0
    opt_rs_history = []
    while True:
        iteration = iteration + 1

        #for col_idx in range(num_var):
        _column = x_mat[:, col_idx]
        rs_mj = row_sums - _column  # TODO: to change
        _rank_idx = stats.rankdata(rs_mj, method='ordinal')-1
        rearrange_col = np.sort(_column)[::-1][_rank_idx]
        x_mat[:, col_idx] = rearrange_col
        row_sums = rs_mj + rearrange_col  # TODO: to change

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
    bound = optim_func(row_sums)
    print(iteration)
    print(bound)
    print(opt_rs_history)
    #print(x_mat)
    return bound, x_mat

def rearrange_algorithm(level: float, quant, num_steps: int=10, abstol: float=0,
                        lookback: int=0, max_ra: int=0, method: str="lower",
                        sample: bool=True):
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
        * `lower` or `best`: for best VaR
        * `upper` or `worst`: for worst VaR

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

    if not 0 < level < 1:
        raise ValueError("Level needs to be between zero and one!")
    if abstol < 0:
        raise ValueError("Absolute tolerance needs to be non-negative!")
    if num_steps < 2:
        raise ValueError("Number of discretization points needs to be at least 2")

    if method in ['worst', 'upper']:
        prob_under = level + (1-level)*np.arange(num_steps)/num_steps
        prob_over = level + (1-level)*np.arange(1, num_steps+1)/num_steps
        optim_func = min
    elif method in ['best', 'lower']:
        prob_under = level*np.arange(num_steps)/num_steps
        prob_over = level*np.arange(1, num_steps+1)/num_steps
        optim_func = max
    else:
        raise NotImplementedError("Only best and worst VaR are supported right now.")


    # Determine underline{X}^*
    result_low = basic_rearrange(quant, prob_under, tol=abstol, tol_type="absolute",
                           lookback=lookback, max_ra=max_ra, optim_func=optim_func, #method=method,
                           level=level)

    # Determine overline{X}^*
    result_up = basic_rearrange(quant, prob_over, tol=abstol, tol_type="absolute",
                          lookback=lookback, max_ra=max_ra, optim_func=optim_func, #method=method,
                          level=level)
    #print(result_low)
