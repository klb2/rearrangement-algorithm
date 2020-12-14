import numpy as np

def _indices_opp_ordered_to(x):
    return np.argsort(np.argsort(x)[::-1])

def rearrange(quant, prob, tol, tol_type, lookback, max_ra, method, sample,
              is_sorted=True, verbose=False, *args, **kwargs):
    """Function to do the rearrangement.

    Parameters
    ----------
    quant : list
        List of marginal quantile functions

    prob : list
        Array that contains the probabilities/level sets at which the quantile
        functions are evaluated.

    tol : float
        Tolerance to determine the convergence

    tol_type : str
        Type of tolerance functions ("absolute" or "relative")

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

    *args, **kwargs
        Remaining arguments and keyword arguments are passed to the
        optimization function.


    Returns
    -------
    TODO
    """
    x_mat = np.array([qf[prob] for qf in quant]).T
    num_samples, num_var = np.shape(x_mat)
    if max_ra == 0:
        max_ra = np.Inf

    if method in ['best', 'lower']:
        # TODO: adjustment of levels that are -Inf (for 0 quantile)
        opt_func = max
    elif method in ['worst', 'upper']:
        # TODO: adjustment of levels that are +Inf (for 1 quantile)
        opt_func = min
    else:
        raise NotImplementedError("Only best and worst VaR are supported right now.")

    if tol_type.startswith("abs"):
        tol_func = lambda x, y: abs(x-y)
    else:
        tol_func = lambda x, y: abs((x-y)/y)

    #TODO: verbose

    #x_lst = _col_split(x_mat)
    x_lst = x_mat
    row_sums = np.sum(x_lst, axis=1)

    num_col_no_change = 0
    len_opt_row_sums = 64
    opt_row_sums = []#np.zeros(len_opt_row_sums)
    iter_num = 0
    col_num = 0

    while True:
        iter_num = iter_num + 1
        col_num = 0 if col_num >= num_var else col_num+1
        y_lst = np.copy(x_lst) # there should be a better solution
        y_rs = np.copy(row_sums)

        y_col_j = y_lst[:, col_num]
        _rs_mj = y_rs - y_col_j
        yj = x_lst[:, col_num][_indices_opp_ordered_to(_rs_mj)]
        y_lst[:, col_num] = yj
        _rs_mj = _rs_mj + yj

        #TODO: verbose

        opt_rs_cur_col = opt_func(y_rs)
        #if iter_num > len_opt_row_sums:
        #    pass
        opt_row_sums.append(opt_rs_cur_col)

        if iter_num > lookback:
            pass



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

    # Determine underline{X}^*
    if method in ['worst', 'upper']:
        prob_under = level + (1-level)*np.arange(num_steps)/num_steps
        prob_over = level + (1-level)*np.arange(1, num_steps+1)/num_steps
    elif method in ['best', 'lower']:
        prob_under = level*np.arange(num_steps)/num_steps
        prob_over = level*np.arange(1, num_steps+1)/num_steps
    else:
        raise NotImplementedError("Only best and worst VaR are supported right now.")


    result_low = rearrange(quant, prob_under, tol=abstol, tol_type="absolute",
                           lookback=lookback, max_ra=max_ra, method=method,
                           sample=sample, is_sorted=True)

    result_up = rearrange(quant, prob_over, tol=abstol, tol_type="absolute",
                          lookback=lookback, max_ra=max_ra, method=method,
                          sample=sample, is_sorted=True)
