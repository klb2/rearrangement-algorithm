import numpy as np

def _rearrange(quant, prob, tol, tol_type, lookback, max_ra, method, sample,
               is_sorted=True, verbose=False):
    x_mat = np.array([qf[prob] for qf in quant]).T
    num_samples, num_var = np.shape(x_mat)
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


def rearrange(level: float, quant, num_steps:int=10, abstol:float=0,
              lookback:int=0, max_ra:int=0, method:str="lower",
              sample:bool=True):
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


    result_low = _rearrange(quant, prob_under, tol=abstol, tol_type="absolute",
                            lookback=lookback, max_ra=max_ra, method=method,
                            sample=sample, is_sorted=True)

    result_up = _rearrange(quant, prob_over, tol=abstol, tol_type="absolute",
                           lookback=lookback, max_ra=max_ra, method=method,
                           sample=sample, is_sorted=True)
