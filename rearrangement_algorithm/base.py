import numpy as np
from scipy import stats


def basic_rearrange(x_mat, optim_func, lookback=0, tol=0., tol_type='absolute',
                    max_ra=0, cost_func=np.sum, is_sorted=False, *args,
                    **kwargs):
    """Implementation of the matrix rearrangement algorithm.

    Basic implementation of the rearrangement algorithm to get a permutation of
    a matrix :math:`X` such that each column :math:`j` is oppositely ordered to
    the vector derived by applying a function :math:`\\psi` to the (sub-)matrix
    without column :math:`j`.

    The implementation is based on the R library ``qrmtools`` [1]_.
    A detailed description can be found in [2]_.

    
    Parameters
    ----------
    x_mat : 2D-array or matrix
        Matrix of shape `(num_samples, num_var)` that will get rearranged.

    optim_func : func
        Internal optimization function. It needs to take a vector-argument and
        return a scalar. This usually is either :func:`min` or :func:`max`.

    lookback: int, optional
        Number of rearrangement steps to look back to determine convergence.

    tol: float, optional
        Tolerance to determine convergence.

    tol_type: str, optional
        Tolerance function used. Possible options are `"absolute"` and
        `"relative"`.

    max_ra: int, optional
        Number of maximum column rearrangements. If ``0``, there will be no
        limit.

    cost_func: func, optional
        Cost function :math:`\\psi` that is applied to each row. It takes a 2D
        numpy.array as input and outputs a vector that results by applying the
        function to each row. It needs to take the keyword argument ``axis`` in
        the numpy style. An example for the sum is :func:`numpy.sum`.

    is_sorted: bool, optional
        Indicates wheter the columns of the matrix ``x_mat`` are sorted in
        increasing order.


    Returns
    -------
    x_rearranged: numpy.array
        Rearranged matrix of shape `(num_samples, num_var)`.


    References
    ----------
    .. [1] M. Hofert, K. Hornik, and A. J. McNeil, "qrmtools: Tools for
           Quantitative Risk Management." Version 0.0-13
           (https://cran.r-project.org/web/packages/qrmtools/index.html)

    .. [2] M. Hofert, "Implementing the Rearrangement Algorithm: An Example
           from Computational Risk Management," Risks, vol. 8, no. 2, May 2020.
    """
    num_samples, num_var = np.shape(x_mat)
    if lookback == 0:
        lookback = num_var
    if is_sorted:
        x_mat_sorted = np.copy(x_mat)
    else:
        x_mat_sorted = np.sort(x_mat, axis=0)
    x_mat = np.copy(x_mat)
    #x_mat = np.vstack([np.random.permutation(_col) for _col in x_mat.T]).T  #random permutation
    row_sums = cost_func(x_mat, axis=1)

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
        rs_mj = cost_func(_x_wo_column, axis=1)
        #_rank_idx = stats.rankdata(rs_mj, method='ordinal')-1
        #rearrange_col = np.sort(_column)[::-1][_rank_idx]
        _rank_idx = np.argsort(np.argsort(rs_mj)[::-1])
        rearrange_col = x_mat_sorted[:, col_idx][_rank_idx]
        x_mat[:, col_idx] = rearrange_col
        row_sums = cost_func(x_mat, axis=1)

        opt_rs_new = optim_func(row_sums)
        opt_rs_history.append(opt_rs_new)

        #print("Iteration: {:d}".format(iteration))
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
    return x_mat


def create_matrix_from_quantile(quant, prob, level=1.):
    """Create a matrix approximation from marginal quantile functions

    Given a set of :math:`n` quantile functions, this function returns a matrix
    where each column represents a random variable with the specified marginal
    distribution.
    Each row represents the value of the random variable at a given
    probability.  
    For details, see [1]_.


    Parameters
    ----------
    quant : list
        List of marginal quantile functions

    prob : list of float
        Probability values at which the quantile functions are evaluated.

    level : float
        Confidence level between 0 and 1.


    Returns
    -------
    x_mat : numpy.array
        Matrix with the values of the random variables at the specified
        probabilites.


    References
    ----------
    .. [1] M. Hofert, "Implementing the Rearrangement Algorithm: An Example
           from Computational Risk Management," Risks, vol. 8, no. 2, May 2020.
    """
    x_mat = np.array([qf(prob) for qf in quant]).T
    num_samples, num_var = np.shape(x_mat)
    for _col_idx in range(num_var):
        if np.isinf(x_mat[0, _col_idx]):
            x_mat[0, _col_idx] = quant[_col_idx](level/(2*num_samples))
        if np.isinf(x_mat[-1, _col_idx]):
            x_mat[-1, _col_idx] = quant[_col_idx](level+(1-level)*(1-1/(2*num_samples)))
    return x_mat


def bounds_VaR(level: float, quant, num_steps: int=10, abstol: float=0,
               lookback: int=0, max_ra: int=0, method: str="lower", sample:
               bool=True, cost_func=np.sum):
    """Computing the lower/upper bounds for the best and worst VaR

    This function performs the RA and calculates the lower and upper bounds on
    the worst or best case VaR for a given confidence level.  
    For mathematical details, see [1]_.


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
        convergence. Must be a number in :math:`\\{1, ..., \\text{max_ra}-1\\}`.
        If set to zero, it defaults to ``len(quant)``.

    max_ra : int
        Number of column rearrangements. If zero, it defaults to infinitely
        many.

    method : str
        Risk measure that is approximated.
        Valid options are:

        - `lower` or `best.VaR`: for best VaR
        - `upper` or `worst.VaR`: for worst VaR

    sample : bool
        Indication whether each column of the two working matrices is randomly
        permuted before the rearrangements begin


    Returns
    -------
    bound_low : float
        Lower bound on the VaR

    x_ra_low : numpy.array
        Rearranged matrix for the lower bound on the VaR

    bound_up : float
        Upper bound on the VaR

    x_ra_up : numpy.array
        Rearranged matrix for the upper bound on the VaR


    References
    ----------
    .. [1] P. Embrechts, G. Puccetti, and L. Rüschendorf, "Model uncertainty
           and VaR aggregation," Journal of Banking & Finance, vol. 37, no. 8,
           pp.  2750-2764, Aug. 2013.
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
            cost_func=cost_func)
    bound_low = optim_func(cost_func(x_ra_low, axis=1))

    # Determine overline{X}^*
    x_mat_over = create_matrix_from_quantile(quant, prob_over, level)
    x_ra_up = basic_rearrange(x_mat_over, tol=abstol, tol_type="absolute",
            lookback=lookback, max_ra=max_ra, optim_func=optim_func,
            cost_func=cost_func)
    bound_up = optim_func(cost_func(x_ra_up, axis=1))
    return (bound_low, x_ra_low), (bound_up, x_ra_up)


def bounds_expectation_supermod(quant, num_steps: int=10, abstol: float=0,
                                lookback: int=0, max_ra: int=0, supermod_func=np.sum,
                                method: str="lower"):
    """Computing the lower/upper bounds on the expectation of supermodular functions.

    This function performs the RA and calculates the lower and upper bounds on
    the expected value of a supermodular function of dependent random
    variables.  
    For mathematical details, see [1]_.


    Parameters
    ----------
    quant : list
        List of marginal quantile functions

    num_steps : int
        Number of discretization points

    abstol : float
        Absolute convergence tolerance

    lookback : int
        Number of column rearrangements to look back for deciding about
        convergence. Must be a number in :math:`\\{1, ..., \\text{max_ra}-1\\}`.
        If set to zero, it defaults to ``len(quant)``.

    max_ra : int
        Number of column rearrangements. If zero, it defaults to infinitely
        many.

    supermod_func : callable
        Callable function that represent the supermodular function which is
        applied to the dependent random variables.
        It needs to accept the ``axis=1`` keyword argument and handle it in the
        numpy style, i.e., taking a 2D-array as input and returning a
        column-vector.

    method : str
        Determine if the lower or upper bound on the expected value is computed.
        Valid options are:

        - `lower`: for the lower bound
        - `upper`: for the upper bound


    Returns
    -------
    bound_low : float
        Lower bound on the approximated bound of the expected value

    x_ra_low : numpy.array
        Rearranged matrix for the lower bound

    bound_up : float
        Upper bound on the approximated bound of the expected value

    x_ra_up : numpy.array
        Rearranged matrix for the upper bound


    References
    ----------
    .. [1] G. Puccetti and L. Rüschendorf, "Computation of Sharp Bounds on the
           Expected Value of a Supermodular Function of Risks with Given
           Marginals," Communications in Statistics - Simulation and
           Computation, vol. 44, no. 3, pp. 705-718, Mar. 2015.
    """
    if lookback == 0:
        lookback = len(quant)
    method = method.lower()

    if abstol < 0:
        raise ValueError("Absolute tolerance needs to be non-negative!")
    if num_steps < 2:
        raise ValueError("Number of discretization points needs to be at least 2")

    prob_under = np.arange(num_steps)/num_steps
    prob_over = (np.arange(num_steps)+1)/num_steps

    x_mat_under = create_matrix_from_quantile(quant, prob_under, level=0.)
    x_mat_over = create_matrix_from_quantile(quant, prob_over, level=0.)
    if method in ["lower"]:
        optim_func = max#min
        x_ra_low = basic_rearrange(x_mat_under, tol=abstol, tol_type="absolute",
                                   lookback=lookback, max_ra=max_ra,
                                   optim_func=optim_func, cost_func=supermod_func)
        x_ra_up = basic_rearrange(x_mat_over, tol=abstol, tol_type="absolute",
                                  lookback=lookback, max_ra=max_ra,
                                  optim_func=optim_func, cost_func=supermod_func)
    elif method in ["upper"]:
        x_ra_low = np.sort(x_mat_under, axis=0)
        x_ra_up = np.sort(x_mat_over, axis=0)
    bound_low = np.mean(supermod_func(x_ra_low, axis=1))
    bound_up = np.mean(supermod_func(x_ra_up, axis=1))
    return (bound_low, x_ra_low), (bound_up, x_ra_up)




def bounds_surv_probability(quant, s_level: float, num_steps: int=10, abstol:
                            float=0, lookback: int=0, max_ra: int=0,
                            cost_func=np.sum, method: str="lower", sample:
                            bool=True):
    """Computing the lower/upper bounds on the survival probability of a function of dependent risks
    
    This function performs the RA and calculates the lower and upper bounds on
    the survival probability of function of dependent random variables.  
    For mathematical details, see [1]_.


    Parameters
    ----------
    quant : list
        List of marginal quantile functions

    s_level : float
        Value of the function of the random variables at which the survival
        probability bounds are calculated.

    num_steps : int
        Number of discretization points

    abstol : float
        Absolute convergence tolerance

    lookback : int
        Number of column rearrangements to look back for deciding about
        convergence. Must be a number in :math:`\\{1, ..., \\text{max_ra}-1\\}`.
        If set to zero, it defaults to ``len(quant)``.

    max_ra : int
        Number of column rearrangements. If zero, it defaults to infinitely
        many.

    cost_func : callable
        Callable function that represent the function which is applied to the
        dependent random variables. It needs to accept the ``axis=1`` keyword
        argument and handle it in the numpy style, i.e., taking a 2D-array as
        input and returning a column-vector.

    method : str
        Determine if the lower or upper bound on the expected value is computed.
        Valid options are:

        - `lower`: for the lower bound
        - `upper`: for the upper bound

    sample : bool
        Indication whether each column of the two working matrices is randomly
        permuted before the rearrangements begin


    Returns
    -------
    bound_low : float
        Lower bound on the survival probability

    x_ra_low : numpy.array
        Rearranged matrix for the lower bound

    bound_up : float
        Upper bound on the survival probability

    x_ra_up : numpy.array
        Rearranged matrix for the upper bound


    References
    ----------
    .. [1] G. Puccetti and L. Rüschendorf, "Computation of sharp bounds on the
           distribution of a function of dependent risks," Journal of
           Computational and Applied Mathematics, vol. 236, no. 7, pp.
           1833-1840, Jan. 2012.
    """
    if lookback == 0:
        lookback = len(quant)
    method = method.lower()

    if abstol < 0:
        raise ValueError("Absolute tolerance needs to be non-negative!")
    if num_steps < 2:
        raise ValueError("Number of discretization points needs to be at least 2")

    if method in ['upper']:
        prob_under = lambda alpha: alpha + (1-alpha)*np.arange(num_steps)/num_steps
        prob_over = lambda alpha: alpha + (1-alpha)*np.arange(1, num_steps+1)/num_steps
        optim_func = min
    elif method in ['lower']:
        prob_under = lambda alpha: alpha*np.arange(num_steps)/num_steps
        prob_over = lambda alpha: alpha*np.arange(1, num_steps+1)/num_steps
        optim_func = max
    else:
        raise NotImplementedError

    def find_new_alpha(s, prob_alpha, alpha, alpha_low=0., alpha_high=1.):
        x_mat_alpha = create_matrix_from_quantile(quant, prob_alpha, level=alpha)
        x_ra = basic_rearrange(x_mat_alpha, tol=abstol, tol_type="absolute",
                lookback=lookback, max_ra=max_ra, optim_func=optim_func,
                cost_func=cost_func)
        #x_ra = np.sort(x_mat_alpha, axis=0) # to test comonotonic
        psi_x_ra = cost_func(x_ra, axis=1)
        _g = optim_func(psi_x_ra)
        if _g >= s:
            alpha_high = alpha
            alpha = (alpha + alpha_low)/2.
        else:
            alpha_low = alpha
            alpha = (alpha + alpha_high)/2.
        return alpha, alpha_low, alpha_high, x_ra

    def alpha_loop(alpha, alpha_low, alpha_high, prob_func):
        while np.abs(alpha_high - alpha_low) > 1e-4:
            prob_alpha = prob_func(alpha)
            alpha, alpha_low, alpha_high, x_ra = find_new_alpha(s_level,
                    prob_alpha, alpha, alpha_low, alpha_high)
            #print(alpha, alpha_low, alpha_high)
        return alpha, x_ra

    alpha = .5
    alpha_low = 0.
    alpha_high = 1.
    alpha_under, x_ra_under = alpha_loop(alpha, alpha_low, alpha_high, prob_under)
    alpha_over, x_ra_over = alpha_loop(alpha, alpha_low, alpha_high, prob_over)
    #return (alpha_under, x_ra_under), (alpha_over, x_ra_over)
    return (1.-alpha_under, x_ra_under), (1.-alpha_over, x_ra_over)



def create_comonotonic_ra(level: float, quant, num_steps: int=10):
    """Creating a matrix with comonotonic random variables

    This function creates a rearrangement matrix that represents a joint
    distribution of comonotonic random variables for a given confidence level.
    Both upper and lower bound approximations are returned.  
    For mathematical details, see [1]_


    Parameters
    ----------
    level : float
        Confidence level between 0 and 1.

    quant : list
        List of marginal quantile functions

    num_steps : int
        Number of discretization points


    Returns
    -------
    x_ra_low : numpy.array
        Lower bound approximation of the rearrangement matrix.

    x_ra_up : numpy.array
        Upper bound approximation of the rearrangement matrix.

    References
    ----------
    .. [1] P. Embrechts, G. Puccetti, and L. Rüschendorf, "Model uncertainty
           and VaR aggregation," Journal of Banking & Finance, vol. 37, no. 8,
           pp.  2750-2764, Aug. 2013.
    """
    if num_steps < 2:
        raise ValueError("Number of discretization points needs to be at least 2")

    #prob_under = level*np.arange(num_steps)/num_steps
    #prob_over = level*np.arange(1, num_steps+1)/num_steps
    prob_under = level + (1-level)*np.arange(num_steps)/num_steps
    prob_over = level + (1-level)*np.arange(1, num_steps+1)/num_steps

    x_mat_under = create_matrix_from_quantile(quant, prob_under, level)
    x_mat_over = create_matrix_from_quantile(quant, prob_over, level)
    x_ra_low = np.sort(x_mat_under, axis=0)
    x_ra_up = np.sort(x_mat_over, axis=0)
    return x_ra_low, x_ra_up
