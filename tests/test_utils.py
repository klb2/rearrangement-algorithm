import numpy as np
from scipy import stats
import pytest

from rearrangement_algorithm import (create_matrix_from_quantile,
                                     create_comonotonic_ra, basic_rearrange)


@pytest.mark.parametrize("alpha", [0., 1e-3, .1, .2, .5])
def test_quantile_matrix_uniform(alpha):
    """
    From the paper "Model uncertainty and VaR aggregation" (Embrechts et al.,
    2013).
    """
    num_steps = 50
    prob_under = alpha*np.arange(num_steps)/num_steps
    prob_over = alpha*(np.arange(num_steps)+1)/num_steps
    qf = [stats.uniform.ppf]*3
    expected_under = np.array([_qf(prob_under) for _qf in qf]).T
    expected_over = np.array([_qf(prob_over) for _qf in qf]).T
    mat_under = create_matrix_from_quantile(qf, prob_under, level=alpha)
    mat_over = create_matrix_from_quantile(qf, prob_over, level=alpha)
    assert ((np.shape(mat_over) == np.shape(mat_under) == (num_steps, len(qf)))
            and np.allclose(mat_over, expected_over)
            and np.allclose(mat_under, expected_under))

def test_comonotonic_matrix():
    """
    From the paper "Model uncertainty and VaR aggregation" (Embrechts et al.,
    2013).
    """
    num_steps = 50
    prob_under = np.arange(num_steps)/num_steps
    prob_over = (np.arange(num_steps)+1)/num_steps
    qf = [stats.uniform.ppf, stats.uniform(1, 2).ppf, stats.uniform(-1, 3).ppf]
    expected_under = np.array([_qf(prob_under) for _qf in qf]).T
    expected_over = np.array([_qf(prob_over) for _qf in qf]).T
    mat_under, mat_over = create_comonotonic_ra(level=0., quant=qf, num_steps=num_steps)
    assert np.all(np.diff(mat_over, axis=0) >= 0) and np.all(np.diff(mat_under, axis=0) >= 0)

def test_rearrange_countermonotonic():
    x_mat = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    expected = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]])
    expected2 = np.array([[5, 1], [4, 2], [3, 3], [2, 4], [1, 5]])
    result = basic_rearrange(x_mat, min)
    assert np.all(result == expected) or np.all(result == expected2)
