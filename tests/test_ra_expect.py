import numpy as np
from scipy import stats
import pytest

from rearrange_algorithm import bounds_expectation_supermod


@pytest.mark.parametrize("d,expected", [(3, 5.4802e-2), (5, 6.8604e-3), (20, 2.0612e-9)])
def test_expect_uniform_prod_lower(d, expected):
    """
    From the paper "Computation of Sharp Bounds on the Expected Value of a
    Supermodular Function of Risks with Given Marginals" (Puccetti et al.,
    2015). (Table 1)
    """
    N = int(1e4)
    supermod_func = np.prod
    qf = [stats.uniform.ppf]*d
    results_under, results_over = bounds_expectation_supermod(qf, num_steps=N,
            supermod_func=supermod_func, method="lower", max_ra=0)
    expect_under = results_under[0]
    expect_over = results_over[0]
    assert ((expect_under <= expect_over) and 
            (expected*.96 < expect_under <= expected) and
            (expected <= expect_over < expected*1.02))

@pytest.mark.parametrize("d,expected", [(3, .25), (5, .1667), (20, .0476)])
def test_expect_uniform_prod_upper(d, expected):
    """
    From the paper "Computation of Sharp Bounds on the Expected Value of a
    Supermodular Function of Risks with Given Marginals" (Puccetti et al.,
    2015). (Table 1)
    """
    N = int(1e4)
    supermod_func = np.prod
    qf = [stats.uniform.ppf]*d
    results_under, results_over = bounds_expectation_supermod(qf, num_steps=N,
            supermod_func=supermod_func, method="upper", max_ra=0)
    expect_under = results_under[0]
    expect_over = results_over[0]
    assert ((expect_under <= expect_over) and 
            (expected*.99 < expect_under <= expected) and
            (expected <= expect_over < expected*1.01))
