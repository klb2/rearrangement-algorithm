import numpy as np
from scipy import stats
import pytest

from rearrange_algorithm import bounds_expectation_supermod

def stop_loss_function(x, k=0, *args, **kwargs):
    _sum = np.sum(x, *args, **kwargs) - k
    return np.maximum(_sum, 0)


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


@pytest.mark.parametrize("k,expected", [(0, 3.), (1, 2.), (4, .057159), (5, .020492)])
def test_expect_exp_stop_loss_lower(k, expected):
    """
    From the paper "Computation of Sharp Bounds on the Expected Value of a
    Supermodular Function of Risks with Given Marginals" (Puccetti et al.,
    2015). (Table 3)
    """
    N = int(1e4)
    supermod_func = np.sum
    qf = [stats.expon.ppf]*3
    results_under, results_over = bounds_expectation_supermod(qf, num_steps=N,
            supermod_func=supermod_func, method="lower", max_ra=0)
    expect_under = np.mean(stop_loss_function(results_under[1], k=k, axis=1))
    expect_over = np.mean(stop_loss_function(results_over[1], k=k, axis=1))
    print(expect_under, expect_over)
    assert ((expect_under <= expect_over) and 
            (expected*.95 < expect_under <= expected) and
            (expected <= expect_over < expected*1.03))

@pytest.mark.parametrize("k,expected", [(0, 3.), (1, 2.1496), (5, .56663)])
def test_expect_exp_stop_loss_upper(k, expected):
    """
    From the paper "Computation of Sharp Bounds on the Expected Value of a
    Supermodular Function of Risks with Given Marginals" (Puccetti et al.,
    2015). (Table 3)
    """
    N = int(1e4)
    supermod_func = lambda x, *args, **kwargs: stop_loss_function(x, k=k, *args, **kwargs)
    qf = [stats.expon.ppf]*3
    results_under, results_over = bounds_expectation_supermod(qf, num_steps=N,
            supermod_func=supermod_func, method="upper", max_ra=0)
    expect_under = results_under[0]
    expect_over = results_over[0]
    print(expect_under, expect_over)
    assert ((expect_under <= expect_over) and 
            (expected*.99 < expect_under <= expected) and
            (expected <= expect_over < expected*1.01))
