import numpy as np
from scipy import stats
import pytest

from rearrangement_algorithm import bounds_surv_probability


@pytest.mark.parametrize("s_level,expected", [(1., .25), (1.5, .16), (2., .111), (2.5, .0816)])
def test_surv_prob_pareto_lower(s_level, expected):
    """
    From the paper "Computation of sharp bounds on the distribution of a
    function of dependent risks" (Puccetti et al., 2012). (Table 1)
    """
    qf = [stats.pareto(2, loc=-1).ppf]*3
    results_under, results_over = bounds_surv_probability(qf, s_level,
                                                          method="lower",
                                                          num_steps=750)
    surv_under = results_under[0]
    surv_over = results_over[0]
    assert ((surv_under <= surv_over) and (expected*.98 < surv_under < expected)
            and (expected < surv_over < expected*1.02))

@pytest.mark.parametrize("s_level,expected", [(10., .142), (15, .074), (25, .0306), (30, .022)])
def test_surv_prob_pareto_upper(s_level, expected):
    """
    From the paper "Computation of sharp bounds on the distribution of a
    function of dependent risks" (Puccetti et al., 2012). (Table 1)
    """
    qf = [stats.pareto(2, loc=-1).ppf]*3
    results_under, results_over = bounds_surv_probability(qf, s_level,
                                                          method="upper",
                                                          num_steps=750)
    surv_under = results_under[0]
    surv_over = results_over[0]
    assert ((surv_under <= surv_over) and (expected*.98 < surv_under < expected)
            and (expected < surv_over < expected*1.02))


@pytest.mark.parametrize("s_level,expected", [(0.001, .1615), (.002, .09855), (.005, .0245)])
def test_surv_prob_pareto_lower_product(s_level, expected):
    """
    From the paper "Computation of sharp bounds on the distribution of a
    function of dependent risks" (Puccetti et al., 2012). (Table 2)
    """
    theta = [1.5, 1.8, 2.0, 2.2, 2.5]
    qf = [stats.pareto(t, loc=-1).ppf for t in theta]
    results_under, results_over = bounds_surv_probability(qf, s_level,
                                                          method="lower",
                                                          num_steps=1000,
                                                          cost_func=np.prod)
    surv_under = results_under[0]
    surv_over = results_over[0]
    assert ((surv_under <= surv_over) and (expected-1e-2 < surv_under < expected)
            and (expected < surv_over < expected+1e-2))

@pytest.mark.parametrize("s_level,expected", [(100, .2162), (300, .1598), (500, .1380)])
def test_surv_prob_pareto_upper_product(s_level, expected):
    """
    From the paper "Computation of sharp bounds on the distribution of a
    function of dependent risks" (Puccetti et al., 2012). (Table 2)
    """
    theta = [1.5, 1.8, 2.0, 2.2, 2.5]
    qf = [stats.pareto(t, loc=-1).ppf for t in theta]
    results_under, results_over = bounds_surv_probability(qf, s_level,
                                                          method="upper",
                                                          num_steps=1000,
                                                          cost_func=np.prod)
    surv_under = results_under[0]
    surv_over = results_over[0]
    assert ((surv_under <= surv_over) and (expected-1e-2 < surv_under < expected)
            and (expected < surv_over < expected+1e-2))


@pytest.mark.parametrize("s_level,expected", [(1, .25), (3, .0625), (4, .04)])
def test_surv_prob_pareto_lower_max(s_level, expected):
    """
    From the paper "Computation of sharp bounds on the distribution of a
    function of dependent risks" (Puccetti et al., 2012). (Table 3)
    """
    qf = [stats.pareto(2, loc=-1).ppf]*3
    results_under, results_over = bounds_surv_probability(qf, s_level,
                                                          method="lower",
                                                          num_steps=1500,
                                                          cost_func=np.max)
    surv_under = round(results_under[0], 8)
    surv_over = round(results_over[0], 8)
    assert ((surv_under <= surv_over) and (expected*.98 < surv_under <= expected)
            and (expected <= surv_over < expected*1.01))

@pytest.mark.parametrize("s_level,expected", [(1, .75), (3, .1875), (4, .12)])
def test_surv_prob_pareto_upper_max(s_level, expected):
    """
    From the paper "Computation of sharp bounds on the distribution of a
    function of dependent risks" (Puccetti et al., 2012). (Table 3)
    """
    qf = [stats.pareto(2, loc=-1).ppf]*3
    results_under, results_over = bounds_surv_probability(qf, s_level,
                                                          method="upper",
                                                          num_steps=1500,
                                                          cost_func=np.max)
    surv_under = round(results_under[0], 8)
    surv_over = round(results_over[0], 8)
    print(surv_under, surv_over)
    assert (np.isclose(surv_under, expected, rtol=1e-3) 
            and np.isclose(surv_over, expected, rtol=1e-2))
