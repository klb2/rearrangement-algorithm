import numpy as np
from scipy import stats
import pytest

from rearrange_algorithm import bounds_var


@pytest.mark.parametrize("alpha,expected", [(.5102, .5), (.1111, 2.0), (.16, 1.5)])
def test_var_pareto_lower(alpha, expected):
    """
    From the paper "Computation of sharp bounds on the distribution of a
    function of dependent risks" (Puccetti et al., 2012). (Table 1)
    """
    qf = [stats.pareto(2, loc=-1).ppf]*3
    results_under, results_over = bounds_var(1.-alpha, qf, method="lower",
                                             num_steps=500)
    var_under = results_under[0]
    var_over = results_over[0]
    assert ((var_under <= var_over) and (expected*.98 < var_under < expected)
            and (expected < var_over < expected*1.02))

@pytest.mark.parametrize("alpha,expected", [(.142, 10.), (.04537, 20.), (.02204, 30.)])
def test_var_pareto_upper(alpha, expected):
    """
    From the paper "Computation of sharp bounds on the distribution of a
    function of dependent risks" (Puccetti et al., 2012). (Table 1)
    """
    qf = [stats.pareto(2, loc=-1).ppf]*3
    results_under, results_over = bounds_var(1.-alpha, qf, method="upper",
                                             num_steps=500)
    var_under = results_under[0]
    var_over = results_over[0]
    assert ((var_under <= var_over) and (expected*.99 < var_under < expected)
            and (expected < var_over < expected*1.01))



@pytest.mark.parametrize("alpha,expected", [(.1615, .001), (.064, 0.003), (.025, .005)])
def test_var_pareto_lower_product(alpha, expected):
    """
    From the paper "Computation of sharp bounds on the distribution of a
    function of dependent risks" (Puccetti et al., 2012). (Table 1)
    """
    theta = [1.5, 1.8, 2.0, 2.2, 2.5]
    qf = [stats.pareto(t, loc=-1).ppf for t in theta]
    results_under, results_over = bounds_var(1.-alpha, qf, method="lower",
                                             num_steps=500, cost_func=np.prod)
    var_under = results_under[0]
    var_over = results_over[0]
    #print(var_under, var_over)
    assert ((var_under <= var_over) and (expected-1e-3 < var_under < expected)
            and (expected < var_over < expected+1e-3))

@pytest.mark.parametrize("alpha,expected", [(.216, 100.), (.160, 300.), (.138, 500.)])
def test_var_pareto_upper_product(alpha, expected):
    """
    From the paper "Computation of sharp bounds on the distribution of a
    function of dependent risks" (Puccetti et al., 2012). (Table 1)
    """
    theta = [1.5, 1.8, 2.0, 2.2, 2.5]
    qf = [stats.pareto(t, loc=-1).ppf for t in theta]
    results_under, results_over = bounds_var(1.-alpha, qf, method="upper",
                                             num_steps=1000, cost_func=np.prod)
    var_under = results_under[0]
    var_over = results_over[0]
    #print(var_under, var_over)
    assert ((var_under <= var_over) and (expected*.98 < var_under < expected)
            and (expected < var_over < expected*1.02))
