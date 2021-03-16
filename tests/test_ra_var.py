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
    #alpha = .5102
    results_low_under, results_low_over = bounds_var(
            1.-alpha, qf, method="lower", num_steps=500)
    var_under = results_low_under[0]
    var_over = results_low_over[0]
    #assert ((var_under <= var_over) and (.4998 < var_under < .5) and
    #        (.5 < var_over < .51))
    assert ((var_under <= var_over) and (expected*.98 < var_under < expected)
            and (expected < var_over < expected*1.02))
