Examples
========

In the following, some simple usage examples are given.

Basics
------
We will start with some simple basics that are necessary for all following
examples.

Besides the actual ``rearrangement_algorithm`` package, we will use the
`scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_ module
to specify the marginal distributions.

The ``.ppf`` method of a distribution implements the quantile function, which we
will heavily use in the following examples.

Basic Rearrange
^^^^^^^^^^^^^^^

.. code-block:: python
    :linenos:

    import numpy as np
    import rearrangement_algorithm as ra

    # create a matrix (10x3) where each columns contains numbers 0 to .9
    X = np.tile(np.linspace(0, 1, 10), (3, 1)).T
    print(X)

    # the row sums are varying a lot
    print(np.sum(X, axis=1))

    # rearrange it
    X_ra = ra.basic_rearrange(X, min)
    print(X_ra)

    # the row sums are now very balanced
    print(np.sum(X_ra, axis=1))


Comonotonic Rearrangement
-------------------------
The first example is to generate a comonotonic rearrangement of random
variables with given marginals.  
For the following example, we will use :math:`X_1\sim\exp(1)`,
:math:`X_2\sim\mathcal{N}(0, 1)`, and :math:`X_3\sim\mathcal{U}[0, 1]`.

The important function to create the comonotonic rearrangement is
:py:meth:`rearrangement_algorithm.create_comonotonic_ra`.

.. code-block:: python
    :linenos:

    from scipy import stats
    import rearrangement_algorithm as ra

    # create the random variables
    X1 = stats.expon()  # exponential distribution
    X2 = stats.norm()  # normal distribution
    X3 = stats.uniform()  # uniform distribution

    # quantile functions are given by the .ppf method in scipy
    qf = [X1.ppf, X2.ppf, X3.ppf]

    # create rearrangement
    level = 0.5
    X_ra_lower, X_ra_upper = ra.create_comonotonic_ra(level, qf, 10)

    # all columns are ordered in increasing order
    print(X_ra_lower)
    print(X_ra_upper)


Bounds on the Value-at-Risk (VaR)
---------------------------------
One relevant use case of the rearrangement algorithm is the numerical
approximation of bounds on the value-at-risk (VaR) of dependent risks.

The following example illustrates the use of the
:py:meth:`rearrangement_algorithm.bounds_VaR` function using the following
example from the paper `"Computation of sharp bounds on the distribution of a
function of dependent risks" (Puccetti, Rüschendorf, 2012)
<https://doi.org/10.1016/j.cam.2011.10.015>`_.

Given three Pareto(2)-distributed random variables, we calcuate the lower bound
on the VaR (based on the sum of the three random variables).
For comparison, the numbers can be found in the above paper (Table 1).

.. code-block:: python
    :linenos:

    from scipy import stats
    import rearrangement_algorithm as ra

    # create the quantile functions
    qf = [stats.pareto(2, loc=-1).ppf]*3

    # set the parameters
    alpha = 0.5102
    num_steps = 500

    # calcuate the bounds on the VaR
    lower_under, lower_over = ra.bounds_VaR(1.-alpha, qf, method="lower",
                                            num_steps=num_steps)
    VaR_under = lower_under[0]
    VaR_over = lower_over[0]

    # the expected value is around 0.5
    expected = 0.5
    print("{:.5f} <= {:.3f} <= {:.5f}".format(VaR_under, expected, VaR_over))


Bounds on the Expected Value
----------------------------
TODO


Bounds on the Survival Probability
----------------------------------
TODO
