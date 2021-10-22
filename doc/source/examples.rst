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

The `.ppf` method of a distribution implements the quantile function, which we
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
TODO


Bounds on the Value-at-Risk (VaR)
---------------------------------
TODO


Bounds on the Expected Value
----------------------------
TODO


Bounds on the Survival Probability
----------------------------------
TODO
