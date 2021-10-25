# Rearrangement Algorithm

[![PyPI version](https://badge.fury.io/py/rearrangement-algorithm.svg)](https://badge.fury.io/py/rearrangement-algorithm)
[![Read the docs status](https://readthedocs.org/projects/rearrangement-algorithm/badge/?version=latest&style=flat)](https://rearrangement-algorithm.readthedocs.io/en/dev/)


Python implementation of a rearrangement algorithm that can be used to
calculate bounds on the value-at-risk (VaR) of dependent risks.

The current version includes the following calculations:
- Upper and lower bounds on the VaR of dependent risks[^embrechts2013]
- Upper and lower bounds on the survival probability of functions of dependent
  risks[^puccetti2012]
- Upper and lower bounds on the expected value of supermodular functions of
  dependent risks[^puccetti2015]


## Mathematical Background
Mathematical details and derivations can be found in the
publications[^embrechts2013][^puccetti2012][^puccetti2015].

More information and an overview of extensions can be found on the website of
the [Rearrangement Algorithm
project](https://sites.google.com/site/rearrangementalgorithm/).



## Implementation
Parts of this implementation are based on the
[qrmtools](https://cran.r-project.org/package=qrmtools) R package (version
0.0-13) by M. Hofert, K. Hornik, and A. J. McNeil.
Details on the algorithm and the R implementation can be found in the paper
["Implementing the Rearrangement Algorithm: An Example from Computational Risk
Management", M. Hofert, In: Risks, vol. 8, no. 2,
2020](https://doi.org/10.3390/risks8020047)[^hofert2020].


## Installation
You can install the package via pip
```bash
pip install rearrangement-algorithm
```

If you want to install the latest (unstable) version, you can install the
package from source
```bash
git clone https://gitlab.com/klb2/rearrangement-algorithm.git
cd rearrangement-algorithm
git checkout dev
pip install .
```

## Documentation
You can find the documentation for this package on [Read the
Docs](https://rearrangement-algorithm.readthedocs.io/).  
This also includes some usage examples.


## License and Referencing
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite this package.

Parts of this code are based on the
[qrmtools](https://cran.r-project.org/package=qrmtools) R package (version
0.0-13), which is also released under the GPLv3 license.


## References
[^embrechts2013]: P. Embrechts, G. Puccetti, and L. Rüschendorf, "Model uncertainty and VaR aggregation," J. Bank. Financ., vol. 37, no. 8, pp. 2750-2764, Aug. 2013. [doi:10.1016/j.jbankfin.2013.03.014](https://doi.org/10.1016/j.jbankfin.2013.03.014)

[^puccetti2015]: G. Puccetti and L. Rüschendorf, "Computation of Sharp Bounds on the Expected Value of a Supermodular Function of Risks with Given Marginals," Commun. Stat. - Simul. Comput., vol. 44, no. 3, pp. 705-718, Mar. 2015. [doi:10.1080/03610918.2013.791368](https://doi.org/10.1080/03610918.2013.791368)

[^puccetti2012]: G. Puccetti and L. Rüschendorf, "Computation of sharp bounds on the distribution of a function of dependent risks," J. Comput. Appl. Math., vol. 236, no. 7, pp. 1833-1840, Jan. 2012. [doi:10.1016/j.cam.2011.10.015](https://doi.org/10.1016/j.cam.2011.10.015)

[^hofert2020]: M. Hofert, "Implementing the Rearrangement Algorithm: An Example from Computational Risk Management," Risks, vol. 8, no. 2, May 2020. [doi:10.3390/risks8020047](https://doi.org/10.3390/risks8020047)
