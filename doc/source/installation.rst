Installation
============

You can install the package via pip

.. code-block:: bash
    :linenos:

    pip install rearrangement-algorithm

If you want to install the latest (unstable) version, you can install the
package from source

.. code-block:: bash
    :linenos:

    git clone https://gitlab.com/klb2/rearrangement-algorithm.git
    cd rearrangement-algorithm
    git checkout dev
    pip install .

You can test, if the installation was successful, by importing the package

.. code-block:: python
    :linenos:

    import rearrangement_algorithm as ra
    print(ra.__version__)
