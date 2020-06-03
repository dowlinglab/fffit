fffit
=====

**fffit** is a Python package for fitting molecular mechanics
force fields to experimental properties by using Gaussian process
surrogate models to rapidly screen through large parameter spaces.

.. warning::

    **fffit** is still in early development (0.x releases). The API may
    change unexpectedly.

Installation
~~~~~~~~~~~~

Installation is currently only available from source. We recommend
installing the package within a dedicated conda environment:

.. code-block:: bash

    git clone https://github.com/rsdefever/fffit.git
    cd fffit/
    conda create --name fffit --file requirements.txt
    conda activate fffit
    pip install .

A conda installation may be added in the future.

Credits
~~~~~~~

Development of fffit was supported by the National Science Foundation
under grant NSF Grant Number 1835874 and NSF Grant Number XX.
Any opinions, findings, and conclusions or recommendations expressed
in this material are those of the author(s) and do not necessarily
reflect the views of the National Science Foundation.
