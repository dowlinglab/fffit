fffit
=====

**fffit** is a Python package for fitting molecular mechanics
force fields to experimental properties by using Gaussian process
surrogate models to rapidly screen through large parameter spaces.

Warning
~~~~~~~

**fffit** is still in early development (0.x releases). The API may
change unexpectedly.

Installation
~~~~~~~~~~~~

Installation is currently only available from source. We recommend
installing the package within a dedicated venv or conda environment.
Here we demonstrate with a venv environment:

.. code-block:: bash

    git clone https://github.com/rsdefever/fffit.git
    cd fffit/
    python3 -m venv fffit-env
    source fffit-env/bin/activate
    python3 -m pip install -r requirements.txt
    pip install -e .

Note this will make an editable installation so that any of the changes
you make in your ``fffit``.

More information on virtual environments can be found
`here <https://docs.python.org/3/tutorial/venv.html>_`.

Credits
~~~~~~~

Development of fffit was supported by the National Science Foundation
under grant NSF Award Number OAC-1835630 and NSF Award Number CBET-1917474.
Any opinions, findings, and conclusions or recommendations expressed
in this material are those of the author(s) and do not necessarily
reflect the views of the National Science Foundation.
