.. _install_instructions:

============
Installation
============

``exovetter`` requires the following packages on install:

* numpy
* astropy
* scipy

We also recommend you have the following packages install to access all the
functionality that ``exovetter`` provides:

* matplotlib
* scikit-learn
* lightkurve
* lpproj

.. _install_stable:

Install the Stable Version
==========================

.. warning:: Do not use the released version yet. This package is still in heavy development. See :ref:`install_dev` instead.

To install the released version from PyPI::

    pip install exovetter[all]

.. _install_dev:

Install the Development Version
===============================

First, clone this repository to your local machine::

    git clone https://github.com/spacetelescope/exovetter.git

Then, in the source directory, do the following to install it as an
"editable" install, along with all the dependencies::

    pip install -e .[test,docs,all]

Post-Install
============

To see whether your installation is successful, you can try to import the
package in a Python session:

    >>> import exovetter
