============
Installation
============

First, clone this repository to your local machine::

    git clone https://github.com/spacetelescope/exovetter.git

Then, in the source directory, do the following to install it as an
"editable" install, along with all the dependencies::

    pip install -e .[test,docs,all]

To see whether your installation is successful, you can try to import the
package in a Python session:

    >>> import exovetter
