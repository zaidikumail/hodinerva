halofit
============

This library provides tools for fitting observed data, such as galaxy-galaxy clustering, and galaxy abundance with halo models.


Installation
------------
To install halofit into your environment from the source code::

    $ cd /path/to/root/halofit
    $ pip install -e .

Testing
-------
To run the suite of unit tests::

    $ cd /path/to/root/halofit
    $ pytest

To build html of test coverage::

    $ pytest -v --cov --cov-report html
    $ open htmlcov/index.html

