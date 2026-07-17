hodinerva
============

This library contains code for Halo Occupation Distribution (HOD) modeling of the MINERVA-JWST survey dataset.


Installation
------------
To install halofit into your environment from the source code::

    $ cd /path/to/root/hodinerva
    $ pip install -e .

Testing
-------
To run the suite of unit tests::

    $ cd /path/to/root/hodinerva
    $ pytest

To build html of test coverage::

    $ pytest -v --cov --cov-report html
    $ open htmlcov/index.html

