=======
McRadar
=======

.. image:: https://zenodo.org/badge/249136844.svg
    :target: https://zenodo.org/badge/latestdoi/249136844

.. image:: https://readthedocs.org/projects/mcradar/badge/?version=latest
    :target: https://mcradar.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

`McRadar <https://github.com/jdiasn/McRadar>`_ is an Open Source Python package to simulate the multi-frequency radar
variables using the output from `McSnow <https://doi.org/10.1002/2017MS001167>`_.
The package is built on top of the `PyTmatrix <https://github.com/jleinonen/pytmatrix>`_.

McRadar was initially idealized during the Ice Microphysics workshop (at mount Zugspitze)
to allow easy verification of the radar variables through the evolution of the ice
microphysics. This package inherits several of my ideas that I developed when I was trying
to use PyTmatrix to reproduce observed bimodal spectra.

----------
Limitation
----------

- This package is still not able to deal with particles with an extreme aspect ratio (e.g. larger than 6 or much smaller 0.1 ).

- This initial release only provides the spectra and KDP, but all the additional radar variable will be implemented soon.

-------------
Documentation
-------------

The documentation and a usage guide are available at `https://mcradar.readthedocs.io <https://mcradar.readthedocs.io>`_.
