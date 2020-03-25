************
Installation
************

The installation of McRadar is easy, and it follows the standard installation procedure of the Python packages. 

Requirements
============

- `Python <http://www.python.org/>`_  (recommended >=3.7)
- `Numpy <http://www.numpy.org>`_  (>=1.11)
- `Scipy <https://www.scipy.org>`_  (>=0.18.0)
- `Xarray <http://xarray.pydata.org>`_ (>=0.14.1)
- `Pandas <https://pandas.pydata.org/>`_ (>=0.25.3)
- `gfortran <https://gcc.gnu.org/>`_
- `PyTmatrix <https://github.com/jleinonen/pytmatrix>`_ (>=0.3.1)

Installation at Linux OS
========================

Installing dependencies
-----------------------

All of the required packages can be installed using pip. In case you do not have pip installed yet, you can find an installation description `here <https://pip.pypa.io/en/stable/installing/>`_.

.. code-block:: console

    $ pip install 'name of the required package'

Installing McRadar
------------------

You have two for acquiring the package. The first option is to download it from the `McRadar repository <https://github.com/jdiasn/McRadar>`_, and the second option is to clone it using git. In case you do not have git installed, you can find the git installation guide `here <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_.

.. code-block:: console
 
    $ git clone https://github.com/jdiasn/McRadar.git

After the previous step, you should be able to execute the following command

.. code-block:: console

    $ python setup.py install


Installation at MacOS
=====================

Installing dependencies
-----------------------

All the dependencies can be installed using pip as described previously. However, the installation of PyTmatrix requires gfortran, and it can be installed using Homebrew. In case you do not have Homebrew installed, you can find an installation guide `here <https://brew.sh/>`_.


.. code-block:: console

    # first step
    $ brew cask install gfortran
    # second step
    $ pip install pytmatrix

Installing McRadar
------------------

It follows the same installation procedure described for Linux OS. After downloading or cloning the McRadar package, you only need to execute the following command

.. code-block:: console

	$ python setup.py install 

