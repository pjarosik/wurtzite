# crystalpy
Python package with software for crystallography. 

The package contains functions for:
- crystal structure definition and generation,
- crystal visualization,
- simulation of the crystal defect (dislocations),
- and many more...


## Installation

Requirements:

- Openbabel 3.1.1 at least. This package can be installed in one of the folowing ways:
  - recommended if you are using [Anaconda](https://www.anaconda.com/) (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)) 
    simply install it using the following command: `conda install -c conda-forge openbabel=3.1.1`, 
  - if you are using Ubuntu (or some another Debian-like Linux distribution): `sudo apt install openbabel`
  - then run pip install openbabel>=3.1.1 -L

To install crystalpy run the following command:
```
pip install git+https://github.com/pjarosik/crystalpy.git
```