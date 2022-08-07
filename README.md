<img src="PyRID_Logo_cropped.png" alt="PyRID" height=150px>

PyRID: A Brownian dynamics simulator for reacting and interacting particles written in python.

Disclaimer: PyRID is currently in early state development (version 0.1). As such, there is no guaranty that any of the functions are bug free. Careful validation of any results is recommended.

PyRID is a tool for particle based reaction diffusion simulations and is highly influenced and inspired by 

- `ReaDDy <https://readdy.github.io/>`_
- `MCell <https://mcell.org/>`_
- `Sarkas <https://murillo-group.github.io/sarkas/>`_

PyRID is an attempt to combine several features of the above mentioned tools, which include

- unimolecular and bimolecular Reactions (ReaDDy, MCell)
- pair-interactions (ReaDDy)
- mesh based compartments and surface diffusion (MCell)
- pure Python implementation (Sarkas)

Documenttaion
=============

Go to the PyRID documentation -> https://moritzb90.github.io/PyRID_doc/

Installation
============

1. Install Anaconda python distribution (recommended):

   [Anaconda](https://www.anaconda.com/products/distribution)


2. Download PyRID from the GitHub repository and extract it:

   [PyRID on GitHub](https://github.com/MoritzB90/PyRID)

3. Open Anaconda Prompt on your computer.

4. Go to the directory to which you downloaded PyRID:
   
   ```
   $ cd <your directory>
   ```

5. Pip install PyRID:

   ```
   $ pip install pyrid
   ```

To edit your python scripts and run pyrid we recommend to use Spyder or Jupyter Lab, which both come with the Anaconda distribution.