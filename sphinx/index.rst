.. PyRID documentation master file, created by
   sphinx-quickstart on Wed Jun  8 11:28:11 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. sphinx-build -E -a -b html D:\GitHub\PyRID\sphinx D:\GitHub\PyRID_doc_revised\docs
.. sphinx-build -b html D:\GitHub\PyRID\sphinx D:\GitHub\PyRID_doc_revised\docs

==============================================================================================
PyRID: A Brownian dynamics simulator for reacting and interacting particles written in Python.
==============================================================================================


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


Introduction
============

Several tools have been developed that target the simulation of cell biological processes. MCell and Smoldyn, e.g., are popular representatives of the category of particle-based reaction diffusion simulators. Both are powerful tools but cannot simulate particle-particle interactions. Another tool, ReaDDy, has also been developed to fill the gap between classical molecular dynamics and particle-based reaction-diffusion simulations and allows for reactions and interactions between particles. So why do we need PyRID then? While ReaDDy combines some of the features of MCell, Smoldyn, and classical MD, some features are missing. 
   
   #. PyRID supports triangulated mesh geometries, whereas ReaDDy only supports spherical or box-shaped compartments via external potentials. While these two shapes are sufficient in some situations, they are not in others where we want to, e.g., combine compartments to investigate narrow escape properties or where we are interested in the effect of more complex shapes on molecule dynamics. 
   #. PyRID supports rigid bead models of molecules. Rigid bead models are a method of minimal coarse-graining that have some essential benefits. A rigid bead topology replaces strong, short ranged interactions between atoms. Thereby, integration time steps can be several orders larger than in atomic-scale simulations. Usually, the beads of a rigid bead model do not represent single atoms but pattern the geometry of the molecule of interest :cite:p:`Torre2001`, significantly reducing the number of particles that need to be simulated. In addition, experimentally or theoretically estimated diffusion tensors can be used to describe the diffusive motion of molecules accurately. Importantly, patches on the bead model surface can describe multivalent protein-protein interactions.
   #. PyRID accounts for polydispersity of particle sizes. Polydispersity, however, becomes a critical performance issue if we would like to simulate proteins of very different sizes or even proteins in the presence of large diffusing structures such as synaptic vesicles. PyRID uses a hierarchical grid data structure such that polydisperse systems can be simulated with basically no performance loss.
   #. PyRID supports fixed concentration boundary conditions. These can be useful if we would like to simulate sub-regions within a larger system. PyRID supports fixed concentration boundaries for volume and surface molecules and even supports the overlap of mesh compartments with the simulation box boundary.
   #. PyRID supports simulations in the NPT ensemble using the Berendsen barostat.
   #. PyRID can be modified and expanded reasonably easily by anyone with solid knowledge of the Python programming language. Also, by using technologies such as Cython, PyPy, and Numba, Python can reach up to C speed! Using Numba, the MD tool Sarkas has shown that Python is even suited to do molecular dynamics simulations, although still limited to a single core and single thread computation! Inspired by Sarkas, PyRID is also solely written in Python by extensively using Numbas jit compilation. Thereby, PyRID can achieve comparable performance to that of ReaDDy. In addition, PyRID is lightweight, with the core modules having less than 10.000 lines of code (excluding documentation).


.. image:: /Graphics/PyRID_Overview.png
  :width: 100%


Feature comparison
------------------

Please note that this feature comparison is not complete and biased towards PyRID as only the main features of PyRID are compared to the other tools. Each of the tools mentioned here have some unique abilities and features that are not necessarily supported by the other tools/PyRID. However, to do an all-encompassing comparison would go beyond the scope of this documentation.

.. role:: gbg

.. raw:: html

   <style>
      .gbg {background-color:#00ff00;} 
   </style>


.. |br| raw:: html
   
   <style>
   br {
   display: block;
   margin-top:4px; 
   line-height:10px;
   }
   </style>

   <br>

.. |Gdot| raw:: html

   <style>
   .Gdot {
     height: 15px;
     width: 15px;
     background-color: #84d674;
     border-radius: 50%;
     display: inline-block;
   }
   </style>

   <span class="Gdot"></span>

.. |Ydot| raw:: html

   <style>
   .Ydot {
     height: 15px;
     width: 15px;
     background-color: #ebcd63;
     border-radius: 50%;
     display: inline-block;
   }
   </style>

   <span class="Ydot"></span>


.. |Rdot| raw:: html

   <style>
   .Rdot {
     height: 15px;
     width: 15px;
     background-color: #eb7a63;
     border-radius: 50%;
     display: inline-block;
   }
   </style>

   <span class="Rdot"></span>

.. list-table:: Feature comparison
   :widths: 1 1 1 1 1
   :header-rows: 1

   * - Features
     - PyRID
     - `ReaDDy <https://readdy.github.io/>`_
     - `MCell <https://mcell.org/>`_
     - `Smoldyn <https://www.smoldyn.org/>`_
   * - **Reactions**
     - |Gdot| Very Good (zeroth order, unimolecular, |br|
       bimolecular, arbitrary number of products, |br|
       compartment specific, different reaction paths)
     - |Ydot| Good
     - |Gdot| Excellent (Integration with BioNetGen)
     - |Gdot| Excellent (Integration with BioNetGen)
   * - **Reaction accuracy**
     - |Ydot| Volume: Good (Not exact close to |br|
       boundary, reversible fusion reactions of |br|
       interacting particles do not obey detailed balance) |br|
       |Ydot| Surface: Good (euclidian distance only)
     - |Gdot| Volume: Very Good (Not exact close to boundary, |br|
       reversible fusion reactions obey detailed balance) |br|
       |Ydot| Surface: Good (euclidian distance only)
     - |Gdot| Volume: Very Good, |br|
       |Gdot| Surface: Very Good
     - |Gdot| Volume: Very Good, |br|
       |Gdot| Surface: Very Good
   * - **Diffusion**
     - |Gdot| Anisotropic translational and rotaional |br|
       diffusion, integrated diffusion tensors |br|
       calculation
     - |Ydot| Isotropic translational diffusion
     - |Ydot| Isotropic translational diffusion
     - |Ydot| Anisotropic translational diffusion
   * - **Molecular structure**
     - |Gdot| Molecules modeled explicitly |br|
       (by interaction potential and |br|
       /or rigid bodies).
     - |Ydot| Molecules modeled explicitly |br|
       (only by interaction potential).
     - |Rdot| Indirectly by internal state |br|
       variables (only point particles).
     - |Rdot| Indirectly by internal state variables |br|
       (spherical particle approximation).
   * - **Surfaces**
     - |Gdot| Arbitrary surfaces |br|
       (triangulated mesh, supports obj. format)
     - |Ydot| Only via external potentials (Box and Sphere)
     - |Gdot| Arbitrary surfaces |br|
       triangulated mesh, blender interface)
     - |Gdot| Arbitrary surfaces (6 elementary shapes, |br|
       custom format)
   * - **Interactions**
     - |Gdot| Selection of several pair-potentials, |br|
       custom potentials can be added easily.
     - |Gdot| Selection of 4 potentials, custom potentials |br|
       require altering C++ source code.
     - |Rdot| No Interactions
     - |Rdot|/|Ydot| Excluded volume approximation for spheres.
   * - **Boundary Conditions**
     - |Gdot| Periodic, Repulsive, Fixed concentration |br|
     - |Ydot| Periodic, Repulsive
     - |Gdot| Periodic, Repulsive, Fixed concentration
     - |Gdot| Periodic, Repulsive, Fixed concentration |br|
   * - **Polydispersity**
     - |Gdot| Efficient simulation of polydisperse system |br|
       by the use of a hierarchical grid data structure
     - |Ydot| Polydisperse systems result in performance drop.
     - Does not apply
     - Does not apply
   * - **API**
     - |Gdot| Python
     - |Gdot| Python
     - |Gdot| Blender GUI, Python
     - |Gdot| Python, Text based
   * - **Modifiability**
     - |Gdot| Excellent (All source code in Python, |br|
       little dependencies)
     - |Ydot| Ok (Requires changing C++ source code)
     - |Ydot| Ok (Requires changing C++ source code)
     - |Ydot| Ok (Requires changing C++ source code, |br|
       Libsmoldyn API) |br|


Contents
--------

   .. toctree::
      :maxdepth: 2
      :caption: Users

      Users/Installation
      Users/User_Guide/Contents
      Users/Examples/Contents
      Users/Advanced/Contents

   .. toctree::
      :maxdepth: 3
      :caption: Developers

      Developers/Overview
      Developers/Developer API/Contents
      Developers/Theory/Contents
      Developers/Validation/Contents

   .. toctree::
      :maxdepth: 1
      :caption: References

      References


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



.. toctree::
   :caption: Links

   GitHub <https://github.com/MoritzB90/PyRID>
   PyPI <https://pypi.org/>
   Youtube <https://www.youtube.com/channel/UC4o41QLwsfeh0g981MZPl7w>
   license