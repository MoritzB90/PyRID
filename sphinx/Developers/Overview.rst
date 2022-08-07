Overview
========

**Developer API**

The Developer API section contains the documentation for all the functions and classes used in PyRID. PyRID relies heavily on `Numba <https://numba.pydata.org/>`_. One thing to keep in mind when implementing new functionalities or changing existing code is that Numba will not do bound checking by default. If somewhere in the code an element of, e.g., a numpy array is accessed that is out of bounds, the kernel might just die without you getting any error message. Therefore, during development it is advisable to enable bound checking, which can be done in your python script by

.. code-block:: python

	from numba import config
	config.BOUNDSCHECK = True # False # 

Bound checking will, however, slightly slow down your code.

**Style**

I try to follow the PEP 8 programming style recommendation for python (https://peps.python.org/pep-0008/, https://realpython.com/python-pep8/). However, I may not be super consistent all the time.
Variables are usually written in lower case 'my_variable' and for classes, each word starts with capital letters ('MyClass'). However, I write instances of PyRID classes also starting with capital letters! Also, sometimes it makes more sense to start variables with an upper case letter, e.g., for certain physical properties such as the diffusion coefficient. In such cases I do deviate from the PEP 8 style guide.

**Side noted**

If you want to compare simulation results after making changes to the code, make sure that you use the same random seed and also that everything is sorted.
If, e.g., the resolution of the hierarchical grid is changed, the order in which reactions between particles are added to the reaction handler list will change. Thereby, even though the seed is the same, randomly picking a reaction event from the list will be different. Therefore, activate

- sorting in get_random(self) and Unique_Pairing in append_reaction(self, ...) of the reaction_registry_util module.

to make sure reactions are sorted by the educt tuples.
Also, if you make any changes to the particles or rigid body data structure you may also want to sort particles by their id whenever looping over particle or molecules.

**Theroy**

In the Theory section, I will introduce and discuss the main methods used in PyRID. I start by introducing the scheme by which bead molecules are represented. Followed by the derivation of an algorithm for the propagation of translational and predominantly rotational diffusion. The rotational and translational mobility tensors dictate the translational and rotational motion of anisotopic rigid bodies. Therefore, I outline the calculation of the mobility tensors based on a modified Oseen tensor :cite:p:`Carrasco1999`. One of the main features of PyRID that distinguishes it from other molecular dynamics tools such as LAMMPS, Gromacs and HooMD is the ability to simulate arbitrary unimolecular and bimolecular reactions using a special stochastic simulation algorithm. I describe how these reactions are evaluated in PyRID. Another notable feature of PyRID is its ability to restrict the motion of molecules to complex compartment geometries represented by triangulated meshes. A brief overview of how compartment collisions and surface diffusion are handled is given.

**Validation**

In the Validations section you find some code examples for the validation of certain PyRID functionalities.