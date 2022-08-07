.. _userguide_observables:

===========
Observables
===========

A simulation is worth little, if we are not able to sample system properties.
PyRID supports several observables to be sampled:

#. Energy
#. Pressure
#. Virial
#. Virial tensor
#. Volume (only useful for NPT ensemble simulations)
#. Number
#. Bonds
#. Reactions
#. Position
#. Orientation
#. Force
#. Torque
#. Radial distribution function

Each observable (except Volume) is sampled per molecule type or molecule/particle pair, in the case of bimolecular reactions and bonds.
Observables are written into an .hdf5 file. In addition, system properties such as temperature, box size and the molecule topologies and diffusion properties are written to the .hdf5 file such that one can in principle recover the hole simulation even if the corresponding python script is lost (Not yet true, since the reaction details are not written to the file. However, I will try and add this feature within the next updates).
I recommend to use the software HDF Compass if you want to visually inspect the .hdf5 file. In addition, PyRID has the Evaluation module with which the hdf5 file data can be read, plotted and analyzed.

To observe a system property/measure you simply call the observe method and pass the property name and a list of the molecule types you would like to be observed:

**Energy**

.. code-block:: python
	
	Simulation.observe('Energy', molecules = ['A', 'B'], obs_stride = obs_stride, stepwise = False, binned = False)

If no molecules list is passed, PyRID will sample the property for all molecule types. In addition you can set the stride by which PyRID will sample the respective measure. Also, you can tell PyRID to bin the data. If binned is set to True, PyRID will sum up values in between sampling steps. Especially if you sample the number of reactions it makes sense to turn on binning, since you usually want to know how many reactions occurred within a time interval. For other measures binning is mainly useful if you are interested in mean values. By default binned is False and stepwise is set to True. Stepwise sampling is just regular sampling without the binning. You can turn on stepwsie a binned sampling in parallel.

**Pressure**

.. code-block:: python

	Simulation.observe('Pressure', molecules = ['A'], obs_stride = obs_stride)


**Virial**

.. code-block:: python
	
	Simulation.observe('Virial', molecules = ['A'], obs_stride = obs_stride, binned = True)


**Virial tensor**

.. code-block:: python
	
	Simulation.observe('Virial Tensor', molecules = ['A'], obs_stride = obs_stride, binned = True)


**Volume**

.. code-block:: python
	
	Simulation.observe('Volume', obs_stride = obs_stride, binned = True)


**Number**

.. code-block:: python
	
	Simulation.observe('Number', molecules = ['A', 'B', 'C', 'B2', 'D', 'E', 'F'], obs_stride = obs_stride, binned = True, stepwise = True)


**Bonds**

.. code-block:: python
	
	Simulation.observe('Bonds', obs_stride = obs_stride)
	

**Reactions**

.. code-block:: python
	
	Simulation.observe('Reactions', obs_stride = obs_stride, binned = True)


**Position**

.. code-block:: python

	Simulation.observe('Position', molecules = ['A'], obs_stride = obs_stride)


**Orientation**

.. code-block:: python
	
	Simulation.observe('Orientation', molecules = ['A'], obs_stride = obs_stride)


**Force**

.. code-block:: python
	
	Simulation.observe('Force', molecules = ['A', 'B', 'C'], obs_stride = obs_stride, binned = True)


**Torque**

.. code-block:: python
	
	Simulation.observe('Torque', molecules = ['A', 'B', 'C'], obs_stride = obs_stride)


**Radial distribution function**

.. code-block:: python
	
	Simulation.observe_rdf(rdf_pairs = [['A','A'],['A','B'],['A','C']], rdf_bins = [100,50,50], rdf_cutoff = [20.0,10.0,8.0], stride = obs_stride)

If you sample the radial distribution function (RDF), you need to define the number of bin (spatial resolution) and the cutoff distance for each molecule pair that is sampled.
