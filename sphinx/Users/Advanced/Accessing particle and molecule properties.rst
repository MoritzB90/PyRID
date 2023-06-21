==========================================
Accessing particle and molecule properties
==========================================

.. role:: python(code)
   :language: python


PyRID allows users to sample certain properties/observables and writes these to a local file (see :ref:`userguide_observables`).
However, sometimes you may want to access some system properties directly and write your own Python scripts to analyze data or similar.
You can do so via the Simulation class and its attributes. Tip: If you want to have a look at all the different attributes of the Simulation class, use :python:`dir(Simulation)`. 


**Particles**

All particle properties, are kept in a data structure that can be accessed by the Particles attribute

.. code-block:: python
	
	Simulation.Particles

To get the properties of the particle with index 0, write

.. code-block:: python
	
	In[1]: Simulation.Particles[0]
	
	Out[1]: (1, [ -8.01000665242636,  49.9969587092104 , -10.31255250585599], 
		[-2.0197198684247195,  0.2506851204932846, -1.4518569569533009], 
		[-1.1785113019775793, -2.041241452319315 , -0.8333333333333333], 
		[ 40.96887479809036   , -27.908936738251054  ,  -0.14310326836228807], 
		12876, 'Patch_3', 10, 0., 0, [0, 0], True, 9745, 0., 0, 1.e+10)

What is returned is a numpy structured array. The datatype of a structured array is a composition of simpler datatypes (float, int, string ,...) organized as a sequence of named fields. 
Structured arrays are designed to be able to mimic ‘structs’ in the C language. The problem with what is returned above is that we currently don't know how to interpret the data.
Therefore, we first need to take a look at the field names of the structured array datatype. We can do so by

.. code-block:: python
	
	In[2]: Simulation.Particles[0].dtype.names

	Out[2]: ('next',
		'unique_id',
 		'pos',
 		'pos_local',
 		'coord_local',
 		'force',
 		'rb_id',
 		'type',
 		'type_id',
 		'radius',
 		'number_reactions',
 		'reactions_head',
 		'bound',
 		'bound_with',
 		'cutoff',
 		'h',
 		'next_transition')

The first field is just a 'next' pointer, which is not of any interest for the user. Its value is 1. 
The second field is 'pos'. As such, [ -8.01000665242636,  49.9969587092104 , -10.31255250585599] are the coordinates (x,y,z) of particle 1 at the current point in time.
There is also a field called 'force' at index 4, i.e. [ 40.96887479809036   , -27.908936738251054  ,  -0.14310326836228807] is the force vector acting on particle 0.
We can access individual fields via their name or index:


.. code-block:: python
	
	In[3]: Simulation.Particles[0]['force']

	Out[3]: array([ 40.96887479809036   , -27.908936738251054  , -0.14310326836228807])

	In[4]: Simulation.Particles[0][4]

	Out[4]: array([ 40.96887479809036   , -27.908936738251054  , -0.14310326836228807])

Each particle has assigned a unique id by which they can by identified unambiguously:

.. code-block:: python
	
	In[3]: Simulation.Particles[0]['unique_id']

	Out[3]: 10352


**Molecules**

Similarly, the properties of all the molecules (Rigid Bead molecules: RBs) in the simulation can be accessed by


.. code-block:: python
	
	Simulation.RBs[0]


As for the particles dta structure, a numpy structured array is returned. Its field names are

.. code-block:: python
	
	In[1]: Simulation.RBs[0].dtype.names

	Out[1]: ('next',
		'name',
 		'id',
 		'unique_id',
 		'type_id',
 		'pos',
 		'dX',
 		'force',
 		'torque',
 		'topology',
 		'topology_N',
 		'q',
 		'dq',
 		'B',
 		'orientation_quat',
		'mu_tb',
 		'mu_rb',
 		'mu_tb_sqrt',
		'mu_rb_sqrt',
		'Dtrans',
 		'Drot',
		'radius',
 		'loc_id',
		'compartment',
		'triangle_id',
 		'pos_last',
 		'Theta_t',
 		'Theta_r',
 		'posL',
 		'collision_type',
 		'next_transition',
 		'h')


As you can see, the RBs data structure is a bit more complex. We can again access the value of the different properties via the field name.
As such, 'name', e.g., returns the name of the molecule:

.. code-block:: python
	
	In[2]: Simulation.RBs[0]['name']

	Out[2]: 'IgG3'

'torque' returns the torque:

.. code-block:: python
	
	In[3]: Simulation.RBs[0]['torque']

	Out[3]: array([-17.089577893521152 ,   1.0869521370881923,   0.6020275734384415])

and 'q' returns the rotation/orientation quaternion

.. code-block:: python
	
	In[4]: Simulation.RBs[0]['q']

	Out[4]: array([ 0.43285676412556806,  0.07211461325989324, -0.18277302147641378, 0.879788910435623  ])

We can also have a look at the molecule's topology, i.e. the particles that the molecule is composited of using the 'topology field':

.. code-block:: python
	
	In[5]: Simulation.RBs[0]['topology']

	Out[5]: array([25, 26, 27, 28, 29,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0], dtype=int64)

Returned is an array of length 20 (20 is the default maximum number of particle a molecule can consist of). 
However, the actual molecule may consist of less than 20 particles. The field 'topology_N' keeps the total number of particles. 
Therefore, the proper way to get the particle indices of the molecule is:


.. code-block:: python
	
	In[6]: Simulation.RBs[5]['topology'][0:Simulation.RBs[5]['topology_N']]

	Out[6]: array([25, 26, 27, 28, 29], dtype=int64)


Each molecule has assigned a unique id that can be accessed via the 'unique_id' field. This is not to be confused with the 'id' field, which is not an unambigous identifier!

.. code-block:: python
	
	In[3]: Simulation.RBs[0]['unique_id']

	Out[3]: 548

.. warning::
   The value of the 'id' field is not an unambigous identifier of a molecule and may be reassigned to other molecules during simulation. Use the 'unique_id' field to unambigously identify a molecule!

