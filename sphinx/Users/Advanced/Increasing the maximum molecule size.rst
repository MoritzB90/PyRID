====================================
Increasing the maximum molecule size
====================================

.. role:: python(code)
   :language: python


Currently, by default, the maximum allowed size for rigid bead molecules in PyRID is set to 20 particles/beads.
To change this number, you need to edit the rigidbody_util.py file (:func:`pyrid.molecules.rigidbody_util.RBs`).
In there you find the data type for a structured array named 'item_t_RB'. This data type contains a field named 'topology'.
By default, the data type of this field is :python:`np.int64(20,)`. 
Therefore, to increase the maximum number of particles a molecule can consist of in PyRID, you need to change the number 20 to a number of your choice.

Example where the maximum size has been increased to 50 particles per molecule:


.. code-block:: python
	
	item_t_RB = np.dtype([	('next', np.int64),
				('name', 'U20'),
				('id', np.int64),
				('type_id', np.int64), 
				('pos', np.float64, (3,)), 
				('dX', np.float64, (3,)), 
				('force', np.float64, (3,)), 
				('torque', np.float64, (3,)), 
				('topology', np.int64, (50,)),
				('topology_N', np.int64),
				('q', np.float64, (4,)), 
				('dq', np.float64, (4,)),
				('B', np.float64, (4,4)),
				('orientation_quat', np.float64, (3,3)),
				('mu_tb', np.float64, (3,3)),
				('mu_rb', np.float64, (3,3)),
				('mu_tb_sqrt', np.float64, (3,3)),
				('mu_rb_sqrt', np.float64, (3,3)),
				('Dtrans', np.float64),('Drot', np.float64),
				('radius', np.float64), 
				('loc_id', np.int64), 
				('compartment', np.int64), 
				('triangle_id', np.int64), 
				('pos_last', np.float64, (3,)), 
				('Theta_t', np.float64, (3,)), 
				('Theta_r', np.float64, (3,)), 
				('posL', np.float64, (3,)), 
				('collision_type', np.int64), 
				('next_transition', np.float64), 
				('h', np.int64), ],  align=True)

