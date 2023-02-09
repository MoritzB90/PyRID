===============
Dendritic Spine
===============

.. code-block:: python
	
	file_path='Files/'
	fig_path = 'Figures/'
	file_name='Dendritic_Spine'
	    
	nsteps = 1e5
	stride = int(nsteps/1000)
	obs_stride = int(nsteps/100)
	box_lengths = [150.0,150.0,200.0]
	Temp=293.15
	eta=1e-21
	dt = 0.1

	Simulation = prd.Simulation(box_lengths = box_lengths, 
	                            dt = dt, 
	                            Temp = Temp, 
	                            eta = eta, 
	                            stride = stride, 
	                            write_trajectory = True, 
	                            file_path = file_path, 
	                            file_name = file_name, 
	                            fig_path = fig_path, 
	                            boundary_condition = 'fixed concentration', 
	                            wall_force = 100.0,
	                            nsteps = nsteps, 
	                            seed = 0, 
	                            length_unit = 'nanometer', 
	                            time_unit = 'ns')

.. code-block:: python
	
	Simulation.register_particle_type('Core_1', 1.5) # (Name, Radius)
	Simulation.register_particle_type('Patch_1', 0.0)
	Simulation.register_particle_type('Patch_2', 0.0)
	Simulation.register_particle_type('Patch_3', 0.0)
	Simulation.register_particle_type('Patch_4', 0.0)
	Simulation.register_particle_type('Core_2', 2.5)
	Simulation.register_particle_type('Core_3', 2.5)
	Simulation.register_particle_type('Core_4', 0.25)
	Simulation.register_particle_type('Core_5', 1.5)
	Simulation.register_particle_type('Patch_5', 0.0)

	A_pos = [[0.0,0.0,1.5], [0.0,0.0,-1.5], [1.5,0.0,-1.5]]
	A_types = ['Core_1','Core_1','Patch_1']

	B_pos = prd.distribute_surf.evenly_on_sphere(5,2.5)
	B_types = ['Core_2','Patch_2','Patch_2', 'Patch_2', 'Patch_2', 'Patch_2']

	B2_pos = prd.distribute_surf.evenly_on_sphere(5,2.5)
	B2_types = ['Core_3','Patch_3','Patch_3', 'Patch_3', 'Patch_3', 'Patch_3']

	C_pos = [[0.0,0.0,1.5], [0.0,0.0,-1.5], [1.5,0.0,-1.5]]
	C_types = ['Core_5','Core_5','Patch_5']

	D_pos = [[0.0,0.0,0.0]]
	D_types = ['Core_4']


	Simulation.register_molecule_type('A', A_pos, A_types)
	Simulation.register_molecule_type('B', B_pos, B_types)
	Simulation.register_molecule_type('B2', B2_pos, B2_types)
	Simulation.register_molecule_type('C', C_pos, C_types)
	Simulation.register_molecule_type('D', D_pos, D_types, collision_type = 1)

	D_tt, D_rr = prd.diffusion_tensor(Simulation, 'A')
	Simulation.set_diffusion_tensor('A', D_tt, D_rr)

	D_tt, D_rr = prd.diffusion_tensor(Simulation, 'B')
	Simulation.set_diffusion_tensor('B', D_tt, D_rr)

	D_tt, D_rr = prd.diffusion_tensor(Simulation, 'B2')
	Simulation.set_diffusion_tensor('B2', D_tt, D_rr)

	D_tt, D_rr = prd.diffusion_tensor(Simulation, 'C')
	Simulation.set_diffusion_tensor('C', D_tt, D_rr)

	D_tt, D_rr = prd.diffusion_tensor(Simulation, 'D')
	Simulation.set_diffusion_tensor('D', D_tt, D_rr)

	prd.plot.plot_mobility_matrix('A', Simulation)
	prd.plot.plot_mobility_matrix('B', Simulation)



.. code-block:: python
	
	#-----------------------------------------------------
	# Add Global Pair Interactions
	#-----------------------------------------------------

	k=100.0 #kJ/(avogadro*nm^2)

	Simulation.add_interaction('harmonic_repulsion', 'Core_1', 'Core_1', {'k':k}, bond = False)
	Simulation.add_interaction('harmonic_repulsion', 'Core_2', 'Core_2', {'k':k}, bond = False)
	Simulation.add_interaction('harmonic_repulsion', 'Core_3', 'Core_3', {'k':k}, bond = False)
	Simulation.add_interaction('harmonic_repulsion', 'Core_5', 'Core_5', {'k':k}, bond = False)


	# Simulation.add_interaction('harmonic_repulsion', 'Core_1', 'Core_2', {'k':k}, bond = False)
	# Simulation.add_interaction('harmonic_repulsion', 'Core_1', 'Core_3', {'k':k}, bond = False)
	Simulation.add_interaction('harmonic_repulsion', 'Core_1', 'Core_5', {'k':k}, bond = False)

	Simulation.add_interaction('harmonic_repulsion', 'Core_2', 'Core_3', {'k':k}, bond = False)
	# Simulation.add_interaction('harmonic_repulsion', 'Core_2', 'Core_5', {'k':k}, bond = False)

	# Simulation.add_interaction('harmonic_repulsion', 'Core_3', 'Core_5', {'k':k}, bond = False)

	# Simulation.add_interaction('harmonic_attraction', 'Core_3', 'Patch_3', {'k':k, 'h':10.0, 'd1':2.0, 'd2':0.0}, bond = False)
	# Simulation.add_interaction('harmonic_attraction', 'Core_3', 'Patch_1', {'k':k, 'h':10.0, 'd1':2.0, 'd2':0.0}, bond = False)


	#%%

	#-----------------------------------------------------
	# Add Pair Binding Reactions
	#-----------------------------------------------------

	k=100.0
	h=50.0
	d=0.0
	rc = 2.0


	Simulation.add_bp_reaction('bind', ['Patch_1', 'Patch_4'], ['Patch_1', 'Patch_4'], 0.1, 2.0, 'harmonic_attraction', {'k':k, 'h':h/3 , 'rc':rc})

	Simulation.add_bp_reaction('bind', ['Patch_4', 'Patch_2'], ['Patch_4', 'Patch_2'], 10.0, 2.0, 'harmonic_attraction', {'k':k, 'h':h , 'rc':rc})

	Simulation.add_bp_reaction('bind', ['Patch_5', 'Patch_2'], ['Patch_5', 'Patch_2'], 0.1, 2.0, 'harmonic_attraction', {'k':k, 'h':h/3 , 'rc':rc})


	prd.plot.plot_potential(Simulation, [(prd.potentials.harmonic_repulsion, np.array([3.0,k])), (prd.potentials.piecewise_harmonic, np.array([2.0,k,h,d]))], yU_limits = [-60,430], yF_limits = [-60 ,300 ])


	#%%

	Simulation.add_bp_reaction('absorption', ['Patch_3', 'Core_4'], ['Patch_4'], rate=0.1, radius=3.0)

	#%%

	Simulation.add_um_reaction('production', 'C', 0.0005, ['C']+['D']*25, product_loc = [1]+[0]*25, product_direction = [1]+[-1]*25, radius=1.0)


.. code-block:: python
	
	vertices, triangles, Compartments = prd.load_compartments('Compartments/DendriticSpine.obj')

	Simulation.set_compartments(Compartments, triangles, vertices, mesh_scale = 1e3/2)
	    
	prd.plot.plot_compartments(Simulation, save_fig = True)


.. code-block:: python
	
	#-----------------------------------------------------
	# Fixed concentration at boundary
	#-----------------------------------------------------

	Simulation.fixed_concentration_at_boundary('B', 0, 'Postsynapse', 'Volume')

	Simulation.fixed_concentration_at_boundary('A', 100/Simulation.System.Compartments[1].area, 'Postsynapse', 'Surface')


.. code-block:: python
	
	pos, mol_type_idx, quaternion = Simulation.distribute('PDS', 'Volume', 3, ['B', 'B2'], [400,400], 1, 30) # Distribute in compartment 3
	Simulation.add_molecules('Volume',2, pos, quaternion, mol_type_idx) # Add to compartment 2

	pos, mol_type_idx, quaternion, face_ids = Simulation.distribute('PDS', 'Surface', 2, ['A'], [75])
	Simulation.add_molecules('Surface',2, pos, quaternion, mol_type_idx, face_ids)

	pos, mol_type_idx, quaternion, face_ids = Simulation.distribute('PDS', 'Surface', 2, ['A', 'C'], [25,25], facegroup = 'PSD')
	Simulation.add_molecules('Surface',2, pos, quaternion, mol_type_idx, face_ids)

	#%%

	prd.plot.plot_scene(Simulation, save_fig = True)


.. code-block:: python
	
	Simulation.observe('Number', ['A', 'B', 'B2', 'C', 'D'], obs_stride = obs_stride)

	Simulation.observe('Bonds', obs_stride = obs_stride)


.. code-block:: python
	
	Timer = Simulation.run(progress_stride = 1000, out_linebreak = False)

	Simulation.print_timer()


.. code-block:: python
	
	prd.plot.plot_concentration_profile(Simulation, axis = 0, save_fig = True)

	Evaluation = prd.Evaluation()
	Evaluation.load_file(file_name)

	Evaluation.plot_observable('Number', molecules = ['A', 'B', 'B2', 'C', 'D'], save_fig = True)


.. raw:: html
	
	<iframe width="560" height="315" src="https://www.youtube.com/embed/ZhkUvfedBT4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

.. raw:: html
	
	<iframe width="560" height="315" src="https://www.youtube.com/embed/PZNQIHWmXBw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
	

.. figure:: Figures/Dendritic_Spine_Meshes.png
    :width: 80%
    :name: fig:Dendritic_Spine_Meshes
    