================
Simulation setup
================

First, import pyrid

.. code-block:: python

	import pyrid as prd


The user 'communicates' with PyRID via the Simulation class.
To configure a simulation system in PyRID, some parameters need to be defined by the user.
These include the

#. simulation box size (in nm),
#. integration time step (in ns),
#. temperature (in Kelvin),
#. solvent viscosity (in kJ/(nm^3 ns)),
#. number of simulation steps,

Then, a simulation can be set up by:

.. code-block:: python

	Simulation = prd.Simulation(box_lengths = [50.0,50.0,50.0], 
	                            dt = 0.1, 
	                            Temp = 293.15, 
	                            eta = 1e-21, 
	                            nsteps = 1e5)

By default, energy is given in units of kJ/mol, length in nanometer and time in ns. The user can, however, define different units such as micrometer and s.

.. code-block:: python
	
	Simulation = prd.Simulation(box_lengths = [50.0,50.0,50.0], 
	                            dt = 0.1, 
	                            Temp = 293.15, 
	                            eta = 1e-21, 
	                            nsteps = 1e5,
	                            length_unit = 'nanometer', 
	                            time_unit = 'ns')

In addition, there are many other parameters the user can define. For example the directories for files and figures:

.. code-block:: python
	
	Simulation = prd.Simulation(box_lengths = [50.0,50.0,50.0], 
	                            dt = 0.1, 
	                            Temp = 293.15, 
	                            eta = 1e-21, 
	                            nsteps = 1e5,
	                            length_unit = 'nanometer', 
	                            time_unit = 'ns',
	                            file_path = 'my_directory/',
	                            fig_path = 'Figures/',
	                            file_name='my_first_simulation')

By default, PyRID writes the molecule trajectories into a .xyz file with a stride of 100. You may change these settings:

.. code-block:: python

	Simulation = prd.Simulation(box_lengths = [50.0,50.0,50.0], 
	                            dt = 0.1, 
	                            Temp = 293.15, 
	                            eta = 1e-21, 
	                            nsteps = 1e5,
	                            length_unit = 'nanometer', 
	                            time_unit = 'ns',
	                            file_path = 'my_directory/',
	                            fig_path = 'Figures/',
	                            file_name='my_first_simulation',
	                            write_trajectory = True,
	                            stride = 100)

ALso, the user can set a random seed. Thereby, a simulation will be exactly reproducible:

.. code-block:: python

	Simulation = prd.Simulation(box_lengths = [50.0,50.0,50.0], 
	                            dt = 0.1, 
	                            Temp = 293.15, 
	                            eta = 1e-21, 
	                            nsteps = 1e5,
	                            length_unit = 'nanometer', 
	                            time_unit = 'ns',
	                            file_path = 'my_directory/',
	                            fig_path = 'Figures/',
	                            file_name='my_first_simulation',
	                            write_trajectory = True,
	                            stride = 100,
	                            seed = 0)

PyRID supports three different types of boundary conditions: 'periodic', 'repulsive', 'fixed concentration'. By default, PyRID assumes periodic boundary conditions. If you want to use repulsive boundary conditions, you can set a wall_force constant. By default wall_force = 100. The same constant is also used to resolve collisions of molecules with mesh compartment walls.

.. code-block:: python

	Simulation = prd.Simulation(box_lengths = [50.0,50.0,50.0], 
	                            dt = 0.1, 
	                            Temp = 293.15, 
	                            eta = 1e-21, 
	                            nsteps = 1e5,
	                            length_unit = 'nanometer', 
	                            time_unit = 'ns',
	                            file_path = 'my_directory/',
	                            fig_path = 'Figures/',
	                            file_name='my_first_simulation',
	                            write_trajectory = True,
	                            stride = 100,
	                            seed = 0,
	                            boundary_condition = 'repulsive',
	                            wall_force = 100.0)


Sometimes you may don't want to specify the number of time steps but rather limit the total simulation time. In this case, you can pass sim_time (given in seconds, e.g. 7200 s for a 2 hour simulation) instead of nsteps:

.. code-block:: python

	Simulation = prd.Simulation(box_lengths = [50.0,50.0,50.0], 
	                            dt = 0.1, 
	                            Temp = 293.15, 
	                            eta = 1e-21, 
	                            sim_time = 7200)

Once an instance of the Simulation class is created we may continue configuring our simulation.
For long simulations, it is often convenient to add checkpoints, i.e. PyRID saves the state of the system at certain points. checkpoints are saved in numpy .npz files. You can add checkpoints by:

.. code-block:: python

	Simulation.add_checkpoints(1000, "checkpoints/", 10)

Here, 1000 sets the stride, i.e. PyRID will save the System state each 100th step, "checkpoints/" is the directory, and 10 sets the maximum number of saved files.
If you want to load a checkpoint, you can do so by

.. code-block:: python

	Simulation.load_checkpoint('my_first_simulation', 0)

, where 'my_first_simulation' is the file name and 0 the index of the saved file. In our example there are 10 saved files with indices 0-9. Note that PyRID currently does not save the complete system configuration. That mean, all the molecule types, particles, interactions and reactions need to be declared again, before you can load a checkpoint file. For future releases it would, however, be nice to be able to save complete project files which can then be loaded.
In the next chapter you learn how to add molecules to the system.

Berendsen barostat
------------------

PyRID also supports NPT ensemble simulations using a Berendsen barostat. A barostat can be setup as follows:

.. code-block:: python

    P0 = 0.0 # target pressure
    Tau_P= 1.0 # time constant
    start = 10000 # time step at which to start the barostat
    
    Simulation.add_barostat_berendsen(Tau_P, P0, start)