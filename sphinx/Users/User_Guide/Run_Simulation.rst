.. _userguide_run_simulation:

==============
Run simulation
==============

To run a simulation call:

.. code-block:: python
   
   Simulation.run(progress_stride = 1000, out_linebreak = True)

progress_stride specifies how often the progress bar is updated. out_linebreak determines whether after each progress bar update there is a line break or not. By default out_linebreak is False, such that in your console the impression of an animated progress bar is created. However, if you want to write all messages to a file (sys.stdout = open('output/my_output_file.out', 'w')), it is often more convenient to have line breaks.
Also, by default, the progress bar only contains information about the progress in percent, the iterations per second (it/s) and the number of particle updates per second (pu/s). You can, however, specify more properties to be printed next to the progress bar:

.. code-block:: python
   
   Simulation.run(progress_stride = 1000, out_linebreak = True, progress_bar_properties = ['Time passed', 'step', 'N', 'Pressure', 'Volume', 'Vol_Frac', 'Bonds'])

PyRID automatically times how long the various sub-processes in the simulation loop take. At the end of a simulation, you can print these by calling

.. code-block:: python

   Simulation.print_timer()

.. code-block:: python

   Out:

   force:  72.39429442939345  it/s |  13.81324326567345  ms/it
   integrate:  38.71736020363028  it/s |  25.828207159284492  ms/it
   reactions:  135.0618166766438  it/s |  7.404017098290147  ms/it
   observables:  486.5524890159115  it/s |  2.055276712328765  ms/it
   barostat:  83573.58336609467  it/s |  0.011965503448852886  ms/it
   write:  696.5263566868742  it/s |  1.4356958504149662  ms/it