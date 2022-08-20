# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""


import pyrid as prd


#%%

#-----------------------------------------------------
# Set Parameters
#-----------------------------------------------------


#File name and path
file_path='Files/'
fig_path = 'Figures/'
file_name='Fixed_Concentration'
    

nsteps = 1e5
stride = int(nsteps/1000)
obs_stride = int(nsteps/1000)
box_lengths = [250.0,250.0,350.0]
Temp=293.15
eta=1e-21
dt = 10.0

#%%

#-----------------------------------------------------
# Initialize Simulation
#-----------------------------------------------------

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
                            nsteps = nsteps, 
                            seed = 0, 
                            length_unit = 'nanometer', 
                            time_unit = 'ns')


#%%

#-----------------------------------------------------
# Define Particle Types
#-----------------------------------------------------

Simulation.register_particle_type('Core_0', 2.5)
Simulation.register_particle_type('Core_1', 2.5)
Simulation.register_particle_type('Core_2', 2.5)

#%%

#-----------------------------------------------------
# Import Compartments
#-----------------------------------------------------

vertices, triangles, Compartments = prd.load_compartments('Compartments/Synapse.obj')

Simulation.set_compartments(Compartments, triangles, vertices, mesh_scale = 1e3/2)

#%%

prd.plot.plot_compartments(Simulation, save_fig = True, show = True)

#%%

#-----------------------------------------------------
# Define Molecule Structure
#-----------------------------------------------------


A_pos =[[0.0,0.0,0.0]]
A_types = ['Core_0']

B_pos = [[0.0,0.0,0.0]]
B_types = ['Core_1']

C_pos = [[0.0,0.0,0.0]]
C_types = ['Core_2']

#%%

#-----------------------------------------------------
# Register Molecules
#-----------------------------------------------------

Simulation.register_molecule_type('A', A_pos, A_types, collision_type = 1)
D_tt, D_rr = prd.diffusion_tensor(Simulation, 'A')
Simulation.set_diffusion_tensor('A', D_tt, D_rr)

Simulation.register_molecule_type('B', B_pos, B_types, collision_type = 1)
D_tt, D_rr = prd.diffusion_tensor(Simulation, 'B')
Simulation.set_diffusion_tensor('B', D_tt, D_rr)

Simulation.register_molecule_type('C', C_pos, C_types, collision_type = 1)
D_tt, D_rr = prd.diffusion_tensor(Simulation, 'C')
Simulation.set_diffusion_tensor('C', D_tt, D_rr)


#%%

#-----------------------------------------------------
# Fixed concentration at boundary
#-----------------------------------------------------

Simulation.fixed_concentration_at_boundary('A', 1000/Simulation.System.volume, 'Box', 'Volume')

Simulation.fixed_concentration_at_boundary('B', 1000/Simulation.System.Compartments[1].volume, 'Postsynapse', 'Volume')

Simulation.fixed_concentration_at_boundary('C', 1000/Simulation.System.Compartments[1].area, 'Postsynapse', 'Surface')

#%%          

# #-----------------------------------------------------
# # Distribute Molecules
# #-----------------------------------------------------
    
# points, points_types, quaternion = Simulation.distribute('PDS uniform', 'Volume', 0, ['A'], [100])

# Simulation.add_molecules('Volume',0, points, quaternion, points_types)

# # prd.plot.plot_sphere_packing(0, Simulation, points, points_types, Radius)


# points, points_types, quaternion = Simulation.distribute('PDS uniform', 'Volume', 1, ['B'], [100])

# Simulation.add_molecules('Volume',1, points, quaternion, points_types)


# points, points_types, quaternion, face_ids = Simulation.distribute('PDS', 'Surface', 1, ['C'], [100])

# Simulation.add_molecules('Surface',1, points, quaternion, points_types, face_ids)
    

#%%

# prd.plot.plot_scene(Simulation)

#%%

#-----------------------------------------------------
# Add Observables
#-----------------------------------------------------


Simulation.observe('Number',  molecules = ['A', 'B', 'C'], obs_stride = obs_stride)


#%%

#-----------------------------------------------------
# Start the Simulation
#-----------------------------------------------------

Simulation.run(progress_stride = 1000, out_linebreak = False)

Simulation.print_timer()

#%%

Evaluation = prd.Evaluation()

Evaluation.load_file(file_name)

#%%

import matplotlib.pyplot as plt

Evaluation.plot_observable('Number', molecules = ['A', 'B', 'C'], save_fig = True)

plt.axhline(1000, color = 'k', linestyle = '--', linewidth = 1, zorder = 1)

plt.savefig('Figures//Fixed_Concentration_Number.png', bbox_inches="tight", dpi = 300)

#%%

prd.plot.plot_scene(Simulation, save_fig = True)

