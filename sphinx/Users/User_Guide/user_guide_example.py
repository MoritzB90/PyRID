# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 18:32:27 2022

@author: Moritz
"""

import pyrid as prd

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


Simulation.add_particle_type('core_1', 2.5) # (name, radius)
Simulation.add_particle_type('core_2', 1.5)
Simulation.add_particle_type('core_3', 2.0)
Simulation.add_particle_type('patch_1', 0.0)
Simulation.add_particle_type('patch_2', 0.0)


A_pos = prd.distribute_surf.evenly_on_sphere(3,2.5) # (number, radius)
A_types = ['core_1','patch_1','patch_1', 'patch_1']
Simulation.register_molecule_type('A', A_pos, A_types)

B_pos = [[0.0,0.0,1.5], [0.0,0.0,-2.0], [-1.5,0.0,1.5], [2.0,0.0,-2.0]]
B_types = ['core_2','core_3','patch_2','patch_2']
Simulation.register_molecule_type('B', B_pos, B_types)


D_rr = [[0.005,0,0],[0,0.04,0],[0,0,0.1]]
D_tt = [[0.5,0,0],[0,0.4,0],[0,0,0.1]]

Simulation.set_diffusion_tensor('A', D_tt, D_rr)

D_tt, D_tr, D_rt, D_rr, r_CoD, r_CoM = prd.diffusion_tensor(Simulation, 'B', return_CoD = True, return_coupling = True, return_CoM = True)

Simulation.set_diffusion_tensor('B', D_tt, D_rr)


prd.plot.plot_mobility_matrix('A', Simulation, save_fig = True, show = True)
prd.plot.plot_mobility_matrix('B', Simulation, save_fig = True, show = True)

