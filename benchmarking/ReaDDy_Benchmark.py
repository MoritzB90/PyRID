#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import os
import numpy as np
import time

import readdy
ut = readdy.units


#%%

readdy.__version__


def get_system(with_repulsion, L, particle_radii):
    
    system = readdy.ReactionDiffusionSystem(
        box_size=[L, L, L],
        temperature=293.,
        unit_system={"length_unit": "nanometer", "time_unit": "nanosecond", "energy_unit": "kilojoule / mol"}
    )
    
    system.add_species("A", diffusion_constant=143.1 * ut.micrometer ** 2 / ut.second)
    system.add_species("B", diffusion_constant=71.6 * ut.micrometer ** 2 / ut.second)
    system.add_species("C", diffusion_constant=68.82 * ut.micrometer ** 2 / ut.second)
    
    if with_repulsion:
        force_constant = 10. * ut.kilojoule / ut.mol / (ut.nanometer ** 2)
    else:
        force_constant = 0. * ut.kilojoule / ut.mol / (ut.nanometer ** 2)
    if force_constant.magnitude > 0.:
        all_pairs = [("A","A"), ("B","B"), ("C", "C"), ("A", "B"), ("A", "C"), ("B", "C")]
        for pair in all_pairs:
            distance = particle_radii[pair[0]] + particle_radii[pair[1]]
            system.potentials.add_harmonic_repulsion(pair[0], pair[1], force_constant, 
                                                     interaction_distance=distance)
            
    return system


def configure_reactions(system, reaction_radius):
    
    system.reactions.add_fusion("fusion", "A", "B", "C",
                                rate=1e6 / ut.second,
                                educt_distance=reaction_radius * ut.nanometer)
    system.reactions.add_fission("fission", "C", "A", "B",
                                 rate=5e4 / ut.second,
                                 product_distance=reaction_radius * ut.nanometer)
    
    

def simulate(system, output_file, n, dt, n_steps, observe, multi_threading):
    
    if multi_threading:
        simulation = system.simulation(kernel="CPU")
    else:
        simulation = system.simulation(kernel="SingleCPU")
    
    simulation.output_file = output_file
    simulation.reaction_handler = "Gillespie"
    # simulation.reaction_handler = "UncontrolledApproximation"

    edge_length = system.box_size[0]
    initial_positions_a = np.random.random(size=(int(n/4), 3)) * edge_length - .5 * edge_length
    initial_positions_b = np.random.random(size=(int(n/4), 3)) * edge_length - .5 * edge_length
    initial_positions_c = np.random.random(size=(int(n/2), 3)) * edge_length - .5 * edge_length
    simulation.add_particles("A", initial_positions_a)
    simulation.add_particles("B", initial_positions_b)
    simulation.add_particles("C", initial_positions_c)

    if observe:
        simulation.observe.number_of_particles(stride=100, types=["A", "B", "C"])

    if os.path.exists(simulation.output_file):
        os.remove(simulation.output_file)
        
    simulation.progress_output_stride = 100
    # n_steps = 10000 # int(10000. / dt.magnitude)  # simulate to 10 microseconds
    simulation.run(n_steps=n_steps, timestep=dt)
    
    return simulation
    

#%%
    

numbers = [1000,2000,5000,10000,20000] # [5000] # 
timer = [[],[],[],[],[]]

particle_radii = {"A": 1.5, "B": 3., "C": 3.12}  # in nanometers
dt = 1e-1 * ut.nanosecond
reaction_radius = particle_radii["A"] + particle_radii["B"]

for trials in range(3):
    
    for i,n in enumerate(numbers):
            
        print(n)
        
        n_steps = 3000
        
        rho_tot = 0.003141
        V = n/rho_tot
        L = V**(1/3)
        print(L)
        
        system_with_repulsion = get_system(True, L, particle_radii)
        configure_reactions(system_with_repulsion, reaction_radius)
        
        outfile_with_repulsion = "cytosolic_reactions_with_repulsion.h5"
        
        start = time.perf_counter()
        
        simulation = simulate(system_with_repulsion, outfile_with_repulsion, n, dt, n_steps, observe = False, multi_threading = False)
    
        timer[i].append(time.perf_counter() - start)


#%%

_timer = [[4.4222217189999355, 4.157257560999824, 4.147727719999239],
 [9.349907376999909, 8.296020840999518, 8.292464236999876],
 [23.972112217999893, 23.47658878400034, 23.115324891000455],
 [56.91762770100013, 57.30933346399979, 57.21489187099996],
 [167.44886802099973, 166.4403209840002, 164.73981485000058]]

#%%

import matplotlib.pyplot as plt

its = n_steps/np.mean(timer, axis = 1)
plt.figure(figsize = (4,3), dpi = 150)
plt.plot(numbers, its)
plt.ylabel('it/s')
plt.xlabel('N')

pus = n_steps/np.mean(timer, axis = 1)*np.array(numbers)
plt.figure(figsize = (4,3), dpi = 150)
plt.plot(numbers, pus)
plt.ylabel('pu/s')
plt.xlabel('N')

#%%

plt.figure(figsize = (4,3), dpi = 150)
plt.plot(numbers, 1/pus)
# plt.axhline(0.12e-5)
plt.ylim(0, 0.9e-5)
# plt.xlim(0,20000)
plt.ylabel('s/pu')
plt.xlabel('N')

# perf = simulation._simulation.performance_root()

# print(perf)

#%%

# import matplotlib.pyplot as plt

# traj_with_repulsion = readdy.Trajectory(outfile_with_repulsion)
# times, counts = traj_with_repulsion.read_observable_number_of_particles()

# plt.figure()
# plt.plot(counts[:,0]+counts[:,1]+counts[:,2])

#%%

