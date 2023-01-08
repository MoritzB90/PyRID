# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""


import pyrid as prd
import numpy as np

for _ in range(5):
        
    #%%
    
    #-----------------------------------------------------
    # Set Parameters
    #-----------------------------------------------------
    
    file_path='Files/'
    fig_path = 'Figures/'
    file_name='Benchmarking'
        
    nsteps = 1e4
    stride = 1e4
    obs_stride = 1e4
    box_lengths = np.array([75.0,75.0,75.0])
    Temp=293.15
    eta=1e-21
    dt = 0.1
    
    #%%
    
    #-----------------------------------------------------
    # Initialize System
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
                                boundary_condition = 'periodic', 
                                nsteps = nsteps, 
                                seed = 0, 
                                length_unit = 'nanometer', 
                                time_unit = 'ns')
    
    
    #%%
    
    #-----------------------------------------------------
    # Define Particle Types
    #-----------------------------------------------------
    
    Simulation.register_particle_type('Core_1', 2.5) # (Name, Radius)
    Simulation.register_particle_type('Core_2', 10.0) # (Name, Radius)
    
    #%%
    
    #-----------------------------------------------------
    # Add Global Pair Interactions
    #-----------------------------------------------------
    
    k=100.0 #kJ/(avogadro*nm^2) 
    
    
    Simulation.add_interaction('harmonic_repulsion', 'Core_1', 'Core_1', {'k':k}, bond = False)
    Simulation.add_interaction('harmonic_repulsion', 'Core_1', 'Core_2', {'k':k}, bond = False)
    Simulation.add_interaction('harmonic_repulsion', 'Core_2', 'Core_2', {'k':k}, bond = False)
    
    #%%
    
    A_pos = [[0.0,0.0,0.0]]
    A_types = ['Core_1']
    
    B_pos = [[0.0,0.0,0.0]]
    B_types = ['Core_2']
    
    #%%
    
    #-----------------------------------------------------
    # Register Molecules
    #-----------------------------------------------------
    
    Simulation.register_molecule_type('A', A_pos, A_types)
    D_tt, D_rr = prd.diffusion_tensor(Simulation, 'A')
    Simulation.set_diffusion_tensor('A', D_tt, D_rr)
    
    Simulation.register_molecule_type('B', B_pos, B_types)
    D_tt, D_rr = prd.diffusion_tensor(Simulation, 'B')
    Simulation.set_diffusion_tensor('B', D_tt, D_rr)
    
    #%%
    
    # prd.plot.plot_mobility_matrix('A', Simulation, save_fig = False, show = True)
    
    
    #%%          
    
    #-----------------------------------------------------
    # Distribute Molecules
    #-----------------------------------------------------
        
    # points, points_types, quaternion = Simulation.distribute('PDS', 'Volume', 0, ['A', 'B'], [1500,125], clustering_factor=1.0, max_trials=300)
    
    points, points_types, quaternion = Simulation.distribute('PDS', 'Volume', 0, ['A'], [3332], clustering_factor=1.0, max_trials=300)
    
    Simulation.add_molecules('Volume',0, points, quaternion, points_types)
    
    # prd.plot.plot_scene(Simulation, save_fig = True)
    
    #%%
    
    # Change molecule A cutoff radius for comparison of HGrid vs regular linked celllist:
    
    # Simulation.System.particle_types['Core_1'][0]['cutoff'] = Simulation.System.particle_types['Core_2'][0]['cutoff']
    
    #%%
    
    #-----------------------------------------------------
    # Add Observables
    #-----------------------------------------------------
    
    
    # Simulation.observe_rdf(rdf_pairs = [['A','A'],['A','B'],['B','B']], rdf_bins = [100,100,100], rdf_cutoff = [20.0,22.5,25.0], stride = obs_stride)
    
    
    # Simulation.observe('Orientation', molecules = ['A', 'B'], obs_stride = obs_stride)
    
    # Simulation.observe('Position', molecules = ['A', 'B'], obs_stride = obs_stride)
    
    
    #%%
    
    #-----------------------------------------------------
    # Start the Simulation
    #-----------------------------------------------------
        
    Simulation.run(progress_stride = 1000, out_linebreak = False)
    
    Simulation.print_timer()
    
    
    #%%
    
    # Evaluation = prd.Evaluation()
    # Evaluation.load_file(file_name)
    
    # Evaluation.plot_rdf([['A','A']], steps = range(5,10), average = True, save_fig = True)
    
    # Evaluation.plot_rdf([['A','B']], steps = range(5,10), average = True, save_fig = True)
    
    # Evaluation.plot_rdf([['B','B']], steps = range(5,10), average = True, save_fig = True)
    
    #%%

Vol = Simulation.System.volume
Vmol_A = 4/3*np.pi*2.5**3
Vmol_B = 4/3*np.pi*10.0**3
N_A = Simulation.System.Nmol[0]
N_B = Simulation.System.Nmol[1]

packing = (Vmol_A*N_A+Vmol_B*N_B)/Vol

#%%

import numpy as np
import pandas as pd

benchmark_hgrid = {}

benchmark_hgrid['it/s'] = {}
benchmark_hgrid['it/s']['cell list'] = np.array([76.129, 75.824, 76.268, 75.379, 74.677])
benchmark_hgrid['it/s']['hierarchical grid'] = np.array([465.402, 458.263, 462.924, 457.895, 462.626])
benchmark_hgrid['it/s']['hierarchical grid \n single component'] = np.array([263.081, 262.68, 264.307, 263.025, 260.499])

benchmark_hgrid['pu/s'] = {}
benchmark_hgrid['pu/s']['cell list'] = benchmark_hgrid['it/s']['cell list']*1129
benchmark_hgrid['pu/s']['hierarchical grid'] = benchmark_hgrid['it/s']['hierarchical grid']*1129
benchmark_hgrid['pu/s']['hierarchical grid \n single component'] = benchmark_hgrid['it/s']['hierarchical grid \n single component']*2169

#%%

df = pd.DataFrame(benchmark_hgrid['it/s'])

dfm_its = pd.melt(df.reset_index(), id_vars='index', value_vars=['cell list',  'hierarchical grid', 'hierarchical grid \n single component'], var_name = 'method', value_name = 'it/s')

df = pd.DataFrame(benchmark_hgrid['pu/s'])

dfm_pus = pd.melt(df.reset_index(), id_vars='index', value_vars=['cell list',  'hierarchical grid', 'hierarchical grid \n single component'], var_name = 'method', value_name = 'pu/s')

#%%

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

# sns.barplot(x="method", y="it/s", data=dfm_its, ci = 'sd')

plt.figure(figsize = (5,3), dpi = 150)
splt = sns.barplot(x="method", y="pu/s", data=dfm_pus, ci = 'sd')

number = [1129,1129,2169]
packing = [0.52, 0.52, 0.34]
for i,p in enumerate(splt.patches):
    splt.annotate('$N = {0}$ \n $\eta = {1}$'.format(number[i], packing[i]),#format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()+35000), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
    
plt.ylim(0,740000)
plt.xlabel(None)

splt.get_figure().savefig('Figures/benchmark_hgrid.png', dpi = 300, bbox_inches = 'tight')

