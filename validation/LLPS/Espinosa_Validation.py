# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 18:22:41 2022

@author: Moritz
"""



import numpy as np
import pyrid as prd

# sys.stdout = open('output/'+fileName+'.out', 'w')

#%%

#-----------------------------------------------------
# Set Parameters
#-----------------------------------------------------

exp =  '5_sites' #'3_sites'# '4_sites' # str(sys.argv[1]) #
phase = 'Equil' # str(sys.argv[2]) # DC # 
ia_id = -1 # int(sys.argv[3]) # 

print('exp: ', exp)
print('phase: ', phase)
print('ia_id: ', ia_id)


# File name and path
if exp == '3_sites':  
    interaction_strength = np.round(np.linspace(43.5,70,11)/3,1)
    file_path='3_sites/Files/'
    fig_path = '3_sites/Figures/'
    if phase == 'DC':
        file_name = '3_sites_'+phase+'_'+str(interaction_strength[ia_id])
    elif phase == 'Equil':
        file_name= '3_sites_'+phase
elif exp == '4_sites': 
    interaction_strength = np.round(np.linspace(48,80,11)/4,1)
    file_path='4_sites/Files/'
    fig_path = '4_sites/Figures/'
    if phase == 'DC':
        file_name = '4_sites_'+phase+'_'+str(interaction_strength[ia_id])
    elif phase == 'Equil':
        file_name = '4_sites_'+phase
elif exp == '5_sites':
    interaction_strength = np.round(np.linspace(52.5,80,11)/5,1)
    file_path='5_sites/Files/'
    fig_path = '5_sites/Figures/'
    if phase == 'DC':
        file_name = '5_sites_'+phase+'_'+str(interaction_strength[ia_id])
    elif phase == 'Equil':
        file_name = '5_sites_'+phase

print('interaction_strength: ', interaction_strength[ia_id])


#Simulation properties
if phase == 'DC':
    nsteps = 1e9 + 1 #
    sim_time = 47.5*60*60
    use_checkpoint = True
elif phase == 'Equil':
    nsteps = 1600000 
    sim_time = None
    use_checkpoint = False # True #

stride = int(nsteps/100)
obs_stride = int(nsteps/100)

#%%

#System physical properties

if exp == '5_sites':
    box_lengths = np.array([74.0,74.0,74.0])
elif exp == '3_sites':  
    box_lengths = np.array([83.0,83.0,83.0])
elif exp == '4_sites': 
    box_lengths = np.array([78.3,78.3,78.3])

Temp=179.71
eta=1e-21
dt = 0.0025


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
                            time_unit = 'ns'
                            )

#%%

#-----------------------------------------------------
# Add Barostat
#-----------------------------------------------------

if phase == 'Equil':
    
    P0 = 0.0
    Tau_P=1.0
    start = 266667
    
    Simulation.add_barostat_berendsen(Tau_P, P0, start)


#%%

#-----------------------------------------------------
# Add Checkpoints
#-----------------------------------------------------

if exp == '5_sites':
    Simulation.add_checkpoints(1000, "5_sites/checkpoints/", 1) # stride, directory, max_saves
elif exp == '3_sites':  
    Simulation.add_checkpoints(1000, "3_sites/checkpoints/", 1) # stride, directory, max_saves
elif exp == '4_sites': 
    Simulation.add_checkpoints(1000, "4_sites/checkpoints/", 1) # stride, directory, max_saves

#%%

#-----------------------------------------------------
# Define Particle Types
#-----------------------------------------------------

Simulation.register_particle_type('Core_1', 2.5) # (Name, Radius)
Simulation.register_particle_type('Patch_1', 0.0)

#%%

#-----------------------------------------------------
# Add Global repulsive Pair Interactions
#-----------------------------------------------------

lr = 50
la = 49
EpsR = Simulation.System.kbt/1.5

Simulation.add_interaction('PHS', 'Core_1', 'Core_1', {'EpsR':EpsR, 'lr':lr, 'la':la})

#%%

#-----------------------------------------------------
# Add Pair Binding Reaction
#-----------------------------------------------------

sigma = 2.5*2
alpha =  sigma*0.01 # 0.005 #
rw = 0.12*sigma


if exp == '5_sites':
    eps_csw = interaction_strength[ia_id]
elif exp == '3_sites': 
    eps_csw = interaction_strength[ia_id]
elif exp == '4_sites': 
    eps_csw = interaction_strength[ia_id]


Simulation.add_bp_reaction('bind', ['Patch_1', 'Patch_1'], ['Patch_1', 'Patch_1'], 100/dt, 1.75*rw, 'CSW', {'rw':rw, 'eps_csw':eps_csw, 'alpha':alpha})


#%%

prd.plot.plot_potential(Simulation, [(prd.potentials.CSW, np.array([rw, eps_csw, alpha])), (prd.potentials.PHS, np.array([sigma, EpsR, lr, la]))], yU_limits = [-20,40], yF_limits = [-2e2,4e2], r_limits = [0,6], save_fig = True)

#%%

import matplotlib.pyplot as plt

for molecule_name in Simulation.System.molecule_types:
    plt.plot([0,Simulation.System.molecule_types[molecule_name].r_mean_pair],[0,0])
    print(Simulation.System.molecule_types[molecule_name].r_mean_pair)



#%%

#-----------------------------------------------------
# Define Molecule Structure
#-----------------------------------------------------


if exp == '5_sites':
    A_pos = prd.distribute_surf.evenly_on_sphere(5,2.5)
    A_types = np.array(['Core_1','Patch_1','Patch_1', 'Patch_1', 'Patch_1', 'Patch_1'], dtype = np.dtype('U20'))
    
elif exp == '3_sites': 
    A_pos = prd.distribute_surf.evenly_on_sphere(3,2.5)
    A_types = np.array(['Core_1','Patch_1','Patch_1', 'Patch_1'], dtype = np.dtype('U20'))

elif exp == '4_sites': 
    A_pos = prd.distribute_surf.evenly_on_sphere(4,2.5)
    A_types = np.array(['Core_1','Patch_1','Patch_1', 'Patch_1', 'Patch_1'], dtype = np.dtype('U20'))


#%%

#-----------------------------------------------------
# Register Molecules
#-----------------------------------------------------


Simulation.register_molecule_type('A', A_pos, A_types)

D_tt, D_rr = prd.diffusion_tensor(Simulation, 'A')
Simulation.set_diffusion_tensor('A', D_tt, D_rr)

prd.plot.plot_mobility_matrix('A', Simulation, save_fig = True, show = True)

#%%          

#-----------------------------------------------------
# Distribute Molecules
#-----------------------------------------------------

    
if use_checkpoint == False:
    
    pos, mol_type_idx, quaternion = Simulation.distribute('PDS', 'Volume', 0, ['A'], [2000], clustering_factor=2, max_trials=1000)
           
    Simulation.add_molecules('Volume',0, pos, quaternion, mol_type_idx)
    
    prd.plot.plot_scene(Simulation, save_fig = False) 
    
else:

    if exp == '5_sites':
        Simulation.load_checkpoint('5_sites_Equil', 0, directory = '5_sites/checkpoints/')

    elif exp == '3_sites':  
        Simulation.load_checkpoint('3_sites_Equil', 0, directory = '3_sites/checkpoints/')
    elif exp == '4_sites': 
        Simulation.load_checkpoint('4_sites_Equil', 0, directory = '4_sites/checkpoints/')

    Simulation.System.box_lengths[0] *= 3
    Simulation.System.volume = Simulation.System.box_lengths.prod()
    

#%%

#-----------------------------------------------------
# Add Observables
#-----------------------------------------------------

Simulation.observe('Energy', obs_stride = obs_stride)

Simulation.observe('Volume', obs_stride = obs_stride)

Simulation.observe('Pressure', molecules = ['A'], obs_stride = obs_stride)

#%%

#-----------------------------------------------------
# Start the Simulation
#-----------------------------------------------------

Simulation.run(progress_stride = 1000, progress_bar_properties = ['Pressure', 'Volume', 'Vol_Frac'], out_linebreak = True)

Simulation.print_timer()

#%%

Evaluation = prd.Evaluation()

Evaluation.load_file(file_name, path = file_path+'/hdf5/')

Evaluation.plot_observable('Energy', save_fig = True)
Evaluation.plot_observable('Pressure', save_fig = True)
Evaluation.plot_observable('Volume', save_fig = True)

