# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numpy as np
import glob
import os
from pathlib import Path

from ..molecules import rigidbody_util as rbu
from ..molecules import particles_util as pu

def load_checkpoint(System, directory = 'checkpoints/', file = None):

    """A brief description of what the function (method in case of classes) is and what it’s used for
    
    Parameters
    ----------
    parameter_1 : dtype
        Some Information
    parameter_2 : dtype
        Some Information
    
    Raises
    ------
    NotImplementedError (just an example)
        Brief explanation of why/when this exception is raised
    
    Returns
    -------
    dtype
        Some information
    
    """
    
    if file == None:
        list_of_files = glob.glob(Path(directory) / '*') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getmtime)
        data = np.load(latest_file)
    else:
        data = np.load(Path(directory) / file, allow_pickle=True)
    
    
    print('time step loaded: ', data['step'])
    
    #Particle data:
    P_Data = data['P_Data']
    P_n = data['P_n'] 
    P_occupied_Data = data['P_occupied_Data']
    P_occupied_index = data['P_occupied_index']
    P_occupied_n = data['P_occupied_n']
    
    # System data:
    box_lengths = data['box_lengths']
    # cells_per_dim = data['cells_per_dim']
    HGrid = data['HGrid']
    N = data['N']
    Np = data['Np']
    Nmol = data['Nmol']
    Nbonds = data['Nbonds']
    
    #RBs data:
    RBs_Data = data['RBs_Data'] 
    RBs_n = data['RBs_n'] 
    RBs_occupied_Data = data['RBs_occupied_Data'] 
    RBs_occupied_index = data['RBs_occupied_index'] 
    RBs_occupied_n = data['RBs_occupied_n'] 
    
    # ---------------------------------------
    
    System.box_lengths[:]=box_lengths
    System.volume = System.box_lengths.prod()
    System.N = N
    System.Np = Np
    System.Nmol = Nmol
    System.N_bonds = Nbonds
    
    # ---------------------------------------
    
    RBs = rbu.RBs()
    
    # resize RBs arrays:
    RBs.capacity = len(RBs_Data)
    RBs._resize(RBs.capacity)
    RBs.occupied.capacity = len(RBs_occupied_Data)
    RBs.occupied._resize(RBs.occupied.capacity)
    RBs.occupied.index_capacity = len(RBs_occupied_index)
    RBs.occupied._resize_index(RBs.occupied.index_capacity)
    
    # fill RBs arrays:
    
    RBs.Data[:] = RBs_Data
    RBs.n = RBs_n
    RBs.occupied.Data[:] = RBs_occupied_Data
    RBs.occupied.index[:] = RBs_occupied_index
    RBs.occupied.n = RBs_occupied_n 
            
    #Create Particles array:
    Particles = pu.Particles()
    
    # resize Particles array:
    Particles.capacity = len(P_Data)
    Particles._resize(Particles.capacity)
    Particles.occupied.capacity = len(P_occupied_Data)
    Particles.occupied._resize(Particles.occupied.capacity)
    Particles.occupied.index_capacity = len(P_occupied_index)
    Particles.occupied._resize_index(Particles.occupied.index_capacity)
    
    # fill Particles array
    Particles.Data[:] = P_Data
    Particles.n = P_n
    Particles.occupied.Data[:] = P_occupied_Data
    Particles.occupied.index[:] = P_occupied_index
    Particles.occupied.n = P_occupied_n
    
    
    return RBs, Particles, HGrid

#%%

def save(Simulation, System, RBs, Particles, HGrid, checkpoint_counter):

    """A brief description of what the function (method in case of classes) is and what it’s used for
    
    Parameters
    ----------
    parameter_1 : dtype
        Some Information
    parameter_2 : dtype
        Some Information
    
    Raises
    ------
    NotImplementedError (just an example)
        Brief explanation of why/when this exception is raised
    
    Returns
    -------
    dtype
        Some information
    
    """
    
    np.savez(Path(Simulation.checkpoint[0]['directory']) / (Simulation.file_name + '_' + str(checkpoint_counter)), step = Simulation.current_step, P_Data = Particles.Data, P_n = Particles.n, P_occupied_Data = Particles.occupied.Data, P_occupied_index = Particles.occupied.index, P_occupied_n = Particles.occupied.n, RBs_Data = RBs.Data, RBs_n = RBs.n, RBs_occupied_Data = RBs.occupied.Data, RBs_occupied_index = RBs.occupied.index, RBs_occupied_n = RBs.occupied.n, box_lengths = System.box_lengths, cells_per_dim = System.cells_per_dim, N = System.N, Np = System.Np, Nmol = System.Nmol, Nbonds = System.N_bonds, HGrid = HGrid)
    
    