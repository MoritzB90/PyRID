# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numpy as np
import numba as nb

@nb.njit
def update_particle_level(Particles, particle_types):
    
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
    
    for i0 in range(Particles.occupied.n):
        i = Particles.occupied[i0]  
    
        ptype = Particles[i]['type']
        Particles[i]['h'] = particle_types[str(ptype)][0]['h']
        Particles[i]['cutoff'] = particle_types[str(ptype)][0]['cutoff']




@nb.njit
def update_hgrid(HGrid, box_lengths, N, System):
    
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
    
    eps = 1.0 + 1e-8 # We scale the overall grid size by this factor relatove to the simulation box size. This way, the cell grid is slioghtly larger than the simulation box. If we don't do this, we can get out of bounds errors for points that lie exactly in the plane of a positive box boundary!
    
    N_Levels = len(HGrid[0]['L'])

    hstart = 0
    for i in range(N_Levels):
        
        HGrid[0]['head_start'][i] = hstart
        
        HGrid[0]['cells_per_dim'][i][0] = box_lengths[0]/(2*HGrid[0]['cutoff'][i])
        HGrid[0]['cells_per_dim'][i][1] = box_lengths[1]/(2*HGrid[0]['cutoff'][i])
        HGrid[0]['cells_per_dim'][i][2] = box_lengths[2]/(2*HGrid[0]['cutoff'][i])
        
            
        if np.any(HGrid[0]['cells_per_dim'][i]>System.max_cells_per_dim):
            min_radius = max(System.box_lengths)/System.max_cells_per_dim
            HGrid[0]['cells_per_dim'][i][0] = int(System.box_lengths[0]/min_radius)
            HGrid[0]['cells_per_dim'][i][1] = int(System.box_lengths[1]/min_radius)
            HGrid[0]['cells_per_dim'][i][2] = int(System.box_lengths[2]/min_radius)
        
        HGrid[0]['sh'][i][0] = box_lengths[0]*eps/ HGrid[0]['cells_per_dim'][i][0]
        HGrid[0]['sh'][i][1] = box_lengths[1]*eps/ HGrid[0]['cells_per_dim'][i][1]
        HGrid[0]['sh'][i][2] = box_lengths[2]*eps/ HGrid[0]['cells_per_dim'][i][2]
        
        HGrid[0]['NCells'][i] = HGrid[0]['cells_per_dim'][i].prod()
        
        hstart += HGrid[0]['NCells'][i]
        
    
    
def create_hgrid(Particles, particle_types, box_lengths, N, System):
    
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
    
    eps = 1.0 + 1e-8 # We scale the overall grid size by this factor relatove to the simulation box size. This way, the cell grid is slioghtly larger than the simulation box. If we don't do this, we can get out of bounds errors for points that lie exactly in the plane of a positive box boundary!
    
    Radii = []
    Types = []
    for ptype in particle_types:

        Radii.append(particle_types[ptype][0]['cutoff'])
        Types.append(ptype)
        
    R_T=zip(Radii,Types)
    Radii, Types = zip(*sorted(R_T, reverse=True)) #sort by Radii        
    
    Level = []
    Cells = []
    Cutoffs = []
    current_radius = None
    current_level = -1
    current_cells = None
    N_Levels = 0
    for i, radius in enumerate(Radii):
        
        if radius>0.0:
            cells = np.array(box_lengths/(2*radius), dtype = np.int64)
            if np.any(cells>System.max_cells_per_dim):
                # cells = np.array([50,50,50], dtype = np.int64) 
                min_radius = max(System.box_lengths)/System.max_cells_per_dim
                cells[0] = int(System.box_lengths[0]/min_radius)
                cells[1] = int(System.box_lengths[1]/min_radius)
                cells[2] = int(System.box_lengths[2]/min_radius)
        else:
            if System.current_step == 0:
                print(f'Note: {Types[i]} lacks any bimolecular reactions or interactions!')
            if current_cells is not None:
                cells = current_cells
            else:
                # cells = np.array([50,50,50], dtype = np.int64) 
                cells = np.zeros(3, dtype = np.int64) 
                min_radius = max(System.box_lengths)/System.max_cells_per_dim
                cells[0] = int(System.box_lengths[0]/min_radius)
                cells[1] = int(System.box_lengths[1]/min_radius)
                cells[2] = int(System.box_lengths[2]/min_radius)
                
            
        if np.all(cells == current_cells):
            Level.append(current_level)
            Cells.append(current_cells)
            Cutoffs.append(current_radius)
        else:
   
            current_cells = cells
            current_level += 1
            Level.append(current_level)
            Cells.append(cells)
            Cutoffs.append(radius)
            
            current_radius = radius
            N_Levels += 1
        
    
    NCells = np.zeros(N_Levels, dtype = np.int64)
    cells_per_dim = np.zeros((N_Levels,3), dtype = np.int64)
    Cutoff_HGrid = np.zeros(N_Levels, dtype = np.float64)
    
    for h, cell, ptype, cutoff in zip(Level, Cells, Types, Cutoffs):
        
        particle_types[ptype][0]['h'] = h
        
        NCells[h] = cell.prod()
        cells_per_dim[h][:] = cell
        Cutoff_HGrid[h] = cutoff
    
    item_t_hgrid = np.dtype([('L', np.int64, (N_Levels,)), ('sh', np.float64, (N_Levels,3)), ('head', np.int64, (np.sum(NCells)*2,)), ('head_start', np.int64, (N_Levels,)), ('ls', np.int64, (2*N+1,)), ('NCells', np.int64, (N_Levels,)), ('cells_per_dim', np.int64, (N_Levels,3)), ('cutoff', np.float64, (N_Levels,))],  align=True)
    
    
    HGrid = np.zeros(1, dtype = item_t_hgrid)
    
    hstart = 0
    for i in range(N_Levels):
        
        HGrid[0]['head_start'][i] = hstart
        
        HGrid[0]['L'][i] = i
        
        HGrid[0]['sh'][i][:] = box_lengths*eps/cells_per_dim[i]
        
        HGrid[0]['cutoff'][i] = Cutoff_HGrid[i]
        
        HGrid[0]['NCells'][i] = NCells[i]
        
        HGrid[0]['cells_per_dim'][i][:] = cells_per_dim[i]
        
        hstart += NCells[i]


    update_particle_level(Particles, particle_types)
    
    return HGrid



#%%

# if __name__ == '__main__':

