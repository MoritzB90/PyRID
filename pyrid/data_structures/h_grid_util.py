# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numpy as np
import numba as nb

@nb.njit
def update_particle_level(Particles, particle_types):
    
    """Updates /sets the hierarchical grid level of all particles in the simulation according to the respective particle type.
    
    Parameters
    ----------
    Particles : `object`
        Instance of the Particles class
    particle_types : `nb.types.DictType`
        Numba dictionary containing the names of all particle types and their properties, such as radius and hierarchical grid level.
    
    
    """
    
    for i0 in range(Particles.occupied.n):
        i = Particles.occupied[i0]  
    
        ptype = Particles[i]['type']
        Particles[i]['h'] = particle_types[str(ptype)][0]['h']
        Particles[i]['cutoff'] = particle_types[str(ptype)][0]['cutoff']




@nb.njit
def update_hgrid(HGrid, box_lengths, N, System):
    
    """Updates the hierarchical grid. This function is only called when the Berendsen barostat is used as in this case the simulation box size is changing. 
    This necessitates an update of certain grid parameters such as cell size and number.
    
    Parameters
    ----------
    HGrid : `array like`
        Numpy structures array that is the hierarchical grid. The structured array contains several fields defined by the np.dtype: 
        `np.dtype([('L', np.int64, (N_Levels,)), ('sh', np.float64, (N_Levels,3)), ('head', np.int64, (np.sum(NCells)*2,)), ('head_start', np.int64, (N_Levels,)), ('ls', np.int64, (2*N+1,)), ('NCells', np.int64, (N_Levels,)), ('cells_per_dim', np.int64, (N_Levels,3)), ('cutoff', np.float64, (N_Levels,))],  align=True)`
    box_lengths : `floa64[3]`
        Length of the simulation box in each dimension.
    N : `int64`
        Total number of particles in the simulation.
    System : `object`
        Instance of the System class.
    
    
    """
    
    eps = 1.0 + 1e-8 # We scale the overall grid size by this factor relative to the simulation box size. This way, the cell grid is slightly larger than the simulation box. If we don't do this, we can get out of bounds errors for points that lie exactly in the plane of a positive box boundary!
    
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
    
    """Creates a hierarchical grid.
    
    Parameters
    ----------
    Particles : `object`
        Instance of the Particles class
    particle_types : `nb.types.DictType`
        Numba dictionary containing the names of all particle types and their properties, such as radius and hierarchical grid level.
    box_lengths : `floa64[3]`
        Length of the simulation box in each dimension.
    N : `int64`
        Total number of particles in the simulation.
    System : `object`
        Instance of the System class.
    
    
    Notes
    -----
    
    *Hierarchical Grid*
    
    The computationally most expensive part in molecular dynamics simulations is usually the calculation of the pairwise interaction forces, beacuase to calculate these, we need to determine the distance between particles. When doing this in the most naiv way, i.e. for each particle i we iterate over all the other particles j in the system and calculate the distance, the computation time will increase quadratically with the number of partciles in the system (:math:`O(N^2)`). Even if we take into account Newtons third law, this will decrease the number of computations (:math:`O(\\frac{1}{2}N(N-1))`), but the computational complexity still is quadratic in N. 
    A straight forward method to significantly improve the situation is the linked cell list approach :cite:p:`Allen2017` (pp.195: 5.3.2 Cell structures and linked lists) where the simulation box is divided into :math:`n \\times n \\times n` cells. The number of cells must be chosen such that the side length of the cells in each dimension :math:`s = L/n`, where :math:`L` is the simulation box length, is greater than the maximum cutoff radius for the pairwise molecular interactions (and in our case also bimolecular reactions). This will decrease the computational complexity to :math:`O(14 N \\rho s^3 )`, where :math:`\\rho` is the molecule density (assuming a mostly homogeneous distribution of molecules). Thereby, the computation time increase rather linear with :math:`N` instead of quadratically (:math:`N^2`).
    
    However, one problem with this method is that it does not efficiently handle polydisperse particle size distributions. This becomes a problem when doing minimal coarse graining of proteins and other structures we find in the cell such as vesicles. As mentioned above, n should be chosen such that the cell length is greater than teh maximum cutoff radius. If we would like to simulate, e.g. proteins (:math:`r \\approx 2-5 nm`) in the presence of synaptic vesicles (:math:`r \\approx 20-30 nm`), the cells become way larger than necessary from the perspective of the small proteins. 
    
    One way to approach this problem would be, to choose a small cell size (e.g. based on the samllest cutoff radius) and just iterate not just over the nearest neighbour cells but over as many cells such that the cutoff radius of the larger proteins/SVs is covered. This approach has has a big disadvantages: Whereas for the smaller partciles we actually reduce the number of unnecessary distance calculations, for the larger particles we don't. We can, however, still take advantage of Newton's 3rd law. For this, we only do the distance calculation if the radius of particle i is larger than the radius of particle j. If the radii are equal, we only do the calculation if index i is smaller than index j.
    
    A much better approach has been introduced by :cite:t:`Ogarko2012` and makes use of a so called hierarchical grid. This approach is the one we use in PyRID. In the hierarchical grid approach, each particle is assigned to a different cell grid depending on its cutoff radius, i.e. the grid consists of different levels or hierarchies, each having a different cell size. This has the downside of taking up more memory, howewer, it drastically reduces the number of distance calculations we need to do for the larger particles and also takes advantage of Newtons third law, enabling polydisiperse system simulations with almost no performance loss. The algorithm for the distance checks works as follows:
        
        1. Iterate over all particles
        2. Assign each particle to a cell on their respective level in the hierarchical grid
        3. Iterate over all particles once more
        4. Do a distance check for the nearest neigbour cells on the level the current particle sites in. This is done using the classical linekd cell list algorithm.
        5. Afterwards, do a cross-level search. For this, distance checks are only done on lower hierarchy levels, i.e. on levels with smaller partcile sizes than the current one. This way, we do not need to double check the same particle pair (3rd law). However, in this step, we will have to iterate over potentialy many empty cells (Note: this should, in principle, also be doable in a more efficient way).
        

    
    Returns
    -------
    array like
        Numpy structures array that is the hierarchical grid. The structured array contains several fields defined by the np.dtype: 
        `np.dtype([('L', np.int64, (N_Levels,)), ('sh', np.float64, (N_Levels,3)), ('head', np.int64, (np.sum(NCells)*2,)), ('head_start', np.int64, (N_Levels,)), ('ls', np.int64, (2*N+1,)), ('NCells', np.int64, (N_Levels,)), ('cells_per_dim', np.int64, (N_Levels,3)), ('cutoff', np.float64, (N_Levels,))],  align=True)`
    
    """
    
    eps = 1.0 + 1e-8 # We scale the overall grid size by this factor relative to the simulation box size. This way, the cell grid is slightly larger than the simulation box. If we don't do this, we can get out of bounds errors for points that lie exactly in the plane of a positive box boundary!
    
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

