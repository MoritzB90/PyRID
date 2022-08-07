# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numpy as np
import numba as nb

#%%

@nb.njit
def update_rb_level(RBs, System):
    
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
    
    for i0 in range(RBs.occupied.n):
        i = RBs.occupied[i0]  
    
        mol_type = RBs[i]['name']
        RBs[i]['h'] = System.molecule_types[str(mol_type)].h



@nb.njit
def update_rb_hgrid(HGrid, box_lengths, N, System):
    
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
        
        
    
def create_rb_hgrid(Simulation, RBs, RB_Types, box_lengths, N, System):
    
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
    for mol_type in RB_Types:

        Radii.append(Simulation.System.molecule_types[mol_type].radius)
        Types.append(mol_type)
        
    R_T=zip(Radii,Types)
    Radii, Types = zip(*sorted(R_T, reverse=True)) #sort by Radii        
    
    Level = []
    Cells = []
    Cutoffs = []
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
                cells = np.empty(3, dtype = np.int64) 
                min_radius = max(System.box_lengths)/System.max_cells_per_dim
                cells[0] = int(System.box_lengths[0]/min_radius)
                cells[1] = int(System.box_lengths[1]/min_radius)
                cells[2] = int(System.box_lengths[2]/min_radius)
                
            
   
        current_cells = cells
        current_level += 1
        Level.append(current_level)
        Cells.append(cells)
        Cutoffs.append(radius)
        
        N_Levels += 1
        
    
    NCells = np.empty(N_Levels, dtype = np.int64)
    cells_per_dim = np.empty((N_Levels,3), dtype = np.int64)
    Cutoff_HGrid = np.empty(N_Levels, dtype = np.float64)
    
    for h, cell, ptype, cutoff in zip(Level, Cells, Types, Cutoffs):
        
        Simulation.System.molecule_types[mol_type].h = h
        
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


    update_rb_level(RBs, Simulation.System)
    
    return HGrid




#%%

@nb.njit(fastmath=True)
def radial_distr_function(System, RDF_types, RBs, HGrid, RDF_cutoff, rdf_bins, rdf_hist):
    
    
    """
    
    """
    
    rdf_hist[:] = 0.0
    dr_rdf = RDF_cutoff / float(rdf_bins)

    box_lengths = System.box_lengths
    empty = System.empty
    x_c_min = np.empty(System.dim)
    x_c_max = np.empty(System.dim)
    
    
    pos_shift = np.empty(System.dim)  # position shift to account for periodic boundary conditions.
    
    # Initialize
    HGrid[0]['head'].fill(empty)

    HGrid[0]['ls'].fill(empty)
    

    # Loop over all particles and place them in cells, calculate bond forces and update reaction list.
    for i0 in range(RBs.occupied.n):
        i = RBs.occupied[i0]
        
        typeI_id = RBs[i]['type_id']  
        pos_i = RBs[i]['pos']
        
        
        within_box = True
        if System.boundary_condition_id == 1:
            # Check if particle is within box volume:
            within_box = -box_lengths[0]/2<=pos_i[0]<box_lengths[0]/2 and -box_lengths[1]/2<=pos_i[1]<box_lengths[1]/2 and -box_lengths[2]/2<=pos_i[2]<box_lengths[2]/2
            
        #Only add the particle to the linked cell list if there is actually any pair-interaction or bimol-reaction defined for this partricle type:
        if typeI_id == RDF_types[0] or typeI_id == RDF_types[1]:
        
            h = RBs[i]['h']
            h_start = HGrid[0]['head_start'][h]
            cells_per_dim_prim = HGrid[0]['cells_per_dim'][h]
            
            # Determine what cell, in each direction, the i-th particle is in
            cx = int((pos_i[0]+box_lengths[0]/2) / HGrid[0]['sh'][h][0])
            cy = int((pos_i[1]+box_lengths[1]/2) / HGrid[0]['sh'][h][1])
            cz = int((pos_i[2]+box_lengths[2]/2) / HGrid[0]['sh'][h][2])
            

            if within_box:
                
                c = cx + cy * cells_per_dim_prim[0] + cz * cells_per_dim_prim[0] * cells_per_dim_prim[1]
                # List of particle indices occupying a given cell
                HGrid[0]['ls'][i] = HGrid[0]['head'][h_start+c]
        
                # The last particle found to lie in cell c (head particle)
                HGrid[0]['head'][h_start+c] = i
        
            
#%%
                
    
    # Loop over all Particles
    for i0 in range(RBs.occupied.n):
        i = RBs.occupied[i0] + 1 
 
        pos_i = RBs[i]['pos']
        # typeI_id = RBs_Data[i]['type_id']  
        
        h = RBs[i]['h']
        h_start = HGrid[0]['head_start'][h]
        cells_per_dim_prim = HGrid[0]['cells_per_dim'][h]
        
        
        #%%
        
        cx = int((pos_i[0]+box_lengths[0]/2) / HGrid[0]['sh'][h][0])
        cy = int((pos_i[1]+box_lengths[1]/2) / HGrid[0]['sh'][h][1])
        cz = int((pos_i[2]+box_lengths[2]/2) / HGrid[0]['sh'][h][2])
        
        dcx = int(np.ceil(RDF_cutoff/HGrid[0]['sh'][h][0]))
        dcy = int(np.ceil(RDF_cutoff/HGrid[0]['sh'][h][1]))
        dcz = int(np.ceil(RDF_cutoff/HGrid[0]['sh'][h][2]))
        
        cz_start, cz_end = cz - dcz, cz + dcz + 1
        cy_start, cy_end = cy - dcy, cy + dcy + 1
        cx_start, cx_end = cx - dcx, cx + dcx + 1
        
        
        within_box = True
        if System.boundary_condition_id != 0:
            # Check if particle is within box volume:
            within_box = 0<=cx<cells_per_dim_prim[0] and 0<=cy<cells_per_dim_prim[1] and 0<=cz<cells_per_dim_prim[2]
            
            if cz_start < 0:
                cz_start = 0 
            elif cz_end > cells_per_dim_prim[2]:
                cz_end = cells_per_dim_prim[2]
                
            if cy_start < 0:
                cy_start = 0 
            elif cy_end > cells_per_dim_prim[1]:
                cy_end = cells_per_dim_prim[1]
                
            if cx_start < 0:
                cx_start = 0 
            elif cx_end > cells_per_dim_prim[0]:
                cx_end = cells_per_dim_prim[0]
                
        #%%
        
        if within_box:
            if RDF_types[0] == RDF_types[1]:
                # Check contacts on same hierarchy level:
                for cz_N in range(cz_start, cz_end):
                    cz_shift = 0 + cells_per_dim_prim[2] * (cz_N < 0) - cells_per_dim_prim[2] * (cz_N >= cells_per_dim_prim[2])
                    pos_shift[2] = 0.0 - box_lengths[2] * (cz_N < 0) + box_lengths[2]*(cz_N >= cells_per_dim_prim[2])
        
                    for cy_N in range(cy_start, cy_end):
                        cy_shift = 0 + cells_per_dim_prim[1] * (cy_N < 0) - cells_per_dim_prim[1] * (cy_N >= cells_per_dim_prim[1])
                        pos_shift[1] = 0.0 - box_lengths[1] * (cy_N < 0) + box_lengths[1] * (cy_N >= cells_per_dim_prim[1])
        
                        for cx_N in range(cx_start, cx_end):
                            cx_shift = 0 + cells_per_dim_prim[0] * (cx_N < 0) - cells_per_dim_prim[0] * (cx_N >= cells_per_dim_prim[0])
                            pos_shift[0] = 0.0 - box_lengths[0] * (cx_N < 0) + box_lengths[0] * (cx_N >= cells_per_dim_prim[0])
        
                            # Compute the location of the N-th cell based on shifts
                            c_N = (cx_N + cx_shift) + (cy_N + cy_shift) * cells_per_dim_prim[0] \
                                  + (cz_N + cz_shift) * cells_per_dim_prim[0] * cells_per_dim_prim[1]
                                  
                          
                            # Check neighboring head particle interactions
                            j = HGrid[0]['head'][h_start+c_N]
                    
                            while j != empty:
    
                                if i < j:
                                    
                                    pos_j = RBs[j]['pos']
                                    # typeJ_id = RBs_Data[j]['type_id']  
                                    
                                    # Compute the difference in positions for the i-th and j-th particles
                                    if System.boundary_condition_id == 0:
                                        dx = pos_i[0] - (pos_j[0] + pos_shift[0])
                                        dy = pos_i[1] - (pos_j[1] + pos_shift[1])
                                        dz = pos_i[2] - (pos_j[2] + pos_shift[2])
                                            
                                    else:
                                        dx = pos_i[0] - (pos_j[0])
                                        dy = pos_i[1] - (pos_j[1])
                                        dz = pos_i[2] - (pos_j[2])
        
                                    # Compute distance between particles i and j
                                    r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                                    
                                    if int(r / dr_rdf) < rdf_bins:
                                        rdf_hist[int(r / dr_rdf)] += 2 
                                            
                                # Move down list (ls) of particles for cell interactions with a head particle
                                j = HGrid[0]['ls'][j]
        
            #-----------------------------------------------------------
            else:
    
                hj = h-1
                while hj>=0:
                    
                    cells_per_dim_sec = HGrid[0]['cells_per_dim'][hj]
                    
                    h_start = HGrid[0]['head_start'][hj]
                    
                    x_c_min[0] = pos_i[0] - (RDF_cutoff + HGrid[0]['sh'][hj][0]/2)
                    x_c_min[1] = pos_i[1] - (RDF_cutoff + HGrid[0]['sh'][hj][1]/2)
                    x_c_min[2] = pos_i[2] - (RDF_cutoff + HGrid[0]['sh'][hj][2]/2)
            
                    cx_start = np.floor((x_c_min[0]+box_lengths[0]/2) / HGrid[0]['sh'][hj][0])
                    cy_start = np.floor((x_c_min[1]+box_lengths[1]/2) / HGrid[0]['sh'][hj][1])
                    cz_start = np.floor((x_c_min[2]+box_lengths[2]/2) / HGrid[0]['sh'][hj][2])
                    
                    
                    x_c_max[0] = pos_i[0] + (RDF_cutoff + HGrid[0]['sh'][hj][0]/2)
                    x_c_max[1] = pos_i[1] + (RDF_cutoff + HGrid[0]['sh'][hj][1]/2)
                    x_c_max[2] = pos_i[2] + (RDF_cutoff + HGrid[0]['sh'][hj][2]/2)
            
                    cx_end = np.floor((x_c_max[0]+box_lengths[0]/2) / HGrid[0]['sh'][hj][0])+1
                    cy_end = np.floor((x_c_max[1]+box_lengths[1]/2) / HGrid[0]['sh'][hj][1])+1
                    cz_end = np.floor((x_c_max[2]+box_lengths[2]/2) / HGrid[0]['sh'][hj][2])+1
                    
                    if System.boundary_condition_id != 0:
                        
                        if cz_start < 0:
                            cz_start = 0 
                        elif cz_end > cells_per_dim_sec[2]:
                            cz_end = cells_per_dim_sec[2]
                            
                        if cy_start < 0:
                            cy_start = 0 
                        elif cy_end > cells_per_dim_sec[1]:
                            cy_end = cells_per_dim_sec[1]
                            
                        if cx_start < 0:
                            cx_start = 0 
                        elif cx_end > cells_per_dim_sec[0]:
                            cx_end = cells_per_dim_sec[0]
                            
        
                    for cz_N in range(cz_start, cz_end):
                        cz_shift = 0 + cells_per_dim_sec[2] * (cz_N < 0) - cells_per_dim_sec[2] * (cz_N >= cells_per_dim_sec[2])
                        pos_shift[2] = 0.0 - box_lengths[2] * (cz_N < 0) + box_lengths[2]*(cz_N >= cells_per_dim_sec[2])
                        for cy_N in range(cy_start, cy_end):
                            cy_shift = 0 + cells_per_dim_sec[1] * (cy_N < 0) - cells_per_dim_sec[1] * (cy_N >= cells_per_dim_sec[1])
                            pos_shift[1] = 0.0 - box_lengths[1] * (cy_N < 0) + box_lengths[1]*(cy_N >= cells_per_dim_sec[1])
                            for cx_N in range(cx_start, cx_end):
                                cx_shift = 0 + cells_per_dim_sec[0] * (cx_N < 0) - cells_per_dim_sec[0] * (cx_N >= cells_per_dim_sec[0])
                                pos_shift[0] = 0.0 - box_lengths[0] * (cx_N < 0) + box_lengths[0]*(cx_N >= cells_per_dim_sec[0])
                                
                                c_N = (cx_N + cx_shift) + (cy_N + cy_shift) * cells_per_dim_sec[0] \
                                      + (cz_N + cz_shift) * cells_per_dim_sec[0] * cells_per_dim_sec[1]
                                      
                                
                                j = HGrid[0]['head'][h_start+c_N]
                                
                                while j != empty:
                                    
                                    pos_j = RBs[j]['pos']
                                    # typeJ_id = RBs_Data[j]['type_id']  
                                    
                                    # Compute the difference in positions for the i-th and j-th particles
                                    if System.boundary_condition_id == 0:
                                        dx = pos_i[0] - (pos_j[0] + pos_shift[0])
                                        dy = pos_i[1] - (pos_j[1] + pos_shift[1])
                                        dz = pos_i[2] - (pos_j[2] + pos_shift[2])
                                            
                                    else:
                                        dx = pos_i[0] - (pos_j[0])
                                        dy = pos_i[1] - (pos_j[1])
                                        dz = pos_i[2] - (pos_j[2])
        
                                    # Compute distance between particles i and j
                                    r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                                    
        
                                    if int(r / dr_rdf) < rdf_bins:
                                        rdf_hist[int(r / dr_rdf)] += 1

                                            
                                # Move down list (ls) of particles for cell interactions with a head particle
                                    j = HGrid[0]['ls'][j]
                            
                    hj -= 1




#%%

# if __name__ == '__main__':
    
