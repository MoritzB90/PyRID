# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numba as nb
import numpy as np
# import math

# from ..rand_util import random_util as randu
from ..system import potentials_util as potu
from ..geometry.mesh_util import point_triangle_distance
from ..reactions.update_reactions import convert_particle_type

#%%

@nb.njit()
def react_interact_test(comp_i, loc_type_i, mol_idx_j, mol_idx_i, RBs, System):
    
    """Tests if two particles are valid interaction or reaction partners.
    
    Parameters
    ----------
    comp_i : `int64`
        Compartment index of molecule i
    loc_type_i : `int64`
        Location type of moelcule i (volume : 0, surface : 1)
    mol_idx_j : `int64`
        Index of moelcule j
    mol_idx_i : `int64`
        Index of moelcule i
    RBs : `object`
        Instance of RBs class
    System : `object`
        Instance of System class
        
    Notes
    -----
    We need to consider several cases:
        
    #. If both particles are in the same compartment
        #. and at least one is a volume molecule, interactions and reactions are always allowed.
        #. If both particles are surface molecules, interactions are always valid but reactions are only allowed if the angle between the molecules' triangle normals is less than 90 degree.
    #. If both particles are in different compartments
        #. and both particles are volume molecules, neither interactions nor reactions are allowed.
        #. If both particles are surface particles, interactions are allowed, however, not reactions except binding reactions!
        #. If one of the particles belongs to a surface molecule and the other to a volume molecule
            #. reactions are always allowed if one of the compartments is the simulation box (System), otherwise neither reactions, nor interactions are allowed.
        
    We neglect reactions for surface molecules at an angle above 90 degree because the error introduced by using euclidian distances becomes large. In general, you should not use meshes with high local curvature (relative to the molecule size and reaction readius) and/or large angles between neighboring triangle for this reason. Also, triangles that are far apart in terms of the geodesic distance can come close to each other in terms of euclidian distance. In the latter case, the angle between the triangle is, however, in most cases larger than 90 degree. Thereby, neglecting reactions also prevents this type of erroneous reactions.
    
    """
    
    # comp_i = RBs[mol_idx_i]['compartment']
    comp_j = RBs[mol_idx_j]['compartment']
    
    if comp_i == comp_j: # if both particles are in the same compartment, interactions and reactions are always allowed (between volume molecules but also volume and surface moelcules)
    
        if loc_type_i == 0:
            return True, True
        
        loc_type_j = RBs[mol_idx_j]['loc_id']
        if loc_type_j == 0:
            return True, True
        
        #If both moelcules are surface molecules, we may want to check the angle between their triangles to see whether we want to allow reactions:
        tri_id_I = RBs[mol_idx_i]['triangle_id']
        tri_id_J = RBs[mol_idx_j]['triangle_id']
        normal_i = System.Mesh[tri_id_I]['triangle_coord'][3]
        normal_j = System.Mesh[tri_id_J]['triangle_coord'][3]
        cos_phi = normal_i[0]*normal_j[0]+normal_i[1]*normal_j[1]+normal_i[2]*normal_j[2]
        if cos_phi > 0: # We allow reactions if the angle is smaller that 90 degree
            return True, True
        else:
            return True, False
        
    else: 
        # loc_type_i = RBs[mol_idx_i]['loc_id']
        loc_type_j = RBs[mol_idx_j]['loc_id']
        if loc_type_i == loc_type_j:
            if loc_type_i == 0:
                # If both molecules are volume molecules, neither interactions nor reactions are allowed:
                return False, False
            elif loc_type_i == 1:
                # In the other case, i.e. if both are surface molecules, interactions are allowed, however, not reactions except binding reactions!
                return True, False
        else:
            # In the case where both molecules are from different compartments and different locations (surface vs volume), neither reactions, nor interactions are allowed, except the compartment of one molecule is System (comp_id = 0):
            if comp_i == 0 or comp_j == 0:
                return True, True
            else:
                return False, False
            
        

#%%

@nb.njit(fastmath=True)
def update_force_append_reactions(Particles, System, RBs, HGrid):
    

    
    """
    Updates the pair interaction forces between the particle pairs and between particles and the compartemnt meshes (and the simulation box boundary in case of repulsive boundary conditions).
    Also adds the bimolecular and unimolecular particle reactions to the reactions list.

    Parameters
    ----------
    System : `obj`
        Instance of System class.
    Particles : `obj`
        Instance of Particle class.
    RBs : `obj`
        Instance of RBs (Rigid Bodies) class.
    HGrid : `array_like`
        Hierarchical grid that divides the simulation box into cells. The cell size decreases with the level of the hierarchical grid. Particles are assigned to a level depending on their interaction cutoff radius.
        
        `dtype = np.dtype([('L', np.int64, (N_Levels,)), ('sh', np.float64, (N_Levels,3)), ('head', np.int64, (np.sum(NCells)*2,)), ('head_start', np.int64, (N_Levels,)), ('ls', np.int64, (2*N+1,)), ('NCells', np.int64, (N_Levels,)), ('cells_per_dim', np.int64, (N_Levels,3)), ('cutoff', np.float64, (N_Levels,))],  align=True)`
    

    Notes
    -----
    
    *Hierarchical Grid*
    
    The computationaly most expensive part in molecular dynamics simulations is usually the calculation of the pairwise interaction forces, beacuase to calculate these, we need to determine the distance between particles. When doing this in the most naiv way, i.e. for each particle i we iterate over all the other particles j in the system and calculate the distance, the computation time will increase quadratically with the number of partciles in the system (:math:`O(N^2)`). Even if we take into account Newtons third law, this will decrease the number of computations (:math:`O(\\frac{1}{2}N(N-1))`), but the computational complexity still is quadratic in N. 
    A straight forward method to significantly improve the situation is the linked cell list approach :cite:p:`Allen2017` (pp.195: 5.3.2 Cell structures and linked lists) where the simulation box is divided into :math:`n \\times n \\times n` cells. The number of cells must be chosen such that the side length of the cells in each dimension :math:`s = L/n`, where :math:`L` is the simulation box length, is greater than the maximum cutoff radius for the pairwise molecular interactions (and in our case also bimolecular reactions). This will decrease the computational complexity to :math:`O(14 N \\rho s^3 )`, where :math:`\\rho` is the molecule density (assuming a mostly homogeneous distribution of molecules). Thereby, the computation time increase rather linear with :math:`N` instead of quadratically (:math:`N^2`).
    
    However, one problem with this method is that it does not efficiently handle polydisperse particle size distributions. This becomes a problem when doing minimal coarse graining of proteins and other structures we find in the cell such as vesicles. As mentioned above, n should be chosen such that the cell length is greater than teh maximum cutoff radius. If we would like to simulate, e.g. proteins (:math:`r \\approx 2-5 nm`) in the presence of synaptic vesicles (:math:`r \\approx 20-30 nm`), the cells become way larger than necessary from the perspective of the small proteins. 
    
    One way to approach this problem would be, to choose a small cell size (e.g. based on the samllest cutoff radius) and just iterate not just over the nearest neighbour cells but over as many cells such that the cutoff radius of the larger proteins/SVs is covered. This approach has has a big disadvantages: Whereas for the smaller partciles we actually reduce the number of unnecessary distance calculations, for the larger particles we don't. We can, however, still take advantage of Newton's 3rd law. For this, we only do the distance calculation if the radius of particle i is larger than the radius of particle j. If the radii are equal, we only do the calculation if index i is smaller than index j.
    
    A much better approach has been introduced by :cite:t:`Ogarko2012` and makes use of a so called hierarchical grid. This approach is the one we use in PyRID. In the hierarchical grid approach, each particle is assigned to a different cell grid depending on its cutoff radius, i.e. the grid consists of different levels or hierarchies, each having a different cell size. This has the downside of taking up more memory, howewer, it drastically reduces the number of distance calculations we need to do for the larger particles and also takes advantage of Newtons third law, enabling polydisiperse system simulations with almost no performance loss. The algorithm for the distance checks works as follows:
        
        1. Iterate over all particles
        2. Assign each particle to a cell on their respective level in the hierarchical grid
        3. Iterate over all particles once more
        4. Do a distance check for the nearest neigbour cells on the level the current particle sites in. This is done using the classical linekd cell list algorithm.
        5. Afterwards, do a cross-level search. For this, distance checks are only done on lower hierarchy levels, i.e. on levels with smaller partcile sizes than the current one. This way, we do not need to double check the same particle pair (3rd law). However, in this step, we will have to iterate over potentialy many empty cells (Note: this should, in principle, also be doable in a more efficient way).
        
        
    *Mesh collision response*
    
    The distance between particle and mesh triangles is calculated and based on the distance a repulsive force is determined. 
    The repulsive force is normalized by the total number of triangle collisions. This approach is not exact but works sufficiently well.
    Particle vertex collisions are neglected if one or more particle face collisions have been detected.

    """
    
    Unbind = []

    System.virial_scalar[:] = 0.0
    System.virial_scalar_Wall[:] = 0.0    
    System.virial_tensor[:,:,:] = 0.0
    System.virial_tensor_Wall[:,:,:] = 0.0
    
    pos_tri = np.empty(3, dtype = np.float64)
    
    virial_tensor = System.virial_tensor

    box_lengths = System.box_lengths
    empty = System.empty
    x_c_min = np.empty(System.dim)
    x_c_max = np.empty(System.dim)
    
    System.Epot[:] = 0.0 #Potential Energy
    
    
    pos_shift = np.empty(System.dim)  # position shift to account for periodic boundary conditions.
    dF = np.empty(System.dim)
    
    # Initialize
    HGrid[0]['head'].fill(empty)

    HGrid[0]['ls'].fill(empty)
    

    # Loop over all particles and place them in cells, calculate bond forces and update reaction list.
    for i0 in range(Particles.occupied.n):
        i = Particles.occupied[i0]
        
        ptype_idx_i = Particles[i]['type_id']  
        pos_i = Particles[i]['pos']
        
        mol_idx_i = Particles[i]['rb_id']
        mtype_idx_i = RBs[mol_idx_i]['type_id']
        mpos_i = RBs[mol_idx_i]['pos']
        
        # within_box = True
        # if System.boundary_condition_id == 1:
        # Check if particle is within box volume:
        within_box = -box_lengths[0]/2<=pos_i[0]<box_lengths[0]/2 and -box_lengths[1]/2<=pos_i[1]<box_lengths[1]/2 and -box_lengths[2]/2<=pos_i[2]<box_lengths[2]/2
        
        if within_box:
            
            #Only add the particle to the linked cell list if there is actually any pair-interaction or bimol-reaction defined for this partricle type:
            if System.pair_interaction[ptype_idx_i]:
            
                h = Particles[i]['h']
                h_start = HGrid[0]['head_start'][h]
                cells_per_dim_prim = HGrid[0]['cells_per_dim'][h]
                
                # Determine what cell, in each direction, the i-th particle is in
                cx = int((pos_i[0]+box_lengths[0]/2) / HGrid[0]['sh'][h][0])
                cy = int((pos_i[1]+box_lengths[1]/2) / HGrid[0]['sh'][h][1])
                cz = int((pos_i[2]+box_lengths[2]/2) / HGrid[0]['sh'][h][2])
                
                # within_box = True
                
                # if System.boundary_condition_id == 1:
                #     # Check if particle is within box volume:
                #     within_box = 0<=cx<cells_per_dim_prim[0] and 0<=cy<cells_per_dim_prim[1] and 0<=cz<cells_per_dim_prim[2]
            
            # if within_box:
                
                c = cx + cy * cells_per_dim_prim[0] + cz * cells_per_dim_prim[0] * cells_per_dim_prim[1]
                # List of particle indices occupying a given cell
                HGrid[0]['ls'][i] = HGrid[0]['head'][h_start+c]
        
                # The last particle found to lie in cell c (head particle)
                HGrid[0]['head'][h_start+c] = i
        
        #%%
        
        # Append any unimolecular reaction that is due to the reactions list:
        if len(System.Reactions_Dict)>0:
            if Particles[i]['next_transition']<=System.current_step*System.dt:
                
                reaction_id = System.up_reaction_id[ptype_idx_i]
                
                System.reactions_left += 1
                System.Reactions_Dict[reaction_id].append_reaction(i)
                
        #%%

        if System.boundary_condition_id == 1 and Particles[i]['radius'] > 0.0 and RBs[mol_idx_i]['collision_type'] == 0:
            
            type_name = str(System.particle_id_to_name[Particles[i]['type_id']])
            radius = System.particle_types[type_name][0]['radius']
                    
            for dim in range(3):
                
                sign = np.sign(pos_i[dim])
                r = abs(pos_i[dim]-sign*box_lengths[dim]/2)
                
                if r<radius:
                    
                    dx = pos_i[dim]-sign*box_lengths[dim]/2
                    pot, fr = potu.wall(r, radius, System.wall_force)                            
                    
                    fr /= r
                    System.Epot[mtype_idx_i] += pot
                    
                    Particles[i]['force'][dim] += dx * fr
                    
                    dx_mol = mpos_i[dim] - sign*box_lengths[dim]/2
                    
                    System.virial_scalar_Wall[mtype_idx_i] += fr*dx_mol
                    
                    
                    
        j = Particles[i]['bound_with']
        
        if Particles[i]['bound'] == True and i<j:
                       
            mol_idx_j = Particles[j]['rb_id']
            ptype_idx_j = Particles[j]['type_id']
            pos_j = Particles[j]['pos']
            
            dx = pos_i[0] - pos_j[0]
            dy = pos_i[1] - pos_j[1]
            dz = pos_i[2] - pos_j[2]
            
            if System.boundary_condition_id == 0:
                dx += box_lengths[0] * (dx < -box_lengths[0]/2) - box_lengths[0]*(dx >= box_lengths[0]/2)
                dy += box_lengths[1] * (dy < -box_lengths[1]/2) - box_lengths[1]*(dy >= box_lengths[1]/2)
                dz += box_lengths[2] * (dz < -box_lengths[2]/2) - box_lengths[2]*(dz >= box_lengths[2]/2)
                
            r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            
            if 0.0 < r < System.interaction_args[ptype_idx_i][ptype_idx_j]['cutoff']:
                
                mtype_idx_j = RBs[mol_idx_j]['type_id']
                mpos_j = RBs[mol_idx_j]['pos']
                                    
                dx_mol = mpos_i[0] - mpos_j[0]
                dy_mol = mpos_i[1] - mpos_j[1]
                dz_mol = mpos_i[2] - mpos_j[2]
                
                if System.boundary_condition_id == 0:
                    
                    if dx_mol<=-box_lengths[0]/2:
                        dx_mol += box_lengths[0]
                    elif dx_mol>=box_lengths[0]/2:
                        dx_mol -= box_lengths[0]
                    
                    if dy_mol<=-box_lengths[1]/2:
                        dy_mol += box_lengths[1]
                    elif dy_mol>=box_lengths[1]/2:
                        dy_mol -= box_lengths[1]
                    
                    if dz_mol<=-box_lengths[2]/2:
                        dz_mol += box_lengths[2]
                    elif dz_mol>=box_lengths[2]/2:
                        dz_mol -= box_lengths[2]                
                                
                
                pot, fr = potu.execute_force(System.interaction_args[ptype_idx_i][ptype_idx_j]['id'], r, System.interaction_args[ptype_idx_i][ptype_idx_j]['parameters'])
                
                fr /= r
                System.Epot[mtype_idx_i] += pot
                System.Epot[mtype_idx_j] += pot

                # Update force for the i-th particle:
                    
                dF[0] = dx * fr 
                dF[1] = dy * fr 
                dF[2] = dz * fr
                
                Particles[i]['force'][0] += dF[0]
                Particles[i]['force'][1] += dF[1]
                Particles[i]['force'][2] += dF[2]
                
                # Newton's 3rd law:
                Particles[j]['force'][0] -= dF[0]
                Particles[j]['force'][1] -= dF[1]
                Particles[j]['force'][2] -= dF[2]
                
                System.virial_scalar[mtype_idx_i] += dF[0]*dx_mol+dF[1]*dy_mol+dF[2]*dz_mol
                System.virial_scalar[mtype_idx_j] += dF[0]*dx_mol+dF[1]*dy_mol+dF[2]*dz_mol
                
                virial_tensor[mtype_idx_i][0, 0] += dx_mol * dF[0]
                virial_tensor[mtype_idx_i][0, 1] += dx_mol * dF[1]
                virial_tensor[mtype_idx_i][0, 2] += dx_mol * dF[2]
                virial_tensor[mtype_idx_i][1, 0] += dy_mol * dF[0]
                virial_tensor[mtype_idx_i][1, 1] += dy_mol * dF[1]
                virial_tensor[mtype_idx_i][1, 2] += dy_mol * dF[2]
                virial_tensor[mtype_idx_i][2, 0] += dz_mol * dF[0]
                virial_tensor[mtype_idx_i][2, 1] += dz_mol * dF[1]
                virial_tensor[mtype_idx_i][2, 2] += dz_mol * dF[2]
                
                virial_tensor[mtype_idx_j][0, 0] += dx_mol * dF[0]
                virial_tensor[mtype_idx_j][0, 1] += dx_mol * dF[1]
                virial_tensor[mtype_idx_j][0, 2] += dx_mol * dF[2]
                virial_tensor[mtype_idx_j][1, 0] += dy_mol * dF[0]
                virial_tensor[mtype_idx_j][1, 1] += dy_mol * dF[1]
                virial_tensor[mtype_idx_j][1, 2] += dy_mol * dF[2]
                virial_tensor[mtype_idx_j][2, 0] += dz_mol * dF[0]
                virial_tensor[mtype_idx_j][2, 1] += dz_mol * dF[1]
                virial_tensor[mtype_idx_j][2, 2] += dz_mol * dF[2]
            
            # if the bond is a breakable bond:
            elif System.interaction_args[ptype_idx_i][ptype_idx_j]['breakable'] == True:
                
                Unbind.append([i,j])
                

#%%
    
        # Interaction between particle and mesh faces. Need to check several conditions:
            # 1. Is the particle inside the box volume
            # 2. Is the particle part of a volume molecule
            # 3. Is the particle radius > 0
            # 4. Is the collision type of the particle == 0. (For particles of radius 0, this should never be the case:
            # TODO: Add a test at the start of each simualtion and set the collision type of all radius = 0 particles to 1. This would save us from testing 3.)!
            
        if within_box and System.mesh and RBs[mol_idx_i]['loc_id'] == 0 and Particles[i]['radius'] > 0.0 and RBs[mol_idx_i]['collision_type'] == 0:
            
            radius = Particles[i]['radius']
            face_intersection_count = 0
            r_edge = 1e10
            pp_edge = np.empty(3, dtype = np.float64)
            
            if RBs[mol_idx_i]['compartment'] == 0:
                face_sign = 1
            else:
                face_sign = -1
                
            # face_intersection = False
            
            dF[0] = 0.0
            dF[1] = 0.0
            dF[2] = 0.0
            
            System.time_stamp += 1
                
            cx = int((pos_i[0]+box_lengths[0]/2) / System.cell_length_per_dim[0])
            cy = int((pos_i[1]+box_lengths[1]/2) / System.cell_length_per_dim[1])
            cz = int((pos_i[2]+box_lengths[2]/2) / System.cell_length_per_dim[2])
            
            dcx = int(np.ceil((2*radius)/System.cell_length_per_dim[0]))
            dcy = int(np.ceil((2*radius)/System.cell_length_per_dim[1]))
            dcz = int(np.ceil((2*radius)/System.cell_length_per_dim[2]))
            
            cz_start, cz_end = cz - dcz, cz + dcz + 1
            cy_start, cy_end = cy - dcy, cy + dcy + 1
            cx_start, cx_end = cx - dcx, cx + dcx + 1
            
            if System.boundary_condition_id != 0:
                
                if cz_start < 0:
                    cz_start = 0 
                elif cz_end > System.cells_per_dim[2]:
                    cz_end = System.cells_per_dim[2]
                    
                if cy_start < 0:
                    cy_start = 0 
                elif cy_end > System.cells_per_dim[1]:
                    cy_end = System.cells_per_dim[1]
                    
                if cx_start < 0:
                    cx_start = 0 
                elif cx_end > System.cells_per_dim[0]:
                    cx_end = System.cells_per_dim[0]            
            
            
            for cz_N in range(cz_start, cz_end):
                cz_shift = 0 + System.cells_per_dim[2] * (cz_N < 0) - System.cells_per_dim[2] * (cz_N >= System.cells_per_dim[2])
                pos_shift[2] = 0.0 - box_lengths[2] * (cz_N < 0) + box_lengths[2]*(cz_N >= System.cells_per_dim[2])
    
                for cy_N in range(cy_start, cy_end):
                    cy_shift = 0 + System.cells_per_dim[1] * (cy_N < 0) - System.cells_per_dim[1] * (cy_N >= System.cells_per_dim[1])
                    pos_shift[1] = 0.0 - box_lengths[1] * (cy_N < 0) + box_lengths[1] * (cy_N >= System.cells_per_dim[1])
    
                    for cx_N in range(cx_start, cx_end):
                        cx_shift = 0 + System.cells_per_dim[0] * (cx_N < 0) - System.cells_per_dim[0] * (cx_N >= System.cells_per_dim[0])
                        pos_shift[0] = 0.0 - box_lengths[0] * (cx_N < 0) + box_lengths[0] * (cx_N >= System.cells_per_dim[0])
    
                        # Compute the location of the N-th cell:
                        c_N = (cx_N + cx_shift) + (cy_N + cy_shift) * System.cells_per_dim[0] \
                              + (cz_N + cz_shift) * System.cells_per_dim[0] * System.cells_per_dim[1]
                        
            
                        head = System.CellList.head[c_N]
                        if head != -1:

                            Tri_idx = System.CellList[head]['id']
                            
                            if System.Mesh[Tri_idx]['stamp']!=System.time_stamp:
                                System.Mesh[Tri_idx]['stamp']=System.time_stamp
                                
                                triangle = System.Mesh[Tri_idx]['triangles']
                                
                                if System.boundary_condition_id == 0:
                                    p0 = System.vertices[triangle[0]] + pos_shift
                                    p1 = System.vertices[triangle[1]] + pos_shift
                                    p2 = System.vertices[triangle[2]] + pos_shift
                                    
                                else:
                                    p0 = System.vertices[triangle[0]]
                                    p1 = System.vertices[triangle[1]]
                                    p2 = System.vertices[triangle[2]]
                        
                                r, region = point_triangle_distance(p0,p1,p2,pos_i, pos_tri)
                                
                                if region == 0:
                                    # face_intersection = True

                                    if 0.0 < r < radius:
                                        
                                        normal = face_sign*System.Mesh[Tri_idx]['triangle_coord'][3]
                                        
                                        dx = abs(pos_i[0] - pos_tri[0])
                                        dy = abs(pos_i[1] - pos_tri[1])
                                        dz = abs(pos_i[2] - pos_tri[2])
                                        
                                        face_intersection_count+=1
                                        
                                        pot, fr = potu.wall(r, radius, System.wall_force)                            
                                        
                                        fr /= r
                                        System.Epot[mtype_idx_i] += pot
                    
                                        # Update the force for particle i
                                        dF[0] += dx * fr * normal[0]
                                        dF[1] += dy * fr * normal[1] 
                                        dF[2] += dz * fr * normal[2]
                                        
                                        
                                else:
                                    
                                    if r<r_edge:
                                        r_edge = r
                                        pp_edge[:] = pos_tri
                                
                            next = System.CellList[head]['next']
                                    
                            while next!=-1:
                                
                                
                                Tri_idx = System.CellList[next]['id']
                                
                                # Only continue if this triangle has not yet been checked for collision!
                                if System.Mesh[Tri_idx]['stamp']!=System.time_stamp:
                                    System.Mesh[Tri_idx]['stamp']=System.time_stamp
                                    
                                    triangle = System.Mesh[Tri_idx]['triangles']
                                    
                                    if System.boundary_condition_id == 0:
                                        p0 = System.vertices[triangle[0]] + pos_shift
                                        p1 = System.vertices[triangle[1]] + pos_shift
                                        p2 = System.vertices[triangle[2]] + pos_shift
                                        
                                    else:
                                        p0 = System.vertices[triangle[0]]
                                        p1 = System.vertices[triangle[1]]
                                        p2 = System.vertices[triangle[2]]
                            
                                    r, region = point_triangle_distance(p0,p1,p2,pos_i, pos_tri)
                                    
                                    if region == 0:
                                        # face_intersection = True
                                        
                                        if 0.0 < r < radius:
                                            
                                            normal = face_sign*System.Mesh[Tri_idx]['triangle_coord'][3]
                                            
                                            dx = abs(pos_i[0] - pos_tri[0])
                                            dy = abs(pos_i[1] - pos_tri[1])
                                            dz = abs(pos_i[2] - pos_tri[2])
                                            
                                            face_intersection_count+=1
                                            
                                            pot, fr = potu.wall(r, radius, System.wall_force)                            
                                            
                                            fr /= r
                                            System.Epot[mtype_idx_i] += pot
                        
                                            # Update the force for particle i
                                            dF[0] += dx * fr * normal[0]
                                            dF[1] += dy * fr * normal[1]
                                            dF[2] += dz * fr * normal[2]
                                            
                                            
                                    else:
                                        
                                        if r<r_edge:
                                            r_edge = r
                                            pp_edge[:] = pos_tri
                    
                                next = System.CellList[next]['next']
         
        
            if face_intersection_count>0:
                Particles[i]['force'][0] += dF[0]/face_intersection_count
                Particles[i]['force'][1] += dF[1]/face_intersection_count
                Particles[i]['force'][2] += dF[2]/face_intersection_count
            elif 0.0 < r_edge < radius:
                
                dx = pos_i[0] - pp_edge[0]
                dy = pos_i[1] - pp_edge[1]
                dz = pos_i[2] - pp_edge[2]
                
                face_intersection_count+=1
                
                pot, fr = potu.wall(r_edge, radius, System.wall_force)                            
                
                fr /= r_edge
                System.Epot[mtype_idx_i] += pot

                # Update the force for particle i
                Particles[i]['force'][0] += dx * fr 
                Particles[i]['force'][1] += dy * fr 
                Particles[i]['force'][2] += dz * fr  
                
            # if count>1:
            #     print('triangle contacts: ', count, i, System.current_step)
            
#%%
                
    
    # Loop over all Particles
    for i0 in range(Particles.occupied.n):
        i = Particles.occupied[i0] 
        
        ptype_idx_i = Particles[i]['type_id']  
        #Only continue if there is actually any pair-interaction or bimol-reaction defined for this partricle type:
        if System.pair_interaction[ptype_idx_i]:
        
            mol_idx_i = Particles[i]['rb_id']
            mtype_idx_i = RBs[mol_idx_i]['type_id']
            mpos_i = RBs[mol_idx_i]['pos']
            comp_i = RBs[mol_idx_i]['compartment']
            loc_type_i = RBs[mol_idx_i]['loc_id']
            
            pos_i = Particles[i]['pos']
            
            h = Particles[i]['h']
            h_start = HGrid[0]['head_start'][h]
            cells_per_dim_prim = HGrid[0]['cells_per_dim'][h]
            
            cx = int((pos_i[0]+box_lengths[0]/2) / HGrid[0]['sh'][h][0])
            cy = int((pos_i[1]+box_lengths[1]/2) / HGrid[0]['sh'][h][1])
            cz = int((pos_i[2]+box_lengths[2]/2) / HGrid[0]['sh'][h][2])
            
            cz_start, cz_end = cz - 1, cz + 2
            cy_start, cy_end = cy - 1, cy + 2
            cx_start, cx_end = cx - 1, cx + 2
            
            # within_box = True
            # Check if particle is within box volume:
            within_box = 0<=cx<cells_per_dim_prim[0] and 0<=cy<cells_per_dim_prim[1] and 0<=cz<cells_per_dim_prim[2]
                
            if System.boundary_condition_id != 0:
                # # Check if particle is within box volume:
                # within_box = 0<=cx<cells_per_dim_prim[0] and 0<=cy<cells_per_dim_prim[1] and 0<=cz<cells_per_dim_prim[2]
                
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
            
            if within_box:
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
                                
                                mol_idx_j = Particles[j]['rb_id']
                                ptype_idx_j = Particles[j]['type_id']
                                pos_j = Particles[j]['pos']
                                
                                
                                if i < j and mol_idx_i!=mol_idx_j and Particles[i]['bound_with']!=j and System.interaction_defined[ptype_idx_i][ptype_idx_j]:
                                    
                                    # We only allow interactions and reactions between volume molecules of the same compartment:
                                    interaction_allowed, reactions_allowed = react_interact_test(comp_i, loc_type_i, mol_idx_j, mol_idx_i, RBs, System)
                                    # If neither reactions nor interactions are allow, we can go to the next particle and continue (if interaction_allowes == False, reactions_allowe dis always also False):
                                    if interaction_allowed == False:
                                        j = HGrid[0]['ls'][j]
                                        continue
                                        
                                    System.count+=1
                                    
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
                                    
        
                                    #Binding Reactions:
                                    if System.reaction_args[ptype_idx_i][ptype_idx_j]['bond'] == True:
                                        if (Particles[i]['bound'] == Particles[j]['bound'] == False):
                                            if r <= System.reaction_args[ptype_idx_i][ptype_idx_j]['radius']:
                                            
                                                System.reactions_left += 1
                                                
                                                reaction_id = System.reaction_args[ptype_idx_i][ptype_idx_j]['id']
        
                                                System.Reactions_Dict[reaction_id].append_reaction(i,j)
                                                
                                    # Biparticle reactions (Particle pairs can either take part in a bindin reaction xor in 
                                    # in a biparticle reaction:
                                    elif System.reaction_args[ptype_idx_i][ptype_idx_j]['defined'] == True and reactions_allowed:
                                        if r <= System.reaction_args[ptype_idx_i][ptype_idx_j]['radius']:
                                            
                                            System.reactions_left += 1
                                            reaction_id = System.reaction_args[ptype_idx_i][ptype_idx_j]['id']
                                            # Which is the enzyme molecule (inserted second)?
                                            enzyme_id = System.reaction_args[ptype_idx_i][ptype_idx_j]['enzyme']
                                            if ptype_idx_j == enzyme_id:
                                                System.Reactions_Dict[reaction_id].append_reaction(i,j)
                                            else:
                                                System.Reactions_Dict[reaction_id].append_reaction(j,i)
        
                                            
                                    if System.interaction_args[ptype_idx_i][ptype_idx_j]['global']==1: # TODO: Is this condition check actually necessary (we do not coninue until here for bound particle pairs anyway!)?
                                        if 0.0 < r < System.interaction_args[ptype_idx_i][ptype_idx_j]['cutoff']:
                                            
                                            mtype_idx_j = RBs[mol_idx_j]['type_id']
                                            mpos_j = RBs[mol_idx_j]['pos']
                                            
        
                                            dx_mol = mpos_i[0] - mpos_j[0]
                                            if dx_mol<=-box_lengths[0]/2:
                                                dx_mol += box_lengths[0]
                                            elif dx_mol>=box_lengths[0]/2:
                                                dx_mol -= box_lengths[0]
                                            
                                            dy_mol = mpos_i[1] - mpos_j[1]
                                            if dy_mol<=-box_lengths[1]/2:
                                                dy_mol += box_lengths[1]
                                            elif dy_mol>=box_lengths[1]/2:
                                                dy_mol -= box_lengths[1]
                                                
                                            dz_mol = mpos_i[2] - mpos_j[2]
                                            if dz_mol<=-box_lengths[2]/2:
                                                dz_mol += box_lengths[2]
                                            elif dz_mol>=box_lengths[2]/2:
                                                dz_mol -= box_lengths[2]                
                            
        
                                            pot, fr = potu.execute_force(System.interaction_args[ptype_idx_i][ptype_idx_j]['id'], r, System.interaction_args[ptype_idx_i][ptype_idx_j]['parameters'])
                                            
                                            fr /= r
                                            System.Epot[mtype_idx_i] += pot
                                            System.Epot[mtype_idx_j] += pot
        
                                            # Update force of the i-th particle
                                            dF[0] = dx * fr 
                                            dF[1] = dy * fr 
                                            dF[2] = dz * fr
                                            
                                            Particles[i]['force'][0] += dF[0]
                                            Particles[i]['force'][1] += dF[1]
                                            Particles[i]['force'][2] += dF[2]
                            
                                            # Newton's 3rd law
                                            Particles[j]['force'][0] -= dF[0]
                                            Particles[j]['force'][1] -= dF[1]
                                            Particles[j]['force'][2] -= dF[2]
                                            
                                            System.virial_scalar[mtype_idx_i] += dF[0]*dx_mol+dF[1]*dy_mol+dF[2]*dz_mol
                                            System.virial_scalar[mtype_idx_j] += dF[0]*dx_mol+dF[1]*dy_mol+dF[2]*dz_mol
                                            
                                            virial_tensor[mtype_idx_i][0, 0] += dx_mol * dF[0]
                                            virial_tensor[mtype_idx_i][0, 1] += dx_mol * dF[1]
                                            virial_tensor[mtype_idx_i][0, 2] += dx_mol * dF[2]
                                            virial_tensor[mtype_idx_i][1, 0] += dy_mol * dF[0]
                                            virial_tensor[mtype_idx_i][1, 1] += dy_mol * dF[1]
                                            virial_tensor[mtype_idx_i][1, 2] += dy_mol * dF[2]
                                            virial_tensor[mtype_idx_i][2, 0] += dz_mol * dF[0]
                                            virial_tensor[mtype_idx_i][2, 1] += dz_mol * dF[1]
                                            virial_tensor[mtype_idx_i][2, 2] += dz_mol * dF[2]
                                            
                                            virial_tensor[mtype_idx_j][0, 0] += dx_mol * dF[0]
                                            virial_tensor[mtype_idx_j][0, 1] += dx_mol * dF[1]
                                            virial_tensor[mtype_idx_j][0, 2] += dx_mol * dF[2]
                                            virial_tensor[mtype_idx_j][1, 0] += dy_mol * dF[0]
                                            virial_tensor[mtype_idx_j][1, 1] += dy_mol * dF[1]
                                            virial_tensor[mtype_idx_j][1, 2] += dy_mol * dF[2]
                                            virial_tensor[mtype_idx_j][2, 0] += dz_mol * dF[0]
                                            virial_tensor[mtype_idx_j][2, 1] += dz_mol * dF[1]
                                            virial_tensor[mtype_idx_j][2, 2] += dz_mol * dF[2]
                                            
                                # Move down list (ls) of particles for cell interactions with a head particle
                                j = HGrid[0]['ls'][j]
        
                #-----------------------------------------------------------
        
                hj = h-1
                while hj>=0:
                    
                    cells_per_dim_sec = HGrid[0]['cells_per_dim'][hj]
                    
                    h_start = HGrid[0]['head_start'][hj]
                    
                    x_c_min[0] = pos_i[0] - (Particles[i]['cutoff'] + HGrid[0]['sh'][hj][0]/2)
                    x_c_min[1] = pos_i[1] - (Particles[i]['cutoff'] + HGrid[0]['sh'][hj][1]/2)
                    x_c_min[2] = pos_i[2] - (Particles[i]['cutoff'] + HGrid[0]['sh'][hj][2]/2)
            
                    cx_start = np.floor((x_c_min[0]+box_lengths[0]/2) / HGrid[0]['sh'][hj][0])
                    cy_start = np.floor((x_c_min[1]+box_lengths[1]/2) / HGrid[0]['sh'][hj][1])
                    cz_start = np.floor((x_c_min[2]+box_lengths[2]/2) / HGrid[0]['sh'][hj][2])
                    
                    
                    x_c_max[0] = pos_i[0] + (Particles[i]['cutoff'] + HGrid[0]['sh'][hj][0]/2)
                    x_c_max[1] = pos_i[1] + (Particles[i]['cutoff'] + HGrid[0]['sh'][hj][1]/2)
                    x_c_max[2] = pos_i[2] + (Particles[i]['cutoff'] + HGrid[0]['sh'][hj][2]/2)
            
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
                                    
                                    mol_idx_j = Particles[j]['rb_id']
                                    ptype_idx_j = Particles[j]['type_id']
                                    pos_j = Particles[j]['pos']
                                            
                                            
                                    if mol_idx_i!=mol_idx_j and Particles[i]['bound_with']!=j and System.interaction_defined[ptype_idx_i][ptype_idx_j]:
                                        
                                        interaction_allowed, reactions_allowed = react_interact_test(comp_i, loc_type_i, mol_idx_j, mol_idx_i, RBs, System)
                                        # If neither reactions nor interactions are allow, we can go to the next particle and continue (if interaction_allowes == False, reactions_allowe dis always also False):
                                        if interaction_allowed == False:
                                            j = HGrid[0]['ls'][j]
                                            continue
                                        
                                        System.count+=1
                                        
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
                                        
            
                                        #Binding Reactions:
                                        if System.reaction_args[ptype_idx_i][ptype_idx_j]['bond'] == True:
                                            if (Particles[i]['bound'] == Particles[j]['bound'] == False):
                                                if r <= System.reaction_args[ptype_idx_i][ptype_idx_j]['radius']:
                                                
                                                    System.reactions_left += 1
                                                    
                                                    reaction_id = System.reaction_args[ptype_idx_i][ptype_idx_j]['id']
            
                                                    System.Reactions_Dict[reaction_id].append_reaction(i,j)
                                                    
                                        # Biparticle reactions (Particle pairs can either take part in a bindin reaction xor in 
                                        # in a biparticle reaction:
                                        elif System.reaction_args[ptype_idx_i][ptype_idx_j]['defined'] == True and reactions_allowed:
                                            if r <= System.reaction_args[ptype_idx_i][ptype_idx_j]['radius']:
                                                
                                                System.reactions_left += 1
                                                reaction_id = System.reaction_args[ptype_idx_i][ptype_idx_j]['id']
                                                # Which is the enzyme molecule (inserted second)?
                                                enzyme_id = System.reaction_args[ptype_idx_i][ptype_idx_j]['enzyme']
                                                if ptype_idx_j == enzyme_id:
                                                    System.Reactions_Dict[reaction_id].append_reaction(i,j)
                                                else:
                                                    System.Reactions_Dict[reaction_id].append_reaction(j,i)
                                                    
                                                
                                        if System.interaction_args[ptype_idx_i][ptype_idx_j]['global']==1:
                                            
                                            if 0.0 < r < System.interaction_args[ptype_idx_i][ptype_idx_j]['cutoff']:
        
                                                mtype_idx_j = RBs[mol_idx_j]['type_id']
                                                mpos_j = RBs[mol_idx_j]['pos']
                                                
                                                dx_mol = mpos_i[0] - mpos_j[0]
                                                if dx_mol<=-box_lengths[0]/2:
                                                    dx_mol += box_lengths[0]
                                                elif dx_mol>=box_lengths[0]/2:
                                                    dx_mol -= box_lengths[0]
                                                
                                                dy_mol = mpos_i[1] - mpos_j[1]
                                                if dy_mol<=-box_lengths[1]/2:
                                                    dy_mol += box_lengths[1]
                                                elif dy_mol>=box_lengths[1]/2:
                                                    dy_mol -= box_lengths[1]
                                                    
                                                dz_mol = mpos_i[2] - mpos_j[2]
                                                if dz_mol<=-box_lengths[2]/2:
                                                    dz_mol += box_lengths[2]
                                                elif dz_mol>=box_lengths[2]/2:
                                                    dz_mol -= box_lengths[2]                
                            
        
                                                pot, fr = potu.execute_force(System.interaction_args[ptype_idx_i][ptype_idx_j]['id'], r, System.interaction_args[ptype_idx_i][ptype_idx_j]['parameters'])
                                                
                                                fr /= r
                                                System.Epot[mtype_idx_i] += pot
                                                System.Epot[mtype_idx_j] += pot
            
                                                # Update force for particle i
                                                dF[0] = dx * fr 
                                                dF[1] = dy * fr 
                                                dF[2] = dz * fr
                                                
                                                Particles[i]['force'][0] += dF[0]
                                                Particles[i]['force'][1] += dF[1]
                                                Particles[i]['force'][2] += dF[2]
                                                
                                                # Apply Newton's 3rd law
                                                Particles[j]['force'][0] -= dF[0]
                                                Particles[j]['force'][1] -= dF[1]
                                                Particles[j]['force'][2] -= dF[2]
                                                
                                                System.virial_scalar[mtype_idx_i] += dF[0]*dx_mol+dF[1]*dy_mol+dF[2]*dz_mol
                                                System.virial_scalar[mtype_idx_j] += dF[0]*dx_mol+dF[1]*dy_mol+dF[2]*dz_mol
                                                
                                                virial_tensor[mtype_idx_i][0, 0] += dx_mol * dF[0]
                                                virial_tensor[mtype_idx_i][0, 1] += dx_mol * dF[1]
                                                virial_tensor[mtype_idx_i][0, 2] += dx_mol * dF[2]
                                                virial_tensor[mtype_idx_i][1, 0] += dy_mol * dF[0]
                                                virial_tensor[mtype_idx_i][1, 1] += dy_mol * dF[1]
                                                virial_tensor[mtype_idx_i][1, 2] += dy_mol * dF[2]
                                                virial_tensor[mtype_idx_i][2, 0] += dz_mol * dF[0]
                                                virial_tensor[mtype_idx_i][2, 1] += dz_mol * dF[1]
                                                virial_tensor[mtype_idx_i][2, 2] += dz_mol * dF[2]
                                                
                                                virial_tensor[mtype_idx_j][0, 0] += dx_mol * dF[0]
                                                virial_tensor[mtype_idx_j][0, 1] += dx_mol * dF[1]
                                                virial_tensor[mtype_idx_j][0, 2] += dx_mol * dF[2]
                                                virial_tensor[mtype_idx_j][1, 0] += dy_mol * dF[0]
                                                virial_tensor[mtype_idx_j][1, 1] += dy_mol * dF[1]
                                                virial_tensor[mtype_idx_j][1, 2] += dy_mol * dF[2]
                                                virial_tensor[mtype_idx_j][2, 0] += dz_mol * dF[0]
                                                virial_tensor[mtype_idx_j][2, 1] += dz_mol * dF[1]
                                                virial_tensor[mtype_idx_j][2, 2] += dz_mol * dF[2]
                                    
                                            
                                # Move down list (ls) of particles for cell interactions with a head particle
                                    j = HGrid[0]['ls'][j]
                            
                    hj -= 1


    System.virial_scalar*=1/2
    System.virial_tensor*=1/2
    
    for i,j in Unbind:
        
        # If the binding reaction changed the particle types, reverse that:
        educt_type_i = System.particle_types[str(Particles[i]['type'])][0]['bond_educt_id']
        if educt_type_i != -1:
            convert_particle_type(System, educt_type_i, Particles, i)
            # Note: if we convert particle type i, the binding partner j is automatically also converted back, this is already taken care of in convert_particle_type().
        else:
            Particles[i]['bound'] = False
            Particles[j]['bound'] = False
            Particles[i]['bound_with'] = -1
            Particles[j]['bound_with'] = -1
            
            System.N_bonds[Particles[i]['type_id']][Particles[j]['type_id']] -= 1
            System.N_bonds[Particles[j]['type_id']][Particles[i]['type_id']] -= 1
        




