# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""


import numba as nb
import numpy as np

from ..math import random_util as randu
from ..system import distribute_vol_util as distribute_vol
from ..system import distribute_surface_util as distribute_surf
from ..geometry.intersections_util import any_ray_mesh_intersection_test
from ..system import potentials_util as potu

#%%

@nb.njit
def update_bond_force(i,j,Particles, RBs, System):
    
    box_lengths = System.box_lengths
    dF = np.empty(System.dim)
    
    ptype_idx_i = Particles[i]['type_id']  
    pos_i = Particles[i]['pos']
    
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
        
        mol_idx_i = Particles[i]['rb_id']
        mtype_idx_i = RBs[mol_idx_i]['type_id']
        mpos_i = RBs[mol_idx_i]['pos']
        
        mol_idx_j = Particles[j]['rb_id']
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
        
        System.virial_tensor[mtype_idx_i][0, 0] += dx_mol * dF[0]
        System.virial_tensor[mtype_idx_i][0, 1] += dx_mol * dF[1]
        System.virial_tensor[mtype_idx_i][0, 2] += dx_mol * dF[2]
        System.virial_tensor[mtype_idx_i][1, 0] += dy_mol * dF[0]
        System.virial_tensor[mtype_idx_i][1, 1] += dy_mol * dF[1]
        System.virial_tensor[mtype_idx_i][1, 2] += dy_mol * dF[2]
        System.virial_tensor[mtype_idx_i][2, 0] += dz_mol * dF[0]
        System.virial_tensor[mtype_idx_i][2, 1] += dz_mol * dF[1]
        System.virial_tensor[mtype_idx_i][2, 2] += dz_mol * dF[2]
        
        System.virial_tensor[mtype_idx_j][0, 0] += dx_mol * dF[0]
        System.virial_tensor[mtype_idx_j][0, 1] += dx_mol * dF[1]
        System.virial_tensor[mtype_idx_j][0, 2] += dx_mol * dF[2]
        System.virial_tensor[mtype_idx_j][1, 0] += dy_mol * dF[0]
        System.virial_tensor[mtype_idx_j][1, 1] += dy_mol * dF[1]
        System.virial_tensor[mtype_idx_j][1, 2] += dy_mol * dF[2]
        System.virial_tensor[mtype_idx_j][2, 0] += dz_mol * dF[0]
        System.virial_tensor[mtype_idx_j][2, 1] += dz_mol * dF[1]
        System.virial_tensor[mtype_idx_j][2, 2] += dz_mol * dF[2]

#%%

@nb.njit
def delete_molecule(System, RBs, Particles, i):

    """Removes a molecule and its reactions from the simulation.
    
    Parameters
    ----------
    System : `object`
        Instance of System class
    RBs : `object`
        Instance of RBs class
    Particles : `object`
        Instance of Particles class
    i : `int64`
        Index of the molecule which to delete
    
    """
    
    delete_reactions(System, i, RBs[i]['type_id'], False)
    
    delete_particles(System, RBs, Particles, i)
            
    #Delete the rigidbody from the RB dictionary:
    RBs.delete(i)
    System.N -= 1
    System.Nmol[RBs[i]['type_id']] -= 1
    
    
    
    
@nb.njit
def delete_particles(System, RBs, Particles, i):

    """Removes all particle belonging to a molecule and its reactions and bonds from the simulation.
    
    Parameters
    ----------
    System : `object`
        Instance of System class
    RBs : `object`
        Instance of RBs class
    Particles : `object`
        Instance of Particles class
    i : `int64`
        Index of the molecule whose particles to delete
    
    """
    
    #loop over all particles that are part of this rigid body and delete them as well as all their reactions:
    for pi0 in range(RBs[i]['topology_N']):
        pi = RBs[i]['topology'][pi0]
        
        # Delete bond of partner particles if a bond exists:
        if Particles[pi]['bound'] == True:
            pj = Particles[pi]['bound_with']
            Particles[pj]['bound'] = False
            Particles[pj]['bound_with'] = -1 
            
            System.N_bonds[Particles[pi]['type_id']][Particles[pj]['type_id']] -= 1
            System.N_bonds[Particles[pj]['type_id']][Particles[pi]['type_id']] -= 1
            
            # If the binding reaction changed the particle type of pj, reverse that:
            educt_type_j = System.particle_types[str(Particles[pj]['type'])][0]['bond_educt_id']
            if educt_type_j != -1:
                
                delete_reactions(System, pj, Particles[pj]['type_id'], True)
                
                Particles[pj]['type_id'] = educt_type_j
                Particles[pj]['type'] = System.particle_id_to_name[educt_type_j]
                Particles.next_up_reaction(System, pj)
                
                
        delete_reactions(System, pi, Particles[pi]['type_id'], True)  
        
        # Delete Particle
        Particles.delete(pi)
        
        System.Np -= 1
        
          
            
#%%   

@nb.njit
def delete_reactions(System, i, type_id, educt_is_particle):#, reaction_educt_type_id):

    """Removes all reactions of an educt i scheduled for this time step.
    
    Parameters
    ----------
    System : `object`
        Instance of System class
    i : `int64`
        Index of the particle for which to delete all reactions
    
    """
    
    # Delete particle reactions
    if educt_is_particle:
        # delete uniparticle reaction:
        reaction_id = System.up_reaction_id[type_id]
        if reaction_id != -1:
            number_deleted_i = System.Reactions_Dict[reaction_id].delete_reaction_all(i)
            System.reactions_left -= number_deleted_i
                
        # delete all biparticle and unimolecular reactions the particle is involved in, scheduled for this time point:
        number_bpr = System.bp_reaction_ids[type_id]['n_reactions']
        reaction_ids = System.bp_reaction_ids[type_id]['ids'][0:number_bpr]
        for reaction_id in reaction_ids:
            number_deleted_i = System.Reactions_Dict[reaction_id].delete_reaction_all(i)
            System.reactions_left -= number_deleted_i # Note: for uniparticle reactions number_deleted_i is either 1 or 0

     # Delete molecule reactions      
    else:
        # delete unimolecular reaction:
        reaction_id = System.um_reaction_id[type_id]
        if reaction_id != -1:
            number_deleted_i = System.Reactions_Dict[reaction_id].delete_reaction_all(i)
            System.reactions_left -= number_deleted_i # Note: for unimolecular reactions number_deleted_i is either 1 or 0
            

# @nb.njit
# def delete_reactions(System, i, type_id, educt_is_particle):

#     """Removes all reactions of an educt i scheduled for this time step.
    
#     Parameters
#     ----------
#     System : `object`
#         Instance of System class
#     i : `int64`
#         Index of the particle for which to delete all reactions
    
#     """
    

#     if educt_is_particle:
#         for key in System.Reactions_Dict:
#             if System.Reactions_Dict[key].particle_reaction:
#               number_deleted_i = System.Reactions_Dict[key].delete_reaction_all(i)
#               System.reactions_left -= number_deleted_i
#     else:
#         for key in System.Reactions_Dict:
#             if not System.Reactions_Dict[key].particle_reaction:
#               number_deleted_i = System.Reactions_Dict[key].delete_reaction_all(i)
#               System.reactions_left -= number_deleted_i        
        
#%%

@nb.njit
def convert_particle_type(System, product_id, Particles, i):

    """Converts a particle to a different type.
    
    Parameters
    ----------
    System : `object`
        Instance of System class
    product_id : `int64`
        Id of the product particle type.
    Particles : `object`
        Instance of Particles class
    i : `int64`
        Index of the particle whose type to convert
    
    """
    
    # First, we need to delete this reaction and all other reactions of the educt from the lists:
    delete_reactions(System, i, Particles[i]['type_id'], True)
    
    # Next, we delete the bond with partner particles if a bond exists:
    if Particles[i]['bound'] == True:
        j = Particles[i]['bound_with']
        Particles[j]['bound'] = False
        Particles[j]['bound_with'] = -1 
        Particles[i]['bound'] = False
        Particles[i]['bound_with'] = -1 
        
        System.N_bonds[Particles[i]['type_id']][Particles[j]['type_id']] -= 1
        System.N_bonds[Particles[j]['type_id']][Particles[i]['type_id']] -= 1
        
        # If the binding reaction changed the particle type of j, reverse that:
        educt_type_j = System.particle_types[str(Particles[j]['type'])][0]['bond_educt_id']
        if educt_type_j != -1:
            
            delete_reactions(System, j, Particles[j]['type_id'], True)
            
            Particles[j]['type_id'] = educt_type_j
            Particles[j]['type'] = System.particle_id_to_name[educt_type_j]
            Particles.next_up_reaction(System, j)
        
    #At last, convert the particle type
    Particles[i]['type_id'] = product_id
    Particles[i]['type'] = System.particle_id_to_name[product_id]
    Particles.next_up_reaction(System, i)
    
    
@nb.njit
def convert_molecule_type(System, product_id, product_name, RBs, Particles, i):  
    
    """Converts a molecule to a different type.
    
    Parameters
    ----------
    System : `object`
        Instance of System class
    product_id : `int64`
        Id of the product molecule type.
    product_name : `string`
        Name of the product molecule type.
    RBs : `object`
        Instance of RBs class
    Particles : `object`
        Instance of Particles class
    i : `int64`
        Index of the molecule whose type to convert
    
    """
    
    # Delete all unimolecular reactions of this molecule that are scheduled for this time point:
    # (Note: Any bimolecular reaction is currently handled via particles, not molecules. These have already been deleted above!)
    delete_reactions(System, i, RBs[i]['type_id'], False)
    
    System.Nmol[RBs[i]['type_id']] -= 1
    
    delete_particles(System, RBs, Particles, i)
    
    RBs[i]['topology_N'] = 0
    RBs[i]['type_id'] = product_id
    RBs[i]['name'] = product_name
    RBs[i]['mu_rb'][:,:] = System.molecule_types[product_name].mu_rb
    RBs[i]['mu_tb'][:,:] = System.molecule_types[product_name].mu_tb
    RBs[i]['mu_rb_sqrt'][:,:] = System.molecule_types[product_name].mu_rb_sqrt
    RBs[i]['mu_tb_sqrt'][:,:] = System.molecule_types[product_name].mu_tb_sqrt
    RBs.place_particles(System.molecule_types[product_name].pos, System.molecule_types[product_name].types, System.molecule_types[product_name].radii, System, Particles,i, System.molecule_types[product_name].h_membrane)
    RBs[i]['collision_type'] = System.molecule_types[product_name].collision_type
    RBs.next_um_reaction(System, i)   
    
    System.Nmol[product_id] += 1
    
#%%              

# TODO: There is redundant code in what follows!!!

@nb.njit
def update_reactions(System, Particles, RBs):
    
    """Evaluates the bimolecular and unimolecular reactions. 
    
    Parameters
    ----------
    System : `obj`
        Instance of System class.
    Particles : `obj`
        Instance of Particle class.
    RBs : `obj`
        Instance of RBs (Rigid Bodies) class.
        
    
    Notes
    -----
    
    *Reaction handler*
    
    We use a Gillespie reaction handler scheme, similar to the one used in `ReaDDy <https://readdy.github.io/simulation.html>`_:
        
        1. All reactions events occuring within a time step are gathered in lists (a seperate list for each reaction that has been defined).
        2. A reaction type is picked randomly, weighted by the its total reaction rate times the total number of reactions that are in the list of that type (the total reaction rate is the sum of the individual reaction path rates. Different reaction paths allow for the same educts to undergo, e.g., different fusion reactions or for a single molecule to convert to different products).
        3. A random reaction event is picked from the list.
        4. If the reaction is bimolecular, the reaction probability Eq. :eq:`ReactionProb` is tested against a random number between 0 and 1. 
            a. If the reaction was successfull, we pick a random reaction path, weighted by the path's reaction rates. After the execution of the reaction, it is deleted from the list together with all the other reactions the educts participated in. 
            b. If the reaction was not successful, the reaction is deleted from the list.
        5. If the reaction is unimolecular, it is always successful, since, for unimolecular reactions, we draw the time point of the next reaction event from the corresponding distribution (Gillespie SSA). As such, we pick a random reaction path, weighted by the individual path's reaction rates. After the execution of the reaction, it is deleted from the list together with all the other reactions the educt participated in.
        6. Go back to 2. until no reactions are left.
    
    .. math::
        :label: ReactionProb
        
        p = 1-exp\\Big(-\\sum_{i \\in paths} \\lambda_i \cdot \Delta t \\Big)
    
    *Doi bimolecular reaction scheme*
    
    There exist different methods of how to model bimolecular reactions. PyRID, just as ReaDDy :cite:p:`Hoffmann2019`, uses the Doi scheme :cite:p:`Doi1976`. In this scheme, two molecules can react to a single product moelcule if their distance is less than some reaction radius R. In this case the reaction may occur with a microscopic reaction rate :math:`\\lambda`.
    
    """
    
    # Reset the number of successful and total rections for this timestep:
    for reaction_type_index in System.Reactions_Dict:
        for i in range(len(System.Reactions_Dict[reaction_type_index].paths)):
            System.Reactions_Dict[reaction_type_index].paths[i]['n_success'] = 0
        System.Reactions_Dict[reaction_type_index].n_total = System.Reactions_Dict[reaction_type_index].n
        System.Reactions_Dict[reaction_type_index].n_total_binned += System.Reactions_Dict[reaction_type_index].n
    
    # Create arrays for product position and quaternion:
    Product_Pos = np.empty(3, dtype = np.float64)
    Product_Quat = np.empty(4, dtype = np.float64)
    
    
    reactions_total = 0
    for key in range(len(System.Reactions_Dict)):
        reactions_total += System.Reactions_Dict[key].n
    
    if reactions_total != System.reactions_left:
        print('Discrepancy in number of reactions: ', reactions_total, System.reactions_left, System.current_step)
            
    #Start loop:
    count = 0
    while System.reactions_left > 0:
        
        count+=1
        
        if count%100000 == 0:
            print(count)
            raise ValueError('Stuck in while loop! Not able to evaluate all reactions!')
            
        #============================================
        # Randomly select a reaction:
        # First we randomly select the reaction__type_index based on the number of reactions per reaction type and the respective reaction rate.
        
        cum_weights = np.empty(len(System.Reactions_Dict))
        cum_weights[0] = System.Reactions_Dict[0].rate*System.Reactions_Dict[0].n
        for key in range(1,len(System.Reactions_Dict)):
            cum_weights[key] = cum_weights[key-1]+System.Reactions_Dict[key].rate*System.Reactions_Dict[key].n
    
        reaction_type_index = randu.random_choice(cum_weights = cum_weights)
        
        # Second, we randomly choose a reaction from the reaction list of that reaction type:
        reaction_index = System.Reactions_Dict[reaction_type_index].get_random()
        
        
        #%%
        
        # ----------------------
        # Unimolecular Reactions
        # ----------------------
        
        if System.Reactions_Dict[reaction_type_index].bimol == False:
        
            
            i = System.Reactions_Dict[reaction_type_index].get(reaction_index)['educts_index'][0]
            
            System.Reactions_Dict[reaction_type_index].n_success += 1
            
            # Choose reaction path:
            rates = System.Reactions_Dict[reaction_type_index].paths['rate']
            reaction_path_index = randu.random_choice(weights = rates)
            
            System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['n_success'] += 1
            System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['n_success_binned'] += 1
            
            #%%
            
            # ------------------
            # Paticle Conversion
            # ------------------
            
            if System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['type_id'] == 1: #'conversion'     
                        
                #Convert the particle type
                product_id = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_ids'][0]
    
                convert_particle_type(System, product_id, Particles, i)
                
    
    
            #%%
    
            # -------------------------
            # Particle Release Reaction
            # -------------------------
            
            elif System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['type_id'] == 10:
                    
                i_rb = Particles[i]['rb_id']
                
                radius = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['radius']
                
                educt_comp_id = RBs[i_rb]['compartment']
                educt_loc_id = RBs[i_rb]['loc_id']
                educt_pos = np.copy(RBs[i_rb]['pos'])
                educt_Triangle_id = RBs[i_rb]['triangle_id']
                educt_id = RBs[i_rb]['type_id']
                
                # Convert the educt particle:
                product_id_educt = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_ids'][0]
                convert_particle_type(System, product_id_educt, Particles, i)
                
                
                # Release the product molecule:
                k = 1
                
                product_id = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_ids'][k]
                product_loc = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_loc'][k]
                product_name_k = str(System.molecule_id_to_name[product_id])
                
                
                # if the molecule location type didnt change:
                if product_loc == educt_loc_id:
                    comp_id = educt_comp_id
                    # and is volume:
                    if product_loc == 0:
                        # mol_radius = System.molecule_types[str(product_name_k)].radius
                        dX = distribute_vol.random_direction_sphere(radius)
                        not_absorbed = distribute_vol.trace_direction_vector(dX, educt_pos, Product_Pos, System)
                        
                        if not_absorbed == True:
                        
                            Product_Quat[0],Product_Quat[1],Product_Quat[2],Product_Quat[3] = distribute_vol.random_quaternion_tuple()
                            distribute_vol.add_molecule_single(comp_id, System, RBs, Particles, Product_Pos, Product_Quat, product_name_k)
                    else:
                        
                        dX = distribute_surf.point_in_disc(educt_Triangle_id, radius, System)
                        Position, Quaternion, Triangle_id = distribute_surf.trace_direction_vector(dX, educt_pos, educt_Triangle_id, System)
                        
                        if Triangle_id!=-1:
                            distribute_surf.add_molecule_single(educt_comp_id, System, RBs, Particles, Position, Quaternion, Triangle_id, product_name_k)
                        
                else: # The product and educt type are different.
                    if product_loc == 0: 
                        # The product is a volume molecule, so the educt must have been a surface molecule. Therefore, first, we get the direction of release (inside or outside):
                        product_direction = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_direction'][k]
                        
                        if product_direction == 1:
                            # if product_direction == 0, the molecule is released outside. Therefore, the comp_id has to change to 0:
                            comp_id = 0
                        else:
                            comp_id = educt_comp_id
                            
                        # Next, we distribute the molecule:
                        # To make sure that the ray origin, for which we do the "in same compartment test", is in System, we shift a little along the triangle normal
                        Tri_idx = educt_Triangle_id
                        normal = System.Mesh[Tri_idx]['triangle_coord'][3]
                        origin = educt_pos + product_direction*normal*1e-6
                        
                        dX = distribute_vol.random_direction_sphere(radius)
                        not_absorbed = distribute_vol.trace_direction_vector(dX, origin, Product_Pos, System)
                        
                        if not_absorbed == True:
                        
                            Product_Quat[0],Product_Quat[1],Product_Quat[2],Product_Quat[3] = distribute_vol.random_quaternion_tuple()
                            distribute_vol.add_molecule_single(comp_id, System, RBs, Particles, Product_Pos, Product_Quat, product_name_k)
                        
                            
                    elif product_loc == 1:
                        # The educt was a volume molecule but the product is a surface molecule. This is not supported yet:
                        # print('Volume to surface molecule conversion reaction not yet supported')
                        pass       
    
    
            #%%
    
            # --------------------
            # Molecule Decay
            # --------------------
            elif System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['type_id'] == 2:
                

                delete_molecule(System, RBs, Particles, i)
                
                    
            
            #%%
            
            # --------------------
            # Molecule Conversion
            # --------------------
            
            elif System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['type_id'] == 3:
                
                # Get the product type's id and name:
                product_id = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_ids'][0]
                product_name = str(System.molecule_id_to_name[product_id])
                
                
                #Change the Molecule's type:
                convert_molecule_type(System, product_id, product_name, RBs, Particles, i)
                
            
            #%%
                
            # -------------------
            # Molecule Production
            # -------------------
            
            # New molecules are released from the educt molecule. The educt molecule may change its type in addition.
            
            elif System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['type_id'] == 7:
                
                n_products = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['n_products']
                radius = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['radius']
                
                educt_comp_id = RBs[i]['compartment']
                educt_loc_id = RBs[i]['loc_id']
                educt_pos = np.copy(RBs[i]['pos'])
                educt_Triangle_id = RBs[i]['triangle_id']
                educt_id = RBs[i]['type_id']
                
                # Check if the educt changes its type:
                product_id_educt = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_ids'][0]
                if product_id_educt != educt_id:
                    
                    product_name_educt = str(System.molecule_id_to_name[product_id_educt])
    
                    #Change the Molecule's type:
                    convert_molecule_type(System, product_id_educt, product_name_educt, RBs, Particles, i)            
                
                    # # Update molecule position:
                    # RBs.set_pos(Particles, System, Product_Pos, i) 
                    # # Update molecule orientation:
                    # Product_Quat[0],Product_Quat[1],Product_Quat[2],Product_Quat[3] = distribute_vol.random_quaternion_tuple()
                    # RBs.set_orientation_quat(Particles, Product_Quat, System, i)
                else:
                    # We don't need to convert the educt molecule, but we still need to delete the reactions of i from the list (otherwise done in convert_molecule_type()):
                    # System.Reactions_Dict[reaction_type_index].delete_reaction(reaction_index)
                    # System.reactions_left -= 1
                    delete_reactions(System, i, RBs[i]['type_id'], False)
                    RBs.next_um_reaction(System, i)
                
                # Release the other product molecules:
                for k in range(1,n_products):
    
                    product_id = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_ids'][k]
                    product_loc = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_loc'][k]
                    product_name_k = str(System.molecule_id_to_name[product_id])
                    
                    
                    # if the molecule location type didnt change:
                    if product_loc == educt_loc_id:
                        comp_id = educt_comp_id
                        # and is volume:
                        if product_loc == 0:
                            # mol_radius = System.molecule_types[str(product_name_k)].radius
                            dX = distribute_vol.random_direction_sphere(radius)
                            not_absorbed = distribute_vol.trace_direction_vector(dX, educt_pos, Product_Pos, System)
                            
                            if not_absorbed == True:
                            
                                Product_Quat[0],Product_Quat[1],Product_Quat[2],Product_Quat[3] = distribute_vol.random_quaternion_tuple()
                                distribute_vol.add_molecule_single(comp_id, System, RBs, Particles, Product_Pos, Product_Quat, product_name_k)
                        else:
                            
                            dX = distribute_surf.point_in_disc(educt_Triangle_id, radius, System)
                            Position, Quaternion, Triangle_id = distribute_surf.trace_direction_vector(dX, educt_pos, educt_Triangle_id, System)
                            
                            if Triangle_id!=-1:
                                distribute_surf.add_molecule_single(educt_comp_id, System, RBs, Particles, Position, Quaternion, Triangle_id, product_name_k)
                            
                    else: # The product and educt type are different.
                        if product_loc == 0: 
                            # The product is a volume molecule, so the educt must have been a surface molecule. Therefore, first, we get the direction of release (inside or outside):
                            product_direction = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_direction'][k]
                            
                            if product_direction == 1:
                                # if product_direction == 0, the molecule is released outside. Therefore, the comp_id has to change to 0:
                                comp_id = 0
                            else:
                                comp_id = educt_comp_id
                                
                            # Next, we distribute the molecule:
                            # To make sure that the ray origin, for which we do the "in same compartment test", is in System, we shift a little along the triangle normal
                            Tri_idx = educt_Triangle_id
                            normal = System.Mesh[Tri_idx]['triangle_coord'][3]
                            origin = educt_pos + product_direction*normal*1e-6
                            
                            dX = distribute_vol.random_direction_sphere(radius)
                            not_absorbed = distribute_vol.trace_direction_vector(dX, origin, Product_Pos, System)
                            
                            if not_absorbed == True:
                            
                                Product_Quat[0],Product_Quat[1],Product_Quat[2],Product_Quat[3] = distribute_vol.random_quaternion_tuple()
                                distribute_vol.add_molecule_single(comp_id, System, RBs, Particles, Product_Pos, Product_Quat, product_name_k)
                            
                                
                        elif product_loc == 1:
                            # The educt was a volume molecule but the product is a surface molecule. This is not supported yet:
                            # print('Volume to surface molecule conversion reaction not yet supported')
                            pass
                        
                
            
            #%%
            
            # ----------------
            # Molecule Fission
            # ----------------
            
            
            elif System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['type_id'] == 8:
                
                radius = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['radius']
                
    
                educt_comp_id = RBs[i]['compartment']
                educt_loc_id = RBs[i]['loc_id']
                educt_pos = np.copy(RBs[i]['pos'])
                educt_id = RBs[i]['type_id']
                
                # Get the prouct properties:
                product_id_1 = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_ids'][0]
                product_id_2 = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_ids'][1]
                
                product_name_1 = str(System.molecule_id_to_name[product_id_1])
                product_name_2 = str(System.molecule_id_to_name[product_id_2])
                
                # Depending on the location of educt and products molecule placement is different:
                # if the educt is a volume molecule, the products must also be volume molecules:
                if educt_loc_id == 0:
                    w1 = 0.5
                    w2 = 0.5
                    comp_id = educt_comp_id
                    
                    # draw a random vector:
                    dX = distribute_vol.random_direction_sphere(radius)
                    
                    #=============================================================== 
                    #CONVERT the EDUCT to PRODUCT 1:
                    # Set new position and orientation of product 1:
                    not_absorbed = distribute_vol.trace_direction_vector(dX*w1, educt_pos, Product_Pos, System)
                    if not_absorbed == False:
                        delete_molecule(System, RBs, Particles, i)
                    else:
                        convert_molecule_type(System, product_id_1, product_name_1, RBs, Particles, i)  
                        
                        # Update molecule position:
                        RBs.set_pos(Particles, System, Product_Pos, i) 
                        # Update molecule orientation:
                        Product_Quat[0],Product_Quat[1],Product_Quat[2],Product_Quat[3] = distribute_vol.random_quaternion_tuple()
                        RBs.set_orientation_quat(Particles, Product_Quat, System, i)
    
                    
                    #CREATE a NEW molecule for PRODUCT 2:
                    not_absorbed = distribute_vol.trace_direction_vector(-dX*w2, educt_pos, Product_Pos, System)
                    if not_absorbed:
                        
                        Product_Quat[0],Product_Quat[1],Product_Quat[2],Product_Quat[3] = distribute_vol.random_quaternion_tuple()
                        distribute_vol.add_molecule_single(comp_id, System, RBs, Particles, Product_Pos, Product_Quat, product_name_2)
                    #===============================================================
                    
                elif educt_loc_id == 1:
                    #If the educt is a surface molcule we need to check the product locations
                    educt_Triangle_id = RBs[i]['triangle_id']
                    
                    product_loc_1 = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_loc'][0]
                    product_loc_2 = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_loc'][1]
                    
                    # If either of the products is a volume molecule, the other moelcule will keep its location:
                    if product_loc_1 == 0 or product_loc_2 == 0:
                        
                        if product_loc_1 == 0:
                            product_id = 0
                            product_name = product_name_1
                            product_id_convert = 1
                            product_name_convert = product_name_2
                        elif product_loc_2 == 0:
                            product_id = 1
                            product_name = product_name_2
                            product_id_convert = 0
                            product_name_convert = product_name_1
                        else:
                            raise ValueError('Fission reactions of surface molcules can only contain 1 volume product!')
                            
                        w1 = 1
                        w2 = 0
                        #Direction -> 1: outside, -1: inside compartment
                        product_direction = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_direction'][product_id]
                        
                        if product_direction == 1:
                            comp_id = 0
                        else:
                            comp_id = educt_comp_id
                            
                        # Next, we distribute the molecule:
                        # To make sure that the ray origin, for which we do the "in same compartment test", is in System, we shift a little along the triangle normal
                        Tri_idx = educt_Triangle_id
                        normal = System.Mesh[Tri_idx]['triangle_coord'][3]
                        origin = educt_pos + product_direction*normal*1e-6
        
                        dX = distribute_vol.random_direction_Halfsphere(radius, normal)
                        
                        not_absorbed = distribute_vol.trace_direction_vector(dX, origin, Product_Pos, System)
                        
                        if not_absorbed:
                            
                            Product_Quat[0],Product_Quat[1],Product_Quat[2],Product_Quat[3] = distribute_vol.random_quaternion_tuple()
                            distribute_vol.add_molecule_single(comp_id, System, RBs, Particles, Product_Pos, Product_Quat, product_name)
                            
                        # Convert the other educt, the position stays unaltered:
                        convert_molecule_type(System, product_id_convert, product_name_convert, RBs, Particles, i) 
                        
                    else:
                        # Both products are surface molecules.
                        
                        w1 = 0.5
                        w2 = 0.5
                        comp_id = educt_comp_id
                    
                        dX = distribute_surf.point_in_disc(educt_Triangle_id, radius, System)
                        
                        #=============================================================== 
                        #CONVERT the EDUCT to PRODUCT 1:
                        # Set new position and orientation of product 1:
                        Position, Quaternion, Triangle_id = distribute_surf.trace_direction_vector(w1*dX, educt_pos, educt_Triangle_id, System)
                        
                        if Triangle_id == -1:
                            delete_molecule(System, RBs, Particles, i)
                        else:
                            convert_molecule_type(System, product_id_1, product_name_1, RBs, Particles, i)  
                            
                            RBs.set_triangle(Triangle_id, i)
                            RBs.set_pos(Particles, System, Position, i) 
                            # distribute_surf.random_orientation_in_plane(System.Mesh[Triangle_id]['triangle_coord'][3], Quaternion) # Is part of distribute_surf.trace_direction_vector()
                            RBs.set_orientation_quat(Particles, Quaternion, System, i)
                        
                        #CREATE a NEW molecule for PRODUCT 2:
                        Position, Quaternion, Triangle_id = distribute_surf.trace_direction_vector(-w2*dX, educt_pos, educt_Triangle_id, System)
                        if Triangle_id != -1:
                            
                            distribute_surf.add_molecule_single(comp_id, System, RBs, Particles, Position, Quaternion, Triangle_id, product_name_2)
        
        
        
        #%%
        
        # ---------------------
        # Bimolecular Reactions
        # ---------------------
        
        else:
            
            #Next, we test whether the reaction is successfuLL:
            if np.random.rand() < 1-np.exp(-System.Reactions_Dict[reaction_type_index].rate*System.dt):            
                
                i = System.Reactions_Dict[reaction_type_index].get(reaction_index)['educts_index'][0]
                j = System.Reactions_Dict[reaction_type_index].get(reaction_index)['educts_index'][1]
                
                System.Reactions_Dict[reaction_type_index].n_success += 1
                
                # Choose reaction path:
                rates = System.Reactions_Dict[reaction_type_index].paths['rate']
                reaction_path_index = randu.random_choice(weights = rates)
                
                System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['n_success'] += 1
                System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['n_success_binned'] += 1
                
                #The reaction was succefull, so we need to execute the reaction:
                
                #%%
            
                
                # ----------------
                # Particle Binding
                # ----------------
                
                if System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['type_id'] == 0: #'bind'

                    # If the binding reaction is linked to a particle type change, convert the particle types:
                    if System.Reactions_Dict[reaction_type_index].paths[0]['n_products'] > 0:
                        
                        product_id_i = System.Reactions_Dict[reaction_type_index].paths[0]['products_ids'][0]
                        product_id_j = System.Reactions_Dict[reaction_type_index].paths[0]['products_ids'][1]
                        
                        # Check if i is the first or the second educt as defined in the reaction:
                        if Particles[i]['type_id'] == System.Reactions_Dict[reaction_type_index].educts[0]:
    
                            convert_particle_type(System, product_id_i, Particles, i)
                            convert_particle_type(System, product_id_j, Particles, j)
                            
                        else:
                        
                            convert_particle_type(System, product_id_j, Particles, i)
                            convert_particle_type(System, product_id_i, Particles, j)
                            
                    else:
                        # We now need to delete this reaction and all other reactions of the educts from the lists (If the educt types changed, deletio of the reaction is handled by convert_particle_type()):
                        delete_reactions(System, i, Particles[i]['type_id'], True)
                        delete_reactions(System, j, Particles[j]['type_id'], True)
                            
                    #Increase the number of bonds:
                    System.N_bonds[Particles[i]['type_id']][Particles[j]['type_id']] += 1
                    System.N_bonds[Particles[j]['type_id']][Particles[i]['type_id']] += 1
                    
                    #Update the particle's bound state variables:
                    Particles[i]['bound'] = True
                    Particles[j]['bound'] = True
                    Particles[i]['bound_with'] = j
                    Particles[j]['bound_with'] = i
                    
                    #update force:
                    update_bond_force(i,j,Particles, RBs, System)
        
                #%%
                
                # ---------------------------
                # Particle Enzymatic Reaction
                # ---------------------------
                
                elif System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['type_id'] == 4:
                    
    
                    #-----------------------------------
                    #Convert the particle type
                    product_id = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_ids'][0]
                    
                    convert_particle_type(System, product_id, Particles, i)
                    
        
                #%%
        
                # ----------------------------
                # Particle Absorption Reaction
                # ----------------------------
                
                elif System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['type_id'] == 9:
                    
                    #-----------------------------------
                    #Convert the particle type
                    product_id = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_ids'][0]
                    
                    convert_particle_type(System, product_id, Particles, i)
                    
                    #-----------------------------------
                    # delete molecule belonging to particle j:
                    j_rb = Particles[j]['rb_id']
                    delete_molecule(System, RBs, Particles, j_rb)
                        

                #%%
                
                # --------------------
                # Molecule Fusion
                # --------------------
                
                elif System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['type_id'] == 5:
                        
                    # get the corresponding rb_ids:
                    i_rb = Particles[i]['rb_id']
                    j_rb = Particles[j]['rb_id']
                    
                    educt_1_loc_id = RBs[i_rb]['loc_id']
                    educt_2_loc_id = RBs[j_rb]['loc_id']
                        
                    surface_intersection = False
                    if System.mesh == True and (educt_1_loc_id==0 or educt_2_loc_id==0):
                        #The reaction will only be executed if there is no mesh surface element in between the direct path connecting the 2 educt molecules if at least one of the educts is a volume molecule:
                        pos_i = RBs[i_rb]['pos']
                        dX_ij = pos_i-RBs[j_rb]['pos']
                        surface_intersection = any_ray_mesh_intersection_test(pos_i, dX_ij, System)
                    
                    #The reaction was succefull, so we need to execute the reaction:
                    if surface_intersection == False:
                        
                        #Check if location types are different:
                        if educt_1_loc_id != educt_2_loc_id:
                            # For a reaction between a volume and a surface molecule, the product will always be a surface molcule. If mol 2 is a surface molecule, swap indices:
                            if educt_2_loc_id == 1:
                                i_rb_copy = i_rb
                                i_rb = j_rb
                                j_rb = i_rb_copy
                        
                        educt_2_pos = np.copy(RBs[j_rb]['pos'])
                        
                        #-----------------------------------
                        # DELETE second MOLECULE:
                        #-----------------------------------
                        
                        delete_molecule(System, RBs, Particles, j_rb)
                        
                        
                        #-----------------------------------
                        # CONVERT first MOLECULE to PRODUCT:
                        #-----------------------------------
                        
                        product_id = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_ids'][0]
                        product_name = str(System.molecule_id_to_name[product_id])
                        
                        placement_factor = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['placement_factor']
                        
                        convert_molecule_type(System, product_id, product_name, RBs, Particles, i_rb)
                        
                        # Position the product molecule
                        if educt_1_loc_id == educt_2_loc_id: 
                            
                            dX = (educt_2_pos - RBs[i_rb]['pos'])
                            if System.boundary_condition_id == 0: # (periodic)
                                for dim in range(3):
                                    if abs(dX[dim])>System.box_lengths[dim]/2:
                                        dX[dim] = np.sign(dX[dim])*System.box_lengths[dim] - dX[dim]
                              
                            RBs[i_rb]['dX'][:] = dX*placement_factor
                            if educt_1_loc_id == 0: # Both volume molecules
                                RBs.update_particle_pos(Particles, System, i_rb)
                            elif educt_1_loc_id == 1: # Both surface molecules
                                # Travel along surface (The result will of course not be exactly in
                                # the middle between the two molecule, since dX is calculated as euclidian distance, but for a small surface curvature this will be negligable.)
                                # TODO: Rotate dX into triangle plane!?
                                RBs.update_particle_pos_2D(System, Particles, i_rb)
                    
                    
                    else:
                        
                        System.Reactions_Dict[reaction_type_index].n_success -= 1
                        System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['n_success'] -= 1
                        System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['n_success_binned'] -= 1
                        
                        System.Reactions_Dict[reaction_type_index].delete_reaction(reaction_index)
                        System.reactions_left -= 1                
                        
                        
                        
                #%%
                
                # --------------------
                # Molecule Enzymatic
                # --------------------
                
                elif System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['type_id'] == 6:
                        
                    # get the corresponding rb_ids:
                    i_rb = Particles[i]['rb_id']
                    j_rb = Particles[j]['rb_id']
                    
                    #-----------------------------------
                    # CONVERT first MOLECULE to PRODUCT:
                    #-----------------------------------
    
                    product_id = System.Reactions_Dict[reaction_type_index].paths[reaction_path_index]['products_ids'][0]
                    product_name = str(System.molecule_id_to_name[product_id])
                    
                    convert_molecule_type(System, product_id, product_name, RBs, Particles, i_rb)
                                    
                        
                        
#%%

            else:
                
                # If the reaction was not succesfull, we need to delete this reaction from the list but keep all other reactions of the educts:
                
                System.Reactions_Dict[reaction_type_index].delete_reaction(reaction_index)
                
                System.reactions_left -= 1  