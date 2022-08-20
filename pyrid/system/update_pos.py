# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numba as nb
                    
#%%
 
# @nb.njit
# def update_rb_compartments(RBs, Particles, System):
    
#     """Loops through all volume and surface rigid bead molecules and updates their total force and torque, propagates the rotation quaternion and translation vectors. Adds any unimolecular reaction to reactions list if the next reaction is scheduled for the current time step.
    
#     Parameters
#     ----------
#     RBs : object
#         Rigid body class instance
#     Particles : object
#         Particles class instance
#     Systems : object
#         System class instance
    
#     """
    
#     reactions_total = 0
#     for key in range(len(System.Reactions_Dict)):
#         reactions_total += System.Reactions_Dict[key].n
    
#     if reactions_total != System.reactions_left:
#         print('Pos_0: Discrepancy in number of reactions: ', reactions_total, System.reactions_left, System.current_step)
        
#     for i0 in range(RBs.occupied.n):
#         i = RBs.occupied[i0] 
        
#         RBs.update_force_torque(Particles,i)
        
#         if RBs[i]['loc_id'] == 0:
            
#             RBs.update_dq(System,i)
#             RBs.update_dX(System,i)
            
#             RBs.update_orientation_quat(i)
            
#             RBs.update_particle_pos(Particles, System,i)
            
#         if RBs[i]['loc_id'] == 1:
            
#             RBs.update_dq(System,i)
#             RBs.update_dX(System,i)
            
#             RBs.update_orientation_quat(i)
            
#             RBs.update_particle_pos_2D(System, Particles,i)
            
            
#         # Unimolecular reaction:

#         if RBs[i]['next_transition']<=System.current_step*System.dt:
        
#             mtype_idx_i = RBs[i]['type_id']

#             reaction_id = System.um_reaction_id[mtype_idx_i]
#             System.reactions_left += 1
#             System.Reactions_Dict[reaction_id].append_reaction(i)

#     reactions_total = 0
#     for key in range(len(System.Reactions_Dict)):
#         reactions_total += System.Reactions_Dict[key].n
    
#     if reactions_total != System.reactions_left:
#         print('Pos_1: Discrepancy in number of reactions: ', reactions_total, System.reactions_left, System.current_step)
        
    
# @nb.njit
# def update_rb(RBs, Particles, System):
    
#     """Loops through all volume rigid bead molecules and updates their total force and torque, propagates the rotation quaternion and translation vectors. Adds any unimolecular reaction to reactions list if the next reaction is scheduled for the current time step.
    
#     Parameters
#     ----------
#     RBs : object
#         Rigid body class instance
#     Particles : object
#         Particles class instance
#     Systems : object
#         System class instance
    
#     """
    
#     for i0 in range(RBs.occupied.n):
#         i = RBs.occupied[i0] 
        
#         RBs.update_force_torque(Particles,i)
        
#         RBs.update_dq(System,i)
#         RBs.update_dX(System,i)
        
#         RBs.update_orientation_quat(i)
        
#         RBs.update_particle_pos(Particles, System,i)
        
        
#         # Unimolecular reaction:

#         if RBs[i]['next_transition']<=System.current_step*System.dt:
        
#             mtype_idx_i = RBs[i]['type_id']

#             reaction_id = System.um_reaction_id[mtype_idx_i]
#             System.reactions_left += 1
#             System.Reactions_Dict[reaction_id].append_reaction(i)
            
            
            
#%%

@nb.njit
def update_rb_compartments(RBs, Particles, System):
    
    """Loops through all volume and surface rigid bead molecules and updates their total force and torque, propagates the rotation quaternion and translation vectors. Adds any unimolecular reaction to reactions list if the next reaction is scheduled for the current time step.
    
    Parameters
    ----------
    RBs : object
        Rigid body class instance
    Particles : object
        Particles class instance
    Systems : object
        System class instance
    
    """
    
    reactions_total = 0
    for key in range(len(System.Reactions_Dict)):
        reactions_total += System.Reactions_Dict[key].n
    
    if reactions_total != System.reactions_left:
        print('Pos_0: Discrepancy in number of reactions: ', reactions_total, System.reactions_left, System.current_step)
        
    for i0 in range(RBs.occupied.n):
        i = RBs.occupied[i0] 
        
        if RBs[i]['loc_id'] == 0:
            
            if RBs[i]['topology_N']>1:
                RBs.update_force_torque(Particles,i)
                RBs.update_dq(System,i)
                RBs.update_dX(System,i)
                RBs.update_orientation_quat(i)
            else:
                RBs.update_force(Particles, i)
                RBs.update_dX(System,i)
            
            RBs.update_particle_pos(Particles, System,i)
            
        if RBs[i]['loc_id'] == 1:
            
            if RBs[i]['topology_N']>1:
                RBs.update_force_torque(Particles,i)
                RBs.update_dq(System,i)
                RBs.update_dX(System,i)
                # RBs.update_orientation_quat(i) -> is updated after surface ray marching in RBs.update_particle_pos_2D()
            else:
                RBs.update_force(Particles, i)
                RBs.update_dX(System,i)
            
            RBs.update_particle_pos_2D(System, Particles,i)
            
            
        # Unimolecular reaction:

        if RBs[i]['next_transition']<=System.current_step*System.dt:
        
            mtype_idx_i = RBs[i]['type_id']

            reaction_id = System.um_reaction_id[mtype_idx_i]
            System.reactions_left += 1
            System.Reactions_Dict[reaction_id].append_reaction(i)

    reactions_total = 0
    for key in range(len(System.Reactions_Dict)):
        reactions_total += System.Reactions_Dict[key].n
    
    if reactions_total != System.reactions_left:
        print('Pos_1: Discrepancy in number of reactions: ', reactions_total, System.reactions_left, System.current_step)
        
    
@nb.njit
def update_rb(RBs, Particles, System):
    
    """Loops through all volume rigid bead molecules and updates their total force and torque, propagates the rotation quaternion and translation vectors. Adds any unimolecular reaction to reactions list if the next reaction is scheduled for the current time step.
    
    Parameters
    ----------
    RBs : object
        Rigid body class instance
    Particles : object
        Particles class instance
    Systems : object
        System class instance
    
    """
    
    for i0 in range(RBs.occupied.n):
        i = RBs.occupied[i0] 
        
        if RBs[i]['topology_N']>1:
            RBs.update_force_torque(Particles,i)
            RBs.update_dq(System,i)
            RBs.update_dX(System,i)
            RBs.update_orientation_quat(i)
        else:
            RBs.update_force(Particles, i)
            RBs.update_dX(System,i)
        
        RBs.update_particle_pos(Particles, System,i)
        
        
        # Unimolecular reaction:

        if RBs[i]['next_transition']<=System.current_step*System.dt:
        
            mtype_idx_i = RBs[i]['type_id']

            reaction_id = System.um_reaction_id[mtype_idx_i]
            System.reactions_left += 1
            System.Reactions_Dict[reaction_id].append_reaction(i)
            
            