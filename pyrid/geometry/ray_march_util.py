# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numpy as np
import numba as nb
from ..math import transform_util as trf
from ..geometry.intersections_util import ray_mesh_intersection_test, point_in_triangle_barycentric, edge_intersection_barycentric 
from ..geometry.mesh_util import point_triangle_distance

#%%

@nb.njit
def update_to_nearest_triangle(pos, quaternion, triangle_id, System):
    
    """Updates a position vector to the plane of the nearest triangle.
    
    Parameters
    ----------
    pos : `float64[3]`
        Position vector
    quaternion : `float64[4]`
        Quaternion vector
    triangle_id : `int64`
        Triangle id
    System : `object`
        Instance of System class   
    
    
    Returns
    -------
    `int64`
        Triangle id of the closest triangle.
    
    """
    
    a_n = np.empty(3, dtype = np.float64)
    pos_tri = np.empty(3, dtype = np.float64)
    new_pos = np.empty(3, dtype = np.float64)
    
    cx = int((pos[0]-System.origin[0]) / System.cell_length_per_dim[0])
    cy = int((pos[1]-System.origin[1]) / System.cell_length_per_dim[1])
    cz = int((pos[2]-System.origin[2]) / System.cell_length_per_dim[2])

    # Determine cell in 3D volume for i-th particle
    cell = cx + cy * System.cells_per_dim[0] + cz * System.cells_per_dim[0] * System.cells_per_dim[1]
    
    triangles_list = System.CellList.get_triangles(cell)
    
    dist_min = 1e6
    new_Tri_idx = -1
    for Tri_idx in triangles_list:
        
        triangle = System.Mesh[Tri_idx]['triangles']
    
        p0 = System.vertices[triangle[0]]
        p1 = System.vertices[triangle[1]]
        p2 = System.vertices[triangle[2]]
        
        dist, region = point_triangle_distance(p0,p1,p2,pos, pos_tri)
        if dist<dist_min:
            dist_min = dist
            new_pos[:] = pos_tri
            new_Tri_idx = Tri_idx
        
    if new_Tri_idx != -1:
        
        #Rotation into the plane of the triangle:
        sin_phi, cos_phi = trf.quaternion_plane_to_plane(quaternion, System.Mesh[triangle_id]['triangle_coord'][3], System.Mesh[new_Tri_idx]['triangle_coord'][3], a_n)
        
        pos[:] = new_pos
        
    return new_Tri_idx
    
    
#%%

@nb.njit
def nearest_triangle(pos, System):

    """Updates a position vector to the plane of the nearest triangle.
    
    Parameters
    ----------
    pos : `float64[3]`
        position vector
    System : `object`
        Instance of System class   
    
    
    Returns
    -------
    `int64`
        Triangle id of the closest triangle.
    
    """
    
    pos_tri = np.empty(3, dtype = np.float64)
    
    cx = int((pos[0]-System.origin[0]) / System.cell_length_per_dim[0])
    cy = int((pos[1]-System.origin[1]) / System.cell_length_per_dim[1])
    cz = int((pos[2]-System.origin[2]) / System.cell_length_per_dim[2])

    cell = cx + cy * System.cells_per_dim[0] + cz * System.cells_per_dim[0] * System.cells_per_dim[1]
    
    triangles_list = System.CellList.get_triangles(cell)
    
    dist_min = 1e6
    new_pos = np.empty(3)
    new_Tri_idx = -1
    for Tri_idx in triangles_list:
        
        triangle = System.Mesh[Tri_idx]['triangles']
    
        p0 = System.vertices[triangle[0]]
        p1 = System.vertices[triangle[1]]
        p2 = System.vertices[triangle[2]]
        
        dist, region = point_triangle_distance(p0,p1,p2,pos, pos_tri)
        if dist<dist_min:
            dist_min = dist
            new_pos[:] = pos_tri
            new_Tri_idx = Tri_idx
    
    pos[:] = new_pos
    
    return new_Tri_idx

#%%


@nb.njit
def ray_march_volume(pos, dX, System):
    
    """Ray marches along a direction vector and tests and resolves triangle collisions. The algorithm is based on :cite:p:`Amanatides87` : Amanatides et al. 1987 "A Fast Voxel Traversal Algorithm for Ray Tracing" Also see :cite:p:`Ericson2004` : Ericson "Real-Time Collision Detection", chapter 7.4.2 and 7.7
    
    Parameters
    ----------
    pos : `float64[3]`
        Position vector
    dX : `float64[3]`
        Direction vector
    System : `object`
        Instance of System class   
    
    See Also
    --------
    :func:`~pyrid.geometry.intersection_until.ray_mesh_intersection_test`
    
    Returns
    -------
    `boolean`
        True if ray marching was successfull, False if absorptive boundary has been hit.
    
    
    """
    
    
    current_triangle = -1
    
    poi = np.empty(3, dtype = np.float64) # point_of_intersection
    dX_refl = np.empty(3)
    
    passed_all = False
    
    
    Max = 100000
    cx_max = System.cells_per_dim[0]
    cy_max = System.cells_per_dim[1]
    cz_max = System.cells_per_dim[2]
    
    count = 0
    while passed_all == False:
        count += 1
        
        # Need to increase time_stamp such that for the follow up ray, no triangles are excluded from the intersection test.
        # Problem: We may end up intersecting the triangle we are currently located on!
        # Solution: Save the current triangle and exclude it from the intersection test!
        System.time_stamp += 1
        
        t_min = 1e10
        triangle_of_intersection = -1
        intersection = False
        
        crossed_border = False
        correct_intersection_found =  False
        
        
        # INITIALIZATION PHASE:
            
        # identifying the voxel in which the ray System.origin is found:
        cx = np.floor((pos[0]-System.origin[0])/System.cell_length_per_dim[0])
        cy = np.floor((pos[1]-System.origin[1])/System.cell_length_per_dim[1])
        cz = np.floor((pos[2]-System.origin[2])/System.cell_length_per_dim[2])

        
        cx_end = np.floor(((pos[0]+dX[0])-System.origin[0])/System.cell_length_per_dim[0])
        cy_end = np.floor(((pos[1]+dX[1])-System.origin[1])/System.cell_length_per_dim[1])
        cz_end = np.floor(((pos[2]+dX[2])-System.origin[2])/System.cell_length_per_dim[2])
        cx_end -= cx_end//cx_max*cx_max
        cy_end -= cy_end//cy_max*cy_max
        cz_end -= cz_end//cz_max*cz_max
        
        Next = not (cx_end == cx and cy_end == cy and cz_end == cz)
        
        if Next:

            cx_end_init = cx_end
            cy_end_init = cy_end
            cz_end_init = cz_end
            
            cx_start =  cx
            cy_start =  cy
            cz_start =  cz

            # the variables stepX and stepY are initialized to either 1 or -1 indicating whether X and Y are incremented or decremented
            stepX = np.sign(dX[0])
            stepY = np.sign(dX[1])
            stepZ = np.sign(dX[2])
        
            # Next, we determine the value of t at which the ray crosses the ﬁrst vertical voxel boundary and store it in variable tMaxX.
            next_voxel_boundary_x = (cx+stepX)*System.cell_length_per_dim[0] if dX[0]>=0 else cx*System.cell_length_per_dim[0]
            next_voxel_boundary_y = (cy+stepY)*System.cell_length_per_dim[1] if dX[1]>=0 else cy*System.cell_length_per_dim[1]
            next_voxel_boundary_z = (cz+stepZ)*System.cell_length_per_dim[2] if dX[2]>=0 else cz*System.cell_length_per_dim[2]
            
            tMaxX = (next_voxel_boundary_x - (pos[0]-System.origin[0]))/dX[0] if (dX[0]!=0) else Max
            tMaxY = (next_voxel_boundary_y - (pos[1]-System.origin[1]))/dX[1] if (dX[1]!=0) else Max
            tMaxZ = (next_voxel_boundary_z - (pos[2]-System.origin[2]))/dX[2] if (dX[2]!=0) else Max
        
            # Finally, we compute tDeltaX and tDeltaY. TDeltaX indicates how far along the ray we must move (in units of t) for the horizontal component of such a movement to equal the width of a voxel.
            tDeltaX = System.cell_length_per_dim[0]/dX[0]*stepX if (dX[0]!=0) else Max
            tDeltaY = System.cell_length_per_dim[1]/dX[1]*stepY if (dX[1]!=0) else Max
            tDeltaZ = System.cell_length_per_dim[2]/dX[2]*stepZ if (dX[2]!=0) else Max
        
        
        #INCREMENTAL PHASE:
        
        # !!! The algorithm is a bit more complex than usually necessary, because we want to allow for meshes that extend the simulation box!
        
        while(Next):
            
            cell = int(cx + cy * System.cells_per_dim[0] + cz * System.cells_per_dim[0] * System.cells_per_dim[1])
            
            intersection, t_min, triangle_of_intersection = ray_mesh_intersection_test(pos, dX, System, cell, t_min, triangle_of_intersection, current_triangle)
            
            # If there is an intersection, check, whether this intersection occured in the current cell:
            # if intersection: # We do not need to test t_min<=1.0, because this is already done in ray_triangle_intersection()!
            if (tMaxX < tMaxY):
              if (tMaxX < tMaxZ): 
                if tMaxX-tDeltaX<t_min<=tMaxX: # intersection occured in the current cell!
                    correct_intersection_found = True
                else:
                    cx += stepX
                    tMaxX += tDeltaX
                    if cx>=cx_max:
                        
                        boundary_x = System.box_lengths[0]/2
                        tX = (boundary_x - pos[0])/dX[0]
                        pos[:] += tX*dX
                        
                        if System.boundary_condition_id == 1:
                            dX[:] -= tX*dX
                            dX[0] *= -1
                        elif System.boundary_condition_id == 0:
                            dX[:] -= tX*dX
                            pos[0] -= System.box_lengths[0]
                        
                        crossed_border = True

                    elif cx<0:
                        
                        boundary_x = -System.box_lengths[0]/2
                        tX = (boundary_x - pos[0])/dX[0]
                        pos[:] += tX*dX
                        
                        if System.boundary_condition_id == 1:
                            dX[:] -= tX*dX
                            dX[0] *= -1
                        elif System.boundary_condition_id == 0:
                            dX[:] -= tX*dX
                            pos[0] += System.box_lengths[0]
                            
                        crossed_border = True
                        
                    else:
                        # So, in which cell did the intersection occur (only allow for  cells inside the box)?
                        cx_end = cx_start+stepX*np.floor(t_min/tDeltaX)
                        if 0 <= cx_end <= cx_max:
                            cy_end = cy_start+stepY*np.floor(t_min/tDeltaY)
                            if 0 <= cy_end <= cy_max:
                                cz_end = cz_start+stepZ*np.floor(t_min/tDeltaZ)
                                if not 0 <= cz_end <= cz_max:
                                    cy_end = cy_end_init
                                    cx_end = cx_end_init
                                    cz_end = cz_end_init
                            else:
                                cy_end = cy_end_init
                                cx_end = cx_end_init
                        else:
                            cx_end = cx_end_init
                            
                    
              else:
                if tMaxZ-tDeltaZ<t_min<=tMaxZ:
                    correct_intersection_found = True
                else:
                    cz += stepZ
                    tMaxZ += tDeltaZ
                    if cz>=cz_max:

                        boundary_z = System.box_lengths[2]/2
                        tZ = (boundary_z - pos[2])/dX[2]
                        pos[:] += tZ*dX
                        
                        if System.boundary_condition_id == 1:
                            dX[:] -= tZ*dX
                            dX[2] *= -1
                        elif System.boundary_condition_id == 0:
                            dX[:] -= tZ*dX
                            pos[2] -= System.box_lengths[2]
                            
                        crossed_border = True
                        
                    elif cz<0:
                        
                        boundary_z = -System.box_lengths[2]/2
                        tZ = (boundary_z - pos[2])/dX[2]
                        pos[:] += tZ*dX
                        
                        if System.boundary_condition_id == 1:
                            dX[:] -= tZ*dX
                            dX[2] *= -1
                        elif System.boundary_condition_id == 0:
                            dX[:] -= tZ*dX
                            pos[2] += System.box_lengths[2]
                            
                        crossed_border = True
                        
                    else:
                        cx_end = cx_start+stepX*np.floor(t_min/tDeltaX)
                        if 0 <= cx_end <= cx_max:
                            cy_end = cy_start+stepY*np.floor(t_min/tDeltaY)
                            if 0 <= cy_end <= cy_max:
                                cz_end = cz_start+stepZ*np.floor(t_min/tDeltaZ)
                                if not 0 <= cz_end <= cz_max:
                                    cy_end = cy_end_init
                                    cx_end = cx_end_init
                                    cz_end = cz_end_init
                            else:
                                cy_end = cy_end_init
                                cx_end = cx_end_init
                        else:
                            cx_end = cx_end_init
             
            else:
              if (tMaxY < tMaxZ):
                if tMaxY-tDeltaY<t_min<=tMaxY:
                    correct_intersection_found = True
                else:
                    cy += stepY
                    tMaxY += tDeltaY
                    if cy>=cy_max:
                        
                        boundary_y = System.box_lengths[1]/2
                        tY = (boundary_y - pos[1])/dX[1]
                        pos[:] += tY*dX
                        
                        if System.boundary_condition_id == 1:
                            dX[:] -= tY*dX
                            dX[1] *= -1
                        elif System.boundary_condition_id == 0:
                            dX[:] -= tY*dX
                            pos[1] -= System.box_lengths[1]
                            
                        crossed_border = True
                        
                    elif cy<0:
                        cy+=cy_max
                        
                        boundary_y = -System.box_lengths[1]/2
                        tY = (boundary_y - pos[1])/dX[1]
                        pos[:] += tY*dX
                        
                        if System.boundary_condition_id == 1:
                            dX[:] -= tY*dX
                            dX[1] *= -1
                        elif System.boundary_condition_id == 0:
                            dX[:] -= tY*dX
                            pos[1] += System.box_lengths[1]
                        
                        crossed_border = True
                        
                    else:
                        cx_end = cx_start+stepX*np.floor(t_min/tDeltaX)
                        if 0 <= cx_end <= cx_max:
                            cy_end = cy_start+stepY*np.floor(t_min/tDeltaY)
                            if 0 <= cy_end <= cy_max:
                                cz_end = cz_start+stepZ*np.floor(t_min/tDeltaZ)
                                if not 0 <= cz_end <= cz_max:
                                    cy_end = cy_end_init
                                    cx_end = cx_end_init
                                    cz_end = cz_end_init
                            else:
                                cy_end = cy_end_init
                                cx_end = cx_end_init
                        else:
                            cx_end = cx_end_init
                    
              else:   
                if tMaxZ-tDeltaZ<t_min<=tMaxZ:
                    correct_intersection_found = True
                else:
                    cz += stepZ
                    tMaxZ += tDeltaZ
                    if cz>=cz_max:
                        
                        boundary_z = System.box_lengths[2]/2
                        tZ = (boundary_z - pos[2])/dX[2]
                        pos[:] += tZ*dX
                        
                        if System.boundary_condition_id == 1:
                            dX[:] -= tZ*dX
                            dX[2] *= -1
                        elif System.boundary_condition_id == 0:
                            dX[:] -= tZ*dX
                            pos[2] -= System.box_lengths[2]
                            
                        crossed_border = True
                        
                    elif cz<0:
                        
                        boundary_z = -System.box_lengths[2]/2
                        tZ = (boundary_z - pos[2])/dX[2]
                        pos[:] += tZ*dX
                        
                        if System.boundary_condition_id == 1:
                            dX[:] -= tZ*dX
                            dX[2] *= -1
                        elif System.boundary_condition_id == 0:
                            dX[:] -= tZ*dX
                            pos[2] += System.box_lengths[2]
                        
                        crossed_border = True
                        
                    else:
                        cx_end = cx_start+stepX*np.floor(t_min/tDeltaX)
                        if 0 <= cx_end <= cx_max:
                            cy_end = cy_start+stepY*np.floor(t_min/tDeltaY)
                            if 0 <= cy_end <= cy_max:
                                cz_end = cz_start+stepZ*np.floor(t_min/tDeltaZ)
                                if not 0 <= cz_end <= cz_max:
                                    cy_end = cy_end_init
                                    cx_end = cx_end_init
                                    cz_end = cz_end_init
                            else:
                                cy_end = cy_end_init
                                cx_end = cx_end_init
                        else:
                            cx_end = cx_end_init
                        
            
            
            if correct_intersection_found:

                poi[0] = pos[0] + t_min*dX[0]
                poi[1] = pos[1] + t_min*dX[1]
                poi[2] = pos[2] + t_min*dX[2]
                
                trf.collision_response(pos+dX-poi, System.Mesh[triangle_of_intersection]['triangle_coord'][3], dX_refl)
                
                pos[0] = poi[0]
                pos[1] = poi[1]
                pos[2] = poi[2]
                
                
                dX[0] = dX_refl[0]
                dX[1] = dX_refl[1]
                dX[2] = dX_refl[2]
                
                current_triangle = triangle_of_intersection
                
                Next = False
            
            elif crossed_border:
                
                if System.boundary_condition_id == 2:

                    return False
                else:
                    Next = False
                
            else:
                
                # Loop until we reach end point
                Next = not (cx_end == cx and cy_end == cy and cz_end == cz)   
                # if count>30:
                #     Next = False
                
        # After we exit the while loop, we still need to check the las cell for any collisions if correct_intersection_found is still False:
        if correct_intersection_found == False and crossed_border == False:
            cell = int(cx + cy * System.cells_per_dim[0] + cz * System.cells_per_dim[0] * System.cells_per_dim[1])
            
            intersection, t_min, triangle_of_intersection = ray_mesh_intersection_test(pos, dX, System, cell, t_min, triangle_of_intersection, current_triangle)                
            

            # if correct_intersection_found:
            if t_min<=1: # There might have been an intersection in a past cell, therefore, here we check whether t_min <= 1 and not just whether intersection == True.
                correct_intersection_found = True

                poi[0] = pos[0] + t_min*dX[0]
                poi[1] = pos[1] + t_min*dX[1]
                poi[2] = pos[2] + t_min*dX[2]
                
                trf.collision_response(pos+dX-poi, System.Mesh[triangle_of_intersection]['triangle_coord'][3], dX_refl)
                
                pos[0] = poi[0]
                pos[1] = poi[1]
                pos[2] = poi[2]
                    
                dX[0] = dX_refl[0]
                dX[1] = dX_refl[1]
                dX[2] = dX_refl[2]
            
                current_triangle = triangle_of_intersection
                
        if correct_intersection_found == False and crossed_border == False:
            passed_all = True
            
        if count>100:
            print('canceled while loop', pos, dX)
            passed_all = True
            
    
    pos[0] += dX[0]
    pos[1] += dX[1]
    pos[2] += dX[2]    
    
    return True


#%%


@nb.njit
def ray_march_surface(pos, quaternion, dX, triangle_id, System, update_quat=False):
    
    """Ray marches a direction vector across the surface of a triangulated mesh.
    
    Parameters
    ----------
    pos : `float64[3]`
        Position vector
    quaternion : `float64[4]`
        Rotation quaternion
    dX : `float64[3]`
        Direction vector
    triangle_id : `int64`
        Triangle id
    System : `object`
        Instance of System class   
    
    
    Returns
    -------
    `int64`
        Index of the triangle the direction vector ended up on.
    
    """
    
    # pos_init = np.copy(pos)
    # dX_init = np.copy(dX)
    # quaternion_init = np.copy(quaternion)
    # tri_id_init = triangle_id
            
    crossed_border = False
    
    a_n = np.empty(3)
    
    face = triangle_id
    
    p0 = System.vertices[System.Mesh[face]['triangles'][0]]
    p1 = System.vertices[System.Mesh[face]['triangles'][1]]
    p2 = System.vertices[System.Mesh[face]['triangles'][2]]
        
    u,v = trf.barycentric_coord(pos, p0, p1, p2, System.Mesh[face]['barycentric_params'])
    du,dv = trf.barycentric_direction(dX, p0, p1, p2, System.Mesh[face]['barycentric_params'])
    
    in_triangle = point_in_triangle_barycentric(u+du, v+dv)
    
    count = 0
    while in_triangle == False:
        count += 1
        
        reflection = False
        
        edge_id, t_edge = edge_intersection_barycentric(u,v, du, dv)
        
        if  System.Mesh[triangle_id]['border_edge'][edge_id] == 1:
            crossed_border = True
            if System.boundary_condition_id == 2:
                return -1
            
        if edge_id == -1:
            print(pos, dX, quaternion, triangle_id)
            # print('pos_init, dX_init, quaternion_init, tri_id_init: ', pos_init, dX_init, quaternion_init, tri_id_init)
            pos[0], pos[1], pos[2] = System.Mesh[triangle_id]['triangle_centroid']
            print('Warning: triangle neighbour edge not found (surface distribute)! Ray marching has been discarded and the molecule has been repositioned to the current triangles` centroid!')
            return triangle_id # -2
        
        # Go to edge
        u += t_edge*du
        v += t_edge*dv
        du -= t_edge*du
        dv -= t_edge*dv
        
        pos[0], pos[1], pos[2] = trf.cartesian_coord(u, v, p0, p1, p2)
        dX[0], dX[1], dX[2] = trf.cartesian_direction(du, dv, p0, p1, p2)
        
        if crossed_border == False:
            next_face = System.Mesh[triangle_id]['neighbours'][edge_id]
        else:
            if System.boundary_condition_id == 0:
                dim = System.Mesh[triangle_id]['border_dim'][edge_id]

                pos[dim]-=2*pos[dim]

                next_face = nearest_triangle(pos, System)

                crossed_border = False
            elif System.boundary_condition_id == 1:
                reflection = True
                border_normal = System.Mesh[triangle_id]['border_normal'][edge_id]
                dX = dX
                dX_dot_n = dX[0]*border_normal[0]+dX[1]*border_normal[1]+dX[2]*border_normal[2]
                dX[0] = dX[0]-2*dX_dot_n*border_normal[0]
                dX[1] = dX[1]-2*dX_dot_n*border_normal[1]
                dX[2] = dX[2]-2*dX_dot_n*border_normal[2]
                crossed_border = False
        
        if reflection == False:

            
            if update_quat == True:
                # Calculate sin_phi and cos_phi and in parallel upate the moelcules quaternion:
                sin_phi, cos_phi = trf.quaternion_plane_to_plane(quaternion, System.Mesh[triangle_id]['triangle_coord'][3], System.Mesh[next_face]['triangle_coord'][3], a_n)
            else:
                # Calculate sin_phi and cos_phi:
                sin_phi, cos_phi = trf.axis_angle_parameters(System.Mesh[triangle_id]['triangle_coord'][3],System.Mesh[next_face]['triangle_coord'][3], a_n)

            
            # Rotation into the plane of the triangle:
            dX[0], dX[1], dX[2] =  trf.rodrigues_rot(dX, a_n, cos_phi, sin_phi)
        
            triangle_id = next_face
            face = triangle_id
            
            
            p0 = System.vertices[System.Mesh[face]['triangles'][0]]
            p1 = System.vertices[System.Mesh[face]['triangles'][1]]
            p2 = System.vertices[System.Mesh[face]['triangles'][2]]
            
            u,v = trf.barycentric_coord(pos, p0, p1, p2, System.Mesh[face]['barycentric_params'])
            
        du,dv = trf.barycentric_direction(dX, p0, p1, p2, System.Mesh[face]['barycentric_params'])
        
        in_triangle = point_in_triangle_barycentric(u+du, v+dv)

        if count > 100:
            pos[0], pos[1], pos[2] = System.Mesh[triangle_id]['triangle_centroid']
            print('Warning: could not find target triangle for surface moelcule within 100 steps. Ray marching has been discarded and the moleule has been repositioned to the current triangles centroid!')
            return triangle_id
        
    u += du
    v += dv
    pos[0], pos[1], pos[2] = trf.cartesian_coord(u, v, p0, p1, p2)
            
    #----------------
    
    if System.boundary_2d == False: # Still need to check whether a bloundary has been crossed
        
        if System.boundary_condition_id == 0:
            crossed_border = False
            for dim in range(3):
                if pos[dim]>System.box_lengths[dim]/2:
                    pos[dim]-=System.box_lengths[dim]
                    crossed_border = True
                elif pos[dim]<-System.box_lengths[dim]/2:
                    pos[dim]+=System.box_lengths[dim]
                    crossed_border = True
                
            if crossed_border==True:
                #find nearest triangle:
                # print('crossed border')
                triangle_id = update_to_nearest_triangle(pos, quaternion, triangle_id, System)
            
        
        elif System.boundary_condition_id == 2:
            
            crossed_border = False
            for dim in range(3):
                if pos[dim]>System.box_lengths[dim]/2:
                    crossed_border = True
                elif pos[dim]<-System.box_lengths[dim]/2:
                    crossed_border = True
                   
            if crossed_border:
                return -1
                
            #----------------   
            
    return triangle_id