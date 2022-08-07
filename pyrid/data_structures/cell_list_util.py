# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numba as nb
import numpy as np
from numba.experimental import jitclass
from ..geometry.intersections_util import triangle_cell_intersection_test, point_inside_AABB_test

listtype_1 = nb.float64[:]
listtype_2 = nb.int64
listtype_3 = nb.types.ListType(nb.int64)
listtype_4 = nb.int64[:]
listtype_5 = nb.types.ListType(nb.int64[:])

#%%

item_t_CL = np.dtype([('next', np.int64), ('id', np.int64)],  align=True)

spec_CL_array = [
    ('n', nb.int64),
    ('capacity', nb.int64),
    ('item_t', nb.typeof(item_t_CL)),
    ('Data', nb.typeof(np.zeros(1, dtype = item_t_CL))),
    ('head', nb.int64[:]),
]

@jitclass(spec_CL_array)
class CellListMesh(object):
    
    """Dynamic array keeping a cell list for the triangles of a triangulated mesh.
    
    Attributes
    ----------
    n : `int64`
        Number of elements in the dynamic array
    capacity : `int64`
        Capacity of the dynamic array
    Data : `array_like`
        Structured array with fields conatining a free linked list and the triangle ids.
        dtype: np.dtype([('next', np.int64), ('id', np.int64)],  align=True)
    
    Methods
    -------
    method_1(arguments)
        Some general information about this method
    
    """


    def __init__(self):
        self.n = 0 # Count actual elements (Default is 0)
        self.capacity = 1
        self.item_t = item_t_CL
        self.Data = self.make_array(self.capacity)
        self.head = -np.ones(self.capacity, dtype = np.int64)

    def __getitem__(self, k):
        
        """
        Return element at index k.
        """
        
        return self.Data[k]
    
    def __setitem__(self, k, value):
        
        """ 
        Set value of item at index k.
        """
        
        self.Data[k] = value

    def allocate_slot(self):

            if self.n == self.capacity:
                self._resize(2 * self.capacity)      
            
            self.n += 1
            return nb.int64(self.n-1)
        

    def _resize(self, new_cap):
        """
        Resize internal array to capacity new_cap
        """
          
        Data_resized = self.make_array(new_cap) # New bigger array
          
        for k in range(self.n): # Reference all existing values
            Data_resized[k] = self.Data[k]
              
        self.Data = Data_resized # Call A the new bigger array
        self.capacity = new_cap # Reset the capacity
        
    def make_head(self, Ncell):
        self.head = - np.ones(Ncell, dtype = np.int64)
          
    def make_array(self, new_cap):
        """
        Returns a new array with new_cap capacity
        """
        return np.zeros(new_cap, dtype = self.item_t)
    
    def get_triangles(self, cell):
        
        triangles_list = nb.typed.List()
        head = self.head[cell]
        if head != -1:
            # print('id: ', self.Data[head]['id'])
            triangles_list.append(self.Data[head]['id'])
            next = self.Data[head]['next']
            
            while next!=-1:
            # for _ in range(5):
                # print('id: ', self.Data[next]['id'])
                triangles_list.append(self.Data[next]['id'])
                next = self.Data[next]['next']
                
        return triangles_list
    
#%%

@nb.njit
def reverse_cell_mapping(cell, cells_per_dim):
    
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
    
    cz = int(cell / cells_per_dim[0]/ cells_per_dim[1])
    cy = int(cell / cells_per_dim[0])-cz*cells_per_dim[1]
    cx = cell-cy*cells_per_dim[0]-cz*cells_per_dim[0]*cells_per_dim[1]

    return cx, cy, cz

#%%

@nb.njit
def create_cell_list_points(rc, sample_points, Compartment):
    
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
    
    cells_per_dim = (Compartment.box_lengths / rc).astype(np.int64)
    if np.any(cells_per_dim<3):
        print('error: Cell division <3')
    cell_length_per_dim = Compartment.box_lengths / cells_per_dim
    
    # Total number of cells in volume
    Ncell = cells_per_dim.prod()
    # print(Ncell)
    
    cell_list_samples = nb.typed.List.empty_list(listtype_3)
    for _ in range(Ncell):
        cell_list_samples.append(nb.typed.List.empty_list(listtype_2))
    
    N_samples_cell = np.zeros(Ncell, dtype = np.int64)
    nonempty_cells = nb.typed.List.empty_list(listtype_2)
    
    for sample_index, sample in enumerate(sample_points):
        cx = int((sample[0]-Compartment.origin[0]) / cell_length_per_dim[0])
        cy = int((sample[1]-Compartment.origin[1]) / cell_length_per_dim[1])
        cz = int((sample[2]-Compartment.origin[2]) / cell_length_per_dim[2])

        # Determine cell in 3D volume for i-th particle
        cell = cx + cy * cells_per_dim[0] + cz * cells_per_dim[0] * cells_per_dim[1]

        N_samples_cell[cell] += 1
        cell_list_samples[cell].append(sample_index)
                    
        if cell not in nonempty_cells:
            nonempty_cells.append(cell)
            
                        
    return cell_list_samples, N_samples_cell, nonempty_cells, Ncell, cells_per_dim



#%%

@nb.njit##(cache=True)
def create_cell_list_mesh(System, Compartment, cells_per_dim, cell_length_per_dim, triangle_ids, centroid_in_box = False, loose_grid = False):
    
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
    
    # box_lengths = Compartment.box_lengths
    origin = Compartment.origin
    
    Ncell = cells_per_dim.prod()

    CellList = CellListMesh()
    CellList.make_head(Ncell)
    

    for t in triangle_ids:
        
        triangle_index = System.Mesh[t]['triangles']
        
        p0 = System.vertices[triangle_index[0]]
        p1 = System.vertices[triangle_index[1]]
        p2 = System.vertices[triangle_index[2]]
        
        triangle = [p0, p1, p2]
        
        c_tri = np.zeros((3,3), dtype = np.int64)
        
        for i, vertex in enumerate(triangle):
            c_tri[i][0] = int((vertex[0]-origin[0]) / cell_length_per_dim[0])
            if c_tri[i][0]<0:
                c_tri[i][0] = 0
            elif c_tri[i][0]>cells_per_dim[0]-1:
                c_tri[i][0] = cells_per_dim[0]-1
                
            c_tri[i][1] = int((vertex[1]-origin[1]) / cell_length_per_dim[1])
            if c_tri[i][1]<0:
                c_tri[i][1] = 0
            elif c_tri[i][1]>cells_per_dim[1]-1:
                c_tri[i][1] = cells_per_dim[1]-1
            
            c_tri[i][2] = int((vertex[2]-origin[2]) / cell_length_per_dim[2])
            if c_tri[i][2]<0:
                c_tri[i][2] = 0
            elif c_tri[i][2]>cells_per_dim[2]-1:
                c_tri[i][2] = cells_per_dim[2]-1
                
        cx_start = np.min(c_tri[:,0])-1
        if cx_start<0:
            cx_start=0
        cx_end = np.max(c_tri[:,0])+1
        if cx_end>cells_per_dim[0]-1:
            cx_end=cells_per_dim[0]-1
            
        cy_start = np.min(c_tri[:,1])-1
        if cy_start<0:
            cy_start=0
        cy_end = np.max(c_tri[:,1])+1
        if cy_end>cells_per_dim[1]-1:
            cy_end=cells_per_dim[1]-1
            
        cz_start = np.min(c_tri[:,2])-1
        if cz_start<0:
            cz_start=0
        cz_end = np.max(c_tri[:,2])+1
        if cz_end>cells_per_dim[2]-1:
            cz_end=cells_per_dim[2]-1
        
        
        for cx in range(cx_start, cx_end+1):
            for cy in range(cy_start, cy_end+1):
                for cz in range(cz_start, cz_end+1):
                    AABB_extents = cell_length_per_dim/2
                    AABB_center = np.array([(cx+0.5)*cell_length_per_dim[0], (cy+0.5)*cell_length_per_dim[1], (cz+0.5)*cell_length_per_dim[2]])
                    AABB_center += origin
                    
                    c = cx + cy * cells_per_dim[0] + cz * cells_per_dim[0] * cells_per_dim[1]
                    
                    
                    # Currently, a fine grid is used. Here, we iterate over many cells for larger molecules. The fine grid is more performand for the fast voxel traversal algorithm. 
                    #TODO: However, a loose grid would still make sense for particle - Triangle interactions via an energy potential function!
                    if loose_grid:
                        mult = 2 # We take 2*AABB_extents because we want to have a loose grid, such that cells overlap.
                    else:
                        mult = 1
                    if triangle_cell_intersection_test(p0, p1, p2, AABB_center, mult*AABB_extents) == True:
                        
                        inside_box = True
                        if centroid_in_box:
                            inside_box = point_inside_AABB_test(System.Mesh[t]['triangle_centroid'], Compartment.AABB)
                            
                        if inside_box:
                            slot = CellList.allocate_slot()
                            CellList[slot]['next'] = CellList.head[c]
                            CellList.head[c] = slot
                            CellList[slot]['id'] = t
                       
    
    AABB_centers = np.zeros((Ncell,3))
    
    for cx in range(cells_per_dim[0]):
        for cy in range(cells_per_dim[1]):
            for cz in range(cells_per_dim[2]):
                AABB_extents = cell_length_per_dim/2
                AABB_center = np.array([(cx+0.5)*cell_length_per_dim[0], (cy+0.5)*cell_length_per_dim[1], (cz+0.5)*cell_length_per_dim[2]])
                AABB_center += origin
                
                c = cx + cy * cells_per_dim[0] + cz * cells_per_dim[0] * cells_per_dim[1]
                AABB_centers[c] = AABB_center
            
            
    return CellList, AABB_centers
