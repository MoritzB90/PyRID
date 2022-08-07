# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""


import numpy as np
import numba as nb
from ..geometry.mesh_util import triangle_centroid, closest_boundary_point
from ..math import transform_util as trf
import math


#%%

def point_inside_triangle_test(pos, triangle_id, System):
    
    """Tests if a point is inside a triangle.
    
    Parameters
    ----------
    pos : `float64[3]`
        Position vector
    triangle_id : `int64`
        Triangle id
    System : `object`
        Instance of System class       
    
    
    Returns
    -------
    `boolean`
        True if point is inside triangle, otherwise False.
    
    """    
    
    triangle = System.Mesh[triangle_id]['triangles']
    
    p0 = System.vertices[triangle[0]]
    p1 = System.vertices[triangle[1]]
    p2 = System.vertices[triangle[2]]    
    
    u,v = trf.barycentric_coord(pos, p0, p1, p2, System.Mesh[triangle_id]['barycentric_denom_id'])
    
    in_triangle = point_in_triangle_barycentric(u, v)
    
    if in_triangle:
        print('Point is inside triangle (plane)!')
    else:
        print('Point is not inside triangle (plane)!')
        
    return in_triangle


#%%

@nb.njit
def point_inside_AABB_test(pos, AABB):
    
    """Tests is a point is inside an AABB
    
    Parameters
    ----------
    pos : `float64[3]`
        Position vector
    AABB : `float64[2,3]`
        Array containing the two vector points that represent the lower left and upper right corner of the AABB
    
    
    Returns
    -------
    `boolean`
        True if point inside AABB
    
    """
    
    lower = AABB[0]
    upper = AABB[1]
    
    return lower[0] <= pos[0] and pos[0] <= upper[0] and lower[1] <= pos[1] and pos[1] <= upper[1] and lower[2] <= pos[2] and pos[2] <= upper[2] 
          

#%%

@nb.njit
def ray_mesh_intersection_test(pos, dX, System, cell, t_min, triangle_of_intersection, current_triangle):
    
    """Tests if a line segment intersects with a mesh within a given cell by looping over all triangles in the cell and doing a ray triangle collision test.
    
    Parameters
    ----------
    pos : `float64[3]`
        Position vector
    dX : `float64[3]`
        Direction vector
    System : `object`
        Instance of System class   
    cell : `int64`
        Cell in which to check for ray triangle collisions
    t_min : `float64`
        Current minimum parameterized distance coordinate
    triangle_of_intersection : `int64`
        Current triangle of intersection
    current_triangle : `int64`
        In case a collision has been resolved before, this is the current triangle id the ray origin is located on.
    
    Notes
    -----
    t_min and triangle_of_intersection are only updated if a new detected intersection is closer to the ray origin than the t_min that has been passed to the function. The algorithm has to include these kind of tests, because a triangle may intersect with the current cell and is therefore tested for collision but teh collision itself may take place far away from the current cell because the triangle extends the cell. Also, if a collision has been resolved before, the current orign of the ray will be located in the plane of the triangle the ray collided with. Therefore, this triangle (current_triangle) has to be excluded from the collision test, since a collision test will return True. The algorithm is also optimized such that triangles are not tested multiple times. This could in general be the case since triangles may extend severel cells. Therefore, each triangle keeps a flag/time stamp. The time stamp is increased every time a new ray is cast. For more information also see teh original work by :cite:t:`Amanatides87`.
    
    See Also
    --------
    :func:`~pyrid.geometry.intersection_util.ray_triangle_intersection`
    
    Returns
    -------
    `tuple(boolean, float64, int64)`
        intersection_in_cell, t_min, triangle_of_intersection
    
    """
    
    intersection = False
    intersection_in_cell = False
    
    # TODO: First test if ray origin is inside any compartent. If False, test if ray intersects any compartment AABB. If not, we do not need to continue here. If the ray does intersect an AABB we may also start intersection testing from the AABB boundary, thereby reducing the number of cells we have to travers!
    
    head = System.CellList.head[cell]
    if head != -1:
        Tri_idx = System.CellList[head]['id']
        
        # Check if triangle has not yet been tested in another cell for the same ray using the current time_stamp:
        if System.Mesh[Tri_idx]['stamp']!=System.time_stamp and Tri_idx!=current_triangle:
            triangle = System.vertices[System.Mesh[Tri_idx]['triangles']]
            System.Mesh[Tri_idx]['stamp']=System.time_stamp
            
            p0, p1, p2 = triangle[0],triangle[1],triangle[2]
            
            intersection, t = ray_triangle_intersection(pos, dX, p0, p1, p2, None)
            
            if intersection and t<t_min:
                t_min = t
                triangle_of_intersection = Tri_idx
                intersection_in_cell = True
                
        next = System.CellList[head]['next']
        
        while next!=-1:
            
            Tri_idx = System.CellList[next]['id']
            
            if System.Mesh[Tri_idx]['stamp']!=System.time_stamp and Tri_idx!=current_triangle:
                triangle = System.vertices[System.Mesh[Tri_idx]['triangles']]
                System.Mesh[Tri_idx]['stamp']=System.time_stamp
                
                p0, p1, p2 = triangle[0],triangle[1],triangle[2]
                
                intersection, t = ray_triangle_intersection(pos, dX, p0, p1, p2, None)
                
                if intersection and t<t_min:
                    t_min = t
                    triangle_of_intersection = Tri_idx
                    intersection_in_cell = True
    
            next = System.CellList[next]['next']
    
    return intersection_in_cell, t_min, triangle_of_intersection
        
    
    
#%%


@nb.njit
def ray_triangle_intersection(pos, dX, p0, p1, p2, poi = None):
    
    """Tests if a ray intersects with a triangle using the Möller–Trumbore intersection algorithm :cite:p:`Moeller1997`. For reference see also http://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm.
    
    Parameters
    ----------
    pos : `float64[3]`
        Position vector.
    dX : `float64[3]`
        Direction vector.
    p0 : `float64[3]`
        Vertex 1
    p1 : `float64[3]`
        Vertex 2
    p2 : `float64[3]`
        Vertex 3
    poi : `float64[3]`
        Empty point of intersection vector
    
    Raises
    ------
    NotImplementedError (just an example)
        Brief explanation of why/when this exception is raised
    
    Returns
    -------
    `tuple(boolean, float64)`
        True if there is an intersection, distance to triangle plane.
    
    """
    
    
    eps = 1e-10
    
    e0 = p1 - p0
    e1 = p2 - p0
    
    n = np.empty(3)
    # n = np.cross(ab, ac)
    n[0] = e0[1]*e1[2]-e0[2]*e1[1]
    n[1] = e0[2]*e1[0]-e0[0]*e1[2]
    n[2] = e0[0]*e1[1]-e0[1]*e1[0]
    
    # sign = np.sign(np.dot(dX, normal))
    d = -(dX[0]*n[0]+dX[1]*n[1]+dX[2]*n[2])
    sign = np.sign(d)
    d*= sign
    # print('d: ',d)
    if (abs(d) < eps): 
        return False, 0.0
    
    pos_o = np.empty(3)
    pos_o[0] = pos[0] - p0[0]
    pos_o[1] = pos[1] - p0[1]
    pos_o[2] = pos[2] - p0[2]
    
    ood = 1/d
    t = sign*(pos_o[0]*n[0]+pos_o[1]*n[1]+pos_o[2]*n[2])*ood
    if (t < 0.0):
        # print('t:', t)
        return False, t
    if (t > 1):
        # print('t:', t)
        return False, t
    

    e = np.empty(3)
    e[0] = dX[1]*pos_o[2]-dX[2]*pos_o[1]
    e[1] = dX[2]*pos_o[0]-dX[0]*pos_o[2]
    e[2] = dX[0]*pos_o[1]-dX[1]*pos_o[0]
    
    v = -sign*(e1[0]*e[0]+e1[1]*e[1]+e1[2]*e[2])*ood
    if (v < 0.0 or v > 1):
        # print('v:', v)
        return False, t
    # w = np.dot(ab, e)
    w = sign*(e0[0]*e[0]+e0[1]*e[1]+e0[2]*e[2])*ood
    if (w < 0.0 or v + w > 1):
        # print('w:', w)
        return False, t

    # u = 1.0-v-w
    
    if poi is not None:
        poi[0] = pos[0] + t*dX[0]
        poi[1] = pos[1] + t*dX[1]
        poi[2] = pos[2] + t*dX[2]
    
    # print(v,w)
    return True, t



#%%
    
@nb.njit
def point_inside_mesh_test(triangle_ids, Mesh, vertices, point, tolerance = 1e-6):
    
    """Tests if a point is inside or outside a mesh.
    
    Parameters
    ----------
    triangle_ids : `int64[:]`
        Triangle ids of the mesh
    Mesh : `array_like`
        Mesh structured array containing a field 'triangles' that keeps the vertex ids of the mesh triangles
    vertices : `float64[:,3]`
        Mesh vertices
    point : `float64[3]`
        Position vector
    

    
    Returns
    -------
    `bolean`
        True if point is inside mesh.
    
    """
    
    total_angle = 0
    
    for tri_id in triangle_ids:
        triangle = Mesh[tri_id]['triangles']
    
        p0 = vertices[triangle[0]]
        p1 = vertices[triangle[1]]
        p2 = vertices[triangle[2]]
        
        a = np.subtract(p0, point)
        b = np.subtract(p1, point)
        c = np.subtract(p2, point)
        
        angle = trf.solid_angle(a,b,c)
        normal = trf.normal_vector(p0,p1,p2)
        center = triangle_centroid(p0, p1, p2)
        
        faceVec = np.subtract(point, center)
        dot = np.dot(normal, faceVec)
        
        factor = 1 if dot > 0 else -1
        
        total_angle += angle*factor
        
    abs_total = abs(total_angle)
    
    inside = abs(abs_total - (4*math.pi)) < tolerance
    
    return inside

#%%

@nb.njit
def mesh_inside_box_test(Compartment, System):
    
    """Tests if a mesh intersects at some point with the simulation box.
    
    Parameters
    ----------
    Compartment : `object`
        Instance of Compartment class
    System : `object`
        Instance of System class
    
    
    Returns
    -------
    `boolean`
        True if mesh intersects with simulation box.
    
    """

    if np.any(Compartment.AABB[1]<System.AABB[0]) or np.any(Compartment.AABB[0]>System.AABB[1]):
        return False
    
    
    AABB_center = np.zeros(3)
    AABB_extents = System.box_lengths/2
    
    Intersection = False
    
    for t in Compartment.triangle_ids:
        triangle_index = System.Mesh[t]['triangles']
        # triangle = [System.vertices[triangle_index[0]], System.vertices[triangle_index[1]], System.vertices[triangle_index[2]]]
        p0,p1,p2 = System.vertices[triangle_index[0]], System.vertices[triangle_index[1]], System.vertices[triangle_index[2]]
        
        if triangle_cell_intersection_test(p0,p1,p2, AABB_center, AABB_extents) == True:
            Intersection = True
                        
    return Intersection
    
#%%


@nb.njit
def triangle_cell_intersection_test(p0,p1,p2, cell_center, cell_extent):
    
    """Tests if a triangle intersects with a cell. Based on :cite:p:`Ericson2004`, :cite:p:`AkenineMoellser2001` (See "Real-Time Collision Detection", Chapter 5.2.9 "Testing AAB Against Triangle", p.169 ff). The algorithm is used to create the cell list for the mesh compartment triangles.
    
    Parameters
    ----------
    p0 : `float64[3]`
        Vertex 1
    p1 : `float64[3]`
        Vertex 2
    p2 : `float64[3]`
        Vertex 3
    cell_center : `float64[3]`
        Center of the cell
    cell_extent : `float64[3]`
        Extent of cell
    
    
    Returns
    -------
    `boolean`
        True if triangle intersects with cell.
    
    """
    
    

    # Translate triangle as conceptually moving AABB to origin
    p0 = p0 - cell_center
    p1 = p1 - cell_center
    p2 = p2 - cell_center

    # Compute edge vectors for triangle
    e0 = p1 - p0
    e1 = p2 - p1
    e2 = p0 - p2

    # Test axes a00..a22 (category 3)
    # Test axis a00 (d0 = d1)
    # d0 = p0[2]*e0[1]-p0[1]*e0[2]
    d1 = p1[2]*e0[1]-p1[1]*e0[2]
    d2 = p2[2]*e0[1]-p2[1]*e0[2]
    r = cell_extent[1] * abs(e0[2]) + cell_extent[2] * abs(e0[1])
    if (max(-max(d1, d2), min(d1, d2))) > r:
        return False

    # Test axis a01 (d1 = d2)
    d0 = p0[2]*e1[1]-p0[1]*e1[2]
    # d1 = p1[2]*e1[1]-p1[1]*e1[2]
    d2 = p2[2]*e1[1]-p2[1]*e1[2]
    r = cell_extent[1] * abs(e1[2]) + cell_extent[2] * abs(e1[1])
    if (max(-max(d0, d2), min(d0, d2))) > r:
        return False

    # Test axis a02 (d0 = d2)
    d0 = p0[2]*e2[1]-p0[1]*e2[2]
    d1 = p1[2]*e2[1]-p1[1]*e2[2]
    # d2 = p2[2]*e2[1]-p2[1]*e2[2]
    r = cell_extent[1] * abs(e2[2]) + cell_extent[2] * abs(e2[1])
    if (max(-max(d0, d1), min(d0, d1))) > r:
        return False

    # Test axis a10 (d0 = d1)
    # d0 = p0[0]*e0[2]-p0[2]*e0[0]
    d1 = p1[0]*e0[2]-p1[2]*e0[0]
    d2 = p2[0]*e0[2]-p2[2]*e0[0]
    r = cell_extent[0] * abs(e0[2]) + cell_extent[2] * abs(e0[0])
    if (max(-max(d1, d2), min(d1, d2))) > r:
        return False

    # Test axis a11 (d1 = d2)
    d0 = p0[0]*e1[2]-p0[2]*e1[0]
    # d1 = p1[0]*e1[2]-p1[2]*e1[0]
    d2 = p2[0]*e1[2]-p2[2]*e1[0]
    r = cell_extent[0] * abs(e1[2]) + cell_extent[2] * abs(e1[0])
    if (max(-max(d0, d2), min(d0, d2))) > r:
        return False

    # Test axis a12 (d0 = d2)
    d0 = p0[0]*e2[2]-p0[2]*e2[0]
    d1 = p1[0]*e2[2]-p1[2]*e2[0]
    # d2 = p2[0]*e2[2]-p2[2]*e2[0]
    r = cell_extent[0] * abs(e2[2]) + cell_extent[2] * abs(e2[0])
    if (max(-max(d0, d1), min(d0, d1))) > r:
        return False

    # Test axis a20 (d0 = d1)
    # d0 = p0[1]*e0[0]-p0[0]*e0[1]
    d1 = p1[1]*e0[0]-p1[0]*e0[1]
    d2 = p2[1]*e0[0]-p2[0]*e0[1]
    r = cell_extent[0] * abs(e0[1]) + cell_extent[1] * abs(e0[0])
    if (max(-max(d1, d2), min(d1, d2))) > r:
        return False

    # Test axis a21 (d1 = d2)
    d0 = p0[1]*e1[0]-p0[0]*e1[1]
    # d1 = p1[1]*e1[0]-p1[0]*e1[1]
    d2 = p2[1]*e1[0]-p2[0]*e1[1]
    r = cell_extent[0] * abs(e1[1]) + cell_extent[1] * abs(e1[0])
    if (max(-max(d0, d2), min(d0, d2))) > r:
        return False

    # Test axis a22 (d0 = d2)
    d0 = p0[1]*e2[0]-p0[0]*e2[1]
    d1 = p1[1]*e2[0]-p1[0]*e2[1]
    # d2 = p2[1]*e2[0]-p2[0]*e2[1]
    r = cell_extent[0] * abs(e2[1]) + cell_extent[1] * abs(e2[0])
    if (max(-max(d0, d1), min(d0, d1))) > r:
        return False
    

    # Test the three axes corresponding to the face normals of AABB (category 1)
    # Exit if...
    # ... [-cell_extent[0], cell_extent[0]] and [min(p0[0],p1[0],p2[0]), max(p0[0],p1[0],p2[0])] do not overlap
    if max(p0[0], p1[0], p2[0]) < -cell_extent[0] or min(p0[0], p1[0], p2[0]) > cell_extent[0]:
        return False

    # ... [-cell_extent[1], cell_extent[1]] and [min(p0[1],p1[1],p2[1]), max(p0[1],p1[1],p2[1])] do not overlap
    if max(p0[1], p1[1], p2[1]) < -cell_extent[1] or min(p0[1], p1[1], p2[1]) > cell_extent[1]:
        return False

    # ... [-cell_extent[2], cell_extent[2]] and [min(p0[2],p1[2],p2[2]), max(p0[2],p1[2],p2[2])] do not overlap
    if max(p0[2], p1[2], p2[2]) < -cell_extent[2] or min(p0[2], p1[2], p2[2]) > cell_extent[2]:
        return False


    # Test separating axis corresponding to triangle face normal (category 2)
    # normal = np.cross(e0, e1)
    normal_x = e0[1] * e1[2] - e0[2] * e1[1]
    normal_y = e0[2] * e1[0] - e0[0] * e1[2]
    normal_z = e0[0] * e1[1] - e0[1] * e1[0]
    # distance = np.dot(normal, p0)
    distance = normal_x*p0[0] + normal_y*p0[1] + normal_z*p0[2]
    
    r = cell_extent[0] * abs(normal_x) + cell_extent[1] * abs(normal_y) + cell_extent[2] * abs(normal_z)

    if distance > r:
        return False
    
    return True
  
#%%

# ------------------------
# Barycentric Coordinates
# ------------------------

@nb.njit
def edge_intersection_barycentric(u, v, du, dv):
    
    """Tests intersection of a line segment with triangle edges in barycentric coordinates and returns the edge index and distance to the edge. Edges must be numbered counter clockwise.
    
    Parameters
    ----------
    u : `float64`
        Barycentric u coordinate
    v : `float64`
        Barycentric v coordinate
    du : `float64`
        Barycentric u coordinate of the line segment
    dv : `float64`
        Barycentric v coordinate of the line segment
    
    
    Returns
    -------
    `tuple(int64, float64)`
        Edge index, distance to edge
    
    """

    eps = 1e-10
    
    # edge 0
    if dv != 0.0:
        t0 = -v / dv
    else:
        t0 = -1
    
    # edge 1
    if du + dv != 0.0:
        t1 = (1.0 - u - v) / (du + dv)
    else:
        t1 = -1
    
    # edge 2
    if du != 0.0:
        t2 = -u / du
    else:
        t2 = -1
        
        
    # Test for the smallest positive distance to an edge:
    # min([i for i in [t0, t1, t2] if i >= eps])
    
    if t0 >= eps:
        if t0 <= t1 or t1 < eps:
            if t0 <= t2 or t2 < eps:
                return 0, t0 # edge 0
            else: # We already know that t2 < t0 and t0 < t1
                return 2, t2
            
        elif t1 <= t2 or t2 < eps: # We already know that t1<t0 and t1>= eps.
            return 1, t1 # edge 1
        
        else: #  We already know that t2<t1 amd t2>= eps and since t1<t0 -> t2<t0.
            return 2, t2 # edge 2
            
    elif t1 >= eps:
        if t1 <= t2 or t2 < eps: # t0 is already rejected.
            return 1, t1 # edge 1
        else: # t0 and t1 are rejected and we already know that t2 < eps.
            return 2, t2 # edge 2

    elif t2 >= eps: # t0 and t1 are already rejected
        return 2, t2 # edge 2

    else: # t0, t1 and t2 are all larger than eps.
        return -1, -1
    
    

@nb.njit
def point_in_triangle_barycentric(u,v):
    
    """Tests if a point given in barycentric coordinates is inside the triangle.
    
    Parameters
    ----------
    u : `float64`
        Barycentric u coordinate
    v : `float64`
        Barycentric v coordinate
    
    
    Returns
    -------
    `boolean`
        True if point is inside the triangle.
    
    """
    
    eps = -1e-10
    return (eps<=u<=1 and eps<=v<=1 and u+v<=1)


    
#%%

@nb.njit
def any_ray_mesh_intersection_test(origin, dX_O, System):
    
    """Tests if their is any intersection of a line segment with the mesh. Returns as soon as the first intersection is detected. The algorithm is based on :cite:p:`Amanatides87` : Amanatides et al. 1987 "A Fast Voxel Traversal Algorithm for Ray Tracing" Also see :cite:p:`Ericson2004` : Ericson "Real-Time Collision Detection", chapter 7.4.2 and 7.7
    
    Parameters
    ----------
    origin : `float64[3]`
        Position vector
    dX_O : `float64[3]`
        Direction vector
    System : `object`
        Instance of System class       
    
    
    Returns
    -------
    `boolean`
        True if any intersection is found.
    
    """
    
    
    pos = np.copy(origin)
    dX = np.copy(dX_O)
    
    current_triangle = -1
    
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
        
        
        # INITIALIZATION PHASE:
            
        # identifying the voxel in which the ray System.origin is found:
        cx = np.floor((pos[0]-System.origin[0])/System.cell_length_per_dim[0])
        cy = np.floor((pos[1]-System.origin[1])/System.cell_length_per_dim[1])
        cz = np.floor((pos[2]-System.origin[2])/System.cell_length_per_dim[2])

        
        cx_end = np.floor(((pos[0]+dX[0])-System.origin[0])/System.cell_length_per_dim[0])
        cy_end = np.floor(((pos[1]+dX[1])-System.origin[1])/System.cell_length_per_dim[1])
        cz_end = np.floor(((pos[2]+dX[2])-System.origin[2])/System.cell_length_per_dim[2])
        
        if System.boundary_condition_id == 0:
            cx_end -= cx_end//cx_max*cx_max
            cy_end -= cy_end//cy_max*cy_max
            cz_end -= cz_end//cz_max*cz_max
        
        Next = not (cx_end == cx and cy_end == cy and cz_end == cz)
        
        if Next:

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
        
        while(Next):
            
            cell = int(cx + cy * System.cells_per_dim[0] + cz * System.cells_per_dim[0] * System.cells_per_dim[1])
            
            intersection, t_min, triangle_of_intersection = ray_mesh_intersection_test(pos, dX, System, cell, t_min, triangle_of_intersection, current_triangle)
            
            if intersection:
                return True
            
            
            if (tMaxX < tMaxY):
              if (tMaxX < tMaxZ): 
                cx += stepX
                tMaxX += tDeltaX
                
                if System.boundary_condition_id == 0:
                    if cx>=cx_max:
                        
                        boundary_x = System.box_lengths[0]/2
                        tX = (boundary_x - pos[0])/dX[0]
                        pos[:] += tX*dX
                        
                        dX[:] -= tX*dX
                        pos[0] -= System.box_lengths[0]
                        
                        crossed_border = True
    
                    elif cx<0:
                        
                        boundary_x = -System.box_lengths[0]/2
                        tX = (boundary_x - pos[0])/dX[0]
                        pos[:] += tX*dX
                        
                        dX[:] -= tX*dX
                        pos[0] += System.box_lengths[0]
                            
                        crossed_border = True
                            
                    
              else:
                cz += stepZ
                tMaxZ += tDeltaZ
                
                if System.boundary_condition_id == 0:
                    if cz>=cz_max:
    
                        boundary_z = System.box_lengths[2]/2
                        tZ = (boundary_z - pos[2])/dX[2]
                        pos[:] += tZ*dX
                        
                        dX[:] -= tZ*dX
                        pos[2] -= System.box_lengths[2]
                            
                        crossed_border = True
                        
                    elif cz<0:
                        
                        boundary_z = -System.box_lengths[2]/2
                        tZ = (boundary_z - pos[2])/dX[2]
                        pos[:] += tZ*dX
                        
                        dX[:] -= tZ*dX
                        pos[2] += System.box_lengths[2]
                            
                        crossed_border = True
                        
             
            else:
              if (tMaxY < tMaxZ):
                  
                cy += stepY
                tMaxY += tDeltaY
                
                if System.boundary_condition_id == 0:
                    if cy>=cy_max:
                        
                        boundary_y = System.box_lengths[1]/2
                        tY = (boundary_y - pos[1])/dX[1]
                        pos[:] += tY*dX
                        
                        dX[:] -= tY*dX
                        pos[1] -= System.box_lengths[1]
                            
                        crossed_border = True
                        
                    elif cy<0:
                        cy+=cy_max
                        
                        boundary_y = -System.box_lengths[1]/2
                        tY = (boundary_y - pos[1])/dX[1]
                        pos[:] += tY*dX
                        
                        dX[:] -= tY*dX
                        pos[1] += System.box_lengths[1]
                        
                        crossed_border = True
                        
                    
              else:   
                cz += stepZ
                tMaxZ += tDeltaZ
                
                if System.boundary_condition_id == 0:
                    if cz>=cz_max:
                        
                        boundary_z = System.box_lengths[2]/2
                        tZ = (boundary_z - pos[2])/dX[2]
                        pos[:] += tZ*dX
                        
                        dX[:] -= tZ*dX
                        pos[2] -= System.box_lengths[2]
                            
                        crossed_border = True
                        
                    elif cz<0:
                        
                        boundary_z = -System.box_lengths[2]/2
                        tZ = (boundary_z - pos[2])/dX[2]
                        pos[:] += tZ*dX
                        
                        dX[:] -= tZ*dX
                        pos[2] += System.box_lengths[2]
                        
                        crossed_border = True
                        
            
            
            if crossed_border:
                
                Next = False
                
            else:
                
                # Loop until we reach end point
                Next = not (cx_end == cx and cy_end == cy and cz_end == cz)   
                
                
        # After we exit the while loop, we still need to check the las cell for any collisions if correct_intersection_found is still False:
        if crossed_border == False:
            cell = int(cx + cy * System.cells_per_dim[0] + cz * System.cells_per_dim[0] * System.cells_per_dim[1])
            
            intersection, t_min, triangle_of_intersection = ray_mesh_intersection_test(pos, dX, System, cell, t_min, triangle_of_intersection, current_triangle)                
            
            if intersection:

                return True
                
            else:
                
                passed_all = True
            
        if count>100:
            print('canceled while loop', pos, dX)
            passed_all = True
              
    
    return False


#%%

@nb.njit
def ray_mesh_intersection_count(pos, dX, System, cell, time_stamp, comp_id):
    
    """Counts the number of times a line segments collides/intersects with a mesh surface. Can  be used, e.g., to test whether a point sites inside or outside a mesh if we know that the end point of the line segment is outside the mesh. If the intersection count is odd, the point is located inside the mesh.
    
    Parameters
    ----------
    pos : `float64[3]`
        Position vector
    dX : `float64[3]`
        Direction vector
    System : `object`
        Instance of System class   
    cell : `int64`
        Cell in which to check for ray triangle collisions
    time_stamp : `int64`        
        Each triangle keeps a time stamp that indicates whether the triangle has already been tested for collisions with the current ray. The time stamp is increased every time a new ray is cast.
    comp_id : `int64`
        Compartment index
    
    
    See Also
    --------
    :func:`~pyrid.geometry.intersections_util.point_inside_mesh_test2`
    
    Returns
    -------
    `int64`
        Number of intersections
    
    """
    
    intersection = False
    
    count = 0
    head = System.CellList.head[cell]
    if head != -1:
        
        Tri_idx = System.CellList[head]['id']
        
        if comp_id == 0 or System.Mesh[Tri_idx]['comp_id'] == comp_id:
            # Make sure that this triangle has not yet been tested in another cell for the same ray using the time_stamp:
            if System.Mesh[Tri_idx]['stamp']!=time_stamp:
                triangle = System.vertices[System.Mesh[Tri_idx]['triangles']]
                System.Mesh[Tri_idx]['stamp']=time_stamp
                
                p0, p1, p2 = triangle[0],triangle[1],triangle[2]
                
                intersection, t = ray_triangle_intersection(pos, dX, p0, p1, p2)
                
                count += intersection
            
        next = System.CellList[head]['next']

        while next!=-1:
            
            Tri_idx = System.CellList[next]['id']
            
            if comp_id == 0 or System.Mesh[Tri_idx]['comp_id'] == comp_id:
                # Only continue if this triangle has not yet been checked for collision!  
                if System.Mesh[Tri_idx]['stamp']!=time_stamp:
                    triangle = System.vertices[System.Mesh[Tri_idx]['triangles']]
                    System.Mesh[Tri_idx]['stamp']=time_stamp
                    
                    p0, p1, p2 = triangle[0],triangle[1],triangle[2]
                    
                    intersection, t = ray_triangle_intersection(pos, dX, p0, p1, p2)
                    
                    count += intersection
    
            next = System.CellList[next]['next']
        
    return count
    


@nb.njit
def point_inside_mesh_test_raycasting(pos, comp_id, System): #, my_end_point = None):
    
    """Tests if a point id inside or outside a mesh. 
    The algorithm casts a ray from the given point and counts the number of times it intersects with a triangle/the mesh surface.
    If the ray passes an odd number of triangles, it is in the mesh compartment (see 'point in polygon problem'). 
    For efficiency, we divide space into cells and do a fast voxel traversal.
    
    Parameters
    ----------
    pos : `float64[3]`
        Position vector
    comp_id : `int64`
        Compartment index
    System : `object`
        Instance of System class   
    

    Returns
    -------
    boolean
        True if the point is inside the mesh.
    
    """
    
    
    System.time_stamp += 1
    
    # The first thing we can do is check, if the point is inside or outside the mesh AABB:
    
    # If the point is outside the AABB, there is no need to continue and check if it may 
    # still be inside the mesh, which, by definition of the AABB it wont!
    # However, if the compartment is System and not a mesh, the opposite is true. 
    # If the point is outside, i.e. inside some AABB, it might still be inside System. 
    
    # The point is inside comp 0 (System) if it is outside all the other compartments
    if comp_id == 0: 
        inside_system = True
        for comp_id in range(len(System.Compartments)):
            inside_system = not point_inside_AABB_test(pos, System.Compartments[comp_id+1].AABB) and inside_system
            
        if inside_system == True:
            return True
    else:
        inside_comp = point_inside_AABB_test(pos, System.Compartments[comp_id].AABB)
        
        if inside_comp == False:
            return False
        
    # RAYCAST
    
    # if no endpoint is given (we may also pass an endpoint from which we know that 
    # it is outside any mesh), we simply calculate the closest point to the AABB.
    # Thereby, the ray we cast is as short as possible, so we dont need to check that many cells for triangle collisions.
    # if my_end_point is None:
        
    if comp_id == 0:
        end_point = closest_boundary_point(pos, System.AABB_all, System.AABB)
    else:
        end_point = closest_boundary_point(pos, System.Compartments[comp_id].AABB, System.AABB)
    

    dX = end_point - pos
    
    intersection_count = 0
    
    Max = 100000
        
    
    # INITIALIZATION PHASE:
        
    # identifying the voxel in which the ray System.origin is found:
    cx = np.floor((pos[0]-System.origin[0])/System.cell_length_per_dim[0])
    cy = np.floor((pos[1]-System.origin[1])/System.cell_length_per_dim[1])
    cz = np.floor((pos[2]-System.origin[2])/System.cell_length_per_dim[2])
    
    cx_end = np.floor(((pos[0]+dX[0])-System.origin[0])/System.cell_length_per_dim[0])
    cy_end = np.floor(((pos[1]+dX[1])-System.origin[1])/System.cell_length_per_dim[1])
    cz_end = np.floor(((pos[2]+dX[2])-System.origin[2])/System.cell_length_per_dim[2])

    
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
    
    cell = int(cx + cy * System.cells_per_dim[0] + cz * System.cells_per_dim[0] * System.cells_per_dim[1])
    
    intersection_count += ray_mesh_intersection_count(pos, dX, System, cell, System.time_stamp, comp_id)
    
    #INCREMENTAL PHASE:
    
    Next = not (cx_end == cx and cy_end == cy and cz_end == cz)
    while(Next):
        if (tMaxX < tMaxY):
          if (tMaxX < tMaxZ):
            cx += stepX
            tMaxX += tDeltaX
          else:
            cz += stepZ
            tMaxZ += tDeltaZ
         
        else:
          if (tMaxY < tMaxZ):
            cy += stepY
            tMaxY += tDeltaY
          else:
            cz += stepZ
            tMaxZ += tDeltaZ
            
        cell = int(cx + cy * System.cells_per_dim[0] + cz * System.cells_per_dim[0] * System.cells_per_dim[1])
        
        # Loop until we reach end point
        Next = not (cx_end == cx and cy_end == cy and cz_end == cz)   
        
        intersection_count += ray_mesh_intersection_count(pos, dX, System, cell, System.time_stamp, comp_id)
        

    if comp_id == 0: 
        # For System, to be 'inside' means outside, which is the case for 
        # intersection_count beeing even (which of course includes 0).
        return not bool(intersection_count & 1)
    else: 
        # For any mesh compartment, inside is true for intersection_count beeing odd
        return bool(intersection_count & 1)
    
#%%

# if __name__ == '__main__':
    
    