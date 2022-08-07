# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""


import numpy as np
import numba as nb
# import math


#%%

@nb.njit
def triangle_volume_signed(p0, p1, p2):
    
    """Calculates the signed volume of a the tetraeder represented by a triangle and the coordinate system origin. PyRID uses the signed tetraeder volume to calculate the volume of a mesh.
    
    Parameters
    ----------
    p0 : `float64[3]`
        Vertex 1
    p1 : `float64[3]`
        Vertex 2
    p2 : `float64[3]`
        Vertex 3
    
    
    Returns
    -------
    `float64`
        Signed tetraeder volume
    
    """
    
    #p0.dot(p1.cross(p2)) / 6.0
    
    v210 = p2[0]*p1[1]*p0[2]
    v120 = p1[0]*p2[1]*p0[2]
    v201 = p2[0]*p0[1]*p1[2]
    v021 = p0[0]*p2[1]*p1[2]
    v102 = p1[0]*p0[1]*p2[2]
    v012 = p0[0]*p1[1]*p2[2]
    
    return (1.0/6.0)*(-v210 + v120 + v201 - v021 - v102 + v012)


@nb.njit
def mesh_volume(triangles, vertices):
    
    """Calculates the volume of a mesh.
    
    Parameters
    ----------
    triangles : `int64[:,3]`
        List of vertex indices that make up the trinagles of the mesh
    vertices : `float64[:,3]`
        Vertices of the mesh
    
    
    Returns
    -------
    `float64`
        Mesh volume
    
    """
    
    vol = 0
    for t in triangles:
        
        p0 = vertices[t[0]]
        p1 = vertices[t[1]]
        p2 = vertices[t[2]]
        
        vol += triangle_volume_signed(p0, p1, p2)
    
    return vol


@nb.njit
def triangle_area(p0,p1,p2):
    
    """Calculates the area of a triangle.
    
    Parameters
    ----------
    p0 : `float64[3]`
        Vertex 1
    p1 : `float64[3]`
        Vertex 2
    p2 : `float64[3]`
        Vertex 3
    
    
    Returns
    -------
    `float64`
        Triangle area
    
    """
        
    a= np.linalg.norm(p1-p0)
    b= np.linalg.norm(p2-p0)
    c= np.linalg.norm(p2-p1)
    s = (a+b+c)/2
    
    A = np.sqrt(s*(s-a)*(s-b)*(s-c))
    
    return A



@nb.njit
def triangle_centroid(p0,p1,p2):
    
    """Calculates the centroid of a triangle.
    
    Parameters
    ----------
    p0 : `float64[3]`
        Vertex 1
    p1 : `float64[3]`
        Vertex 2
    p2 : `float64[3]`
        Vertex 3
    
    
    Returns
    -------
    `float64`
        Triangle centroid
    
    """

    return (p0+p1+p2)/3


@nb.njit
def closest_boundary_point(pos, AABB, System_AABB):
    
    """Calculates the closest point on a boundary of an axis aligned bounding box (AABB) to a position vector.
    
    Parameters
    ----------
    pos : `float64[3]`
        Position vector.
    AABB : `float64[2,3]`
        Array containing the two vector points that represent the lower left and upper right corner of the AABB
    System_AABB : `float64[2,3]`
        Array containing the two vector points that represent the lower left and upper right corner of the AABB of the simulation box.
        
    
    Returns
    -------
    `float64[3]`
        Point on the AABB boundary that is closest to the position vector.
    
    """
    
    point = np.copy(pos)
    
    min_distance = 1e10
    border_pos = -1
    border_dim = -1
    
    for dim in range(3):
        if AABB[1][dim]<System_AABB[1][dim] and AABB[0][dim]>System_AABB[0][dim]:
            if pos[dim]>0.0:
                border = AABB[1][dim]
            else:
                border = AABB[0][dim]
                
            distance = abs(border - pos[dim])
            if distance < min_distance:
                min_distance = distance
                border_dim = dim
                border_pos = border
                
    if border_dim == -1:
        print('Warning: closest boundary point is not guaranteed to be outside all mesh compartments! (AABB covering all compartments is larger than simulation box in any dimension.)')
        for dim in range(3):
            if pos[dim]>0.0:
                border = System_AABB[1][dim]
            else:
                border = System_AABB[0][dim]
                
            distance = abs(border - pos[dim])
            if distance < min_distance:
                min_distance = distance
                border_dim = dim
                border_pos = border
                
    point[border_dim] = border_pos
    
    return point


#%%

@nb.njit
def point_triangle_distance(p0,p1,p2, pos, pos_tri = None):
    
    """Calculates the distance between a point and a triangle. Based on :cite:p:`Eberly2001`: 3D Game Engine Design, Eberly D. H., Chapter: 14.3 Point to triangle. https://www.geometrictools.com/Documentation/DistancePoint3triangle3.pdf
    
    Parameters
    ----------
    p0 : `float64[3]`
        Vertex 1
    p1 : `float64[3]`
        Vertex 2
    p2 : `float64[3]`
        Vertex 3
    pos : `float64[3]`
        position vector.
    
    
    Returns
    -------
    `tuple(float64, int64)`
        Distance to triangle, triangle region
    
    """
    

    E0 = p1 - p0 # edge 0
    E1 = p2 - p0 # edge 1
    D = p0 - pos # Coordinates of point relative to triangle origin (at p0)
    
    a = E0[0]*E0[0]+E0[1]*E0[1]+E0[2]*E0[2] # np.dot(E0, E0)
    b = E0[0]*E1[0]+E0[1]*E1[1]+E0[2]*E1[2] # np.dot(E0, E1)
    c = E1[0]*E1[0]+E1[1]*E1[1]+E1[2]*E1[2] # np.dot(E1, E1)
    d = E0[0]*D[0]+E0[1]*D[1]+E0[2]*D[2] # np.dot(E0, D)
    e = E1[0]*D[0]+E1[1]*D[1]+E1[2]*D[2] # np.dot(E1, D)
    f = D[0]*D[0]+D[1]*D[1]+D[2]*D[2] # np.dot(D, D)

    s = b * e - c * d
    t = b * d - a * e
    det = abs(a * c - b * b)

    region = -1
    if (s + t) <= det:
        if s < 0.0:
            if t < 0.0:
                # REGION 4
                region = 4
                if d < 0.0: # minimum on edge t = 0 with s > 0
                    t = 0.0
                    if -d >= a:
                        s = 1.0
                        dist_squared = (a + 2.0*d) + f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
                    else:
                        s = -d / a
                        dist_squared = s*(d) + f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
                else: # minimum on edge s = 0
                    s = 0.0
                    if e >= 0.0:
                        t = 0.0
                        dist_squared = f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
                    else:
                        if -e >= c:
                            t = 1.0
                            dist_squared = (c + 2.0*e) + f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
                        else:
                            t = -e / c
                            dist_squared = t*(e) + f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f

            else:
                # REGION 3
                region = 3
                s = 0.0
                if e >= 0.0:
                    t = 0.0
                    dist_squared = f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
                else:
                    if -e >= c:
                        t = 1.0
                        dist_squared = (c + 2.0*e) + f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
                    else:
                        t = -e / c
                        dist_squared = t*(e) + f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f

        else:
            if t < 0.0:
                # REGION 5
                region = 5
                t = 0.0
                if d >= 0.0:
                    s = 0.0
                    dist_squared = f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
                else:
                    if -d >= a:
                        s = 1.0
                        dist_squared = (a + 2.0*d) + f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
                    else:
                        s = -d / a
                        dist_squared = s*(d) + f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
            else:
                # REGION 0
                region = 0
                det_inv = 1.0 / det
                s *= det_inv
                t *= det_inv
                dist_squared = s*(a*s + 2.0*d) + t*(c*t + 2.0*e) + 2.0*b*s*t + f
                
    else:
        if s < 0.0:
            # REGION 2
            region = 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0: # minimum on edge s + t = 1 with s > 0
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                if numer >= denom:
                    s = 1.0
                    t = 0.0 # t = 1 - s
                    dist_squared = (a + 2.0*d) + f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
                else:
                    s = numer / denom
                    t = 1.0 - s
                    dist_squared = s*(a*s + 2.0*d) + t*(c*t + 2.0*e) + 2.0*b*s*t + f
                    
            else: # minimum on edge s = 0 with t <= 1
                s = 0.0
                if tmp1 <= 0.0:
                    t = 1.0
                    dist_squared = (c + 2.0*e) + f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
                else:
                    if e >= 0.0:
                        t = 0.0
                        dist_squared = f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
                    else:
                        t = -e / c
                        dist_squared = t*(e) + f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
        else:
            if t < 0.0:
                # REGION 6
                region = 6
                tmp0 = b + e
                tmp1 = a + d
                if tmp1 > tmp0: # minimum on edge s + t = 1 with t > 0
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        t = 1.0
                        s = 0.0 # s = 1 - t
                        dist_squared = (c + 2.0*e) + f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
                    else:
                        t = numer / denom
                        s = 1.0 - t
                        dist_squared = s*(a*s + 2.0*d) + t*(c*t + 2.0*e) + 2.0*b*s*t + f

                else: # minimum on edge t = 0 with s <= 0
                    t = 0.0
                    if tmp1 <= 0.0:
                        s = 1.0
                        dist_squared = (a + 2.0*d) + f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
                    else:
                        if d >= 0.0:
                            s = 0.0
                            dist_squared = f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
                        else:
                            s = -d / a
                            dist_squared = s*(d) + f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
            else:
                # REGION 1
                region = 1
                numer = c + e - b - d # (c + e) - (b + d)
                if numer <= 0.0:
                    s = 0.0
                    t = 1.0 # t = 1-s
                    dist_squared = (c + 2.0*e) + f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
                else:
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        s = 1.0
                        t = 0.0 # t = 1-s
                        dist_squared = (a + 2.0*d) + f # s*(a*s + 2*b*t + 2*d) + t*(c*t + 2*e) + f
                    else:
                        s = numer / denom
                        t = 1.0 - s
                        dist_squared = s*(a*s + 2.0*d) + t*(c*t + 2.0*e) + 2.0*b*s*t + f
                        
    
    if dist_squared < 0.0:
        dist = 0.0
    else:
        dist = np.sqrt(dist_squared)

    
    if pos_tri is not None:
        pos_tri[0] = p0[0] + s * E0[0] + t * E1[0]
        pos_tri[1] = p0[1] + s * E0[1] + t * E1[1]
        pos_tri[2] = p0[2] + s * E0[2] + t * E1[2]
    
    return dist, region



