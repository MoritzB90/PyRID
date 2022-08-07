# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numpy as np
import numba as nb
import math
import cmath


#%%

# ------------
# Tests
# ------------

@nb.njit
def isclose(a,b, rel_tol=1e-05, abs_tol=0.0):
    
    """
    Returns a boolean array where two arrays are element-wise equal within a tolerance.
    Reduced copy of the numpy fucntion isclose(), which is, however, currently not supported by numba.
    
    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter.
    atol : float
        The absolute tolerance parameter.
        
    """
    
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)



def valid_mobility_tensor_test(Mu, Mu_sqrt):
    
    """Tests whether a mobility tensor i a positive semidefinite matrix. For mobility tensors and diffusion tesnors to have proper physical meaning, they should be real positive semidefinite. . 
    
    Parameters
    ----------
    Mu : `float64[3,3]`
        Mobility tensor
    Mu_sqrt : `float64[3,3]`
        Square root of mobility tensor
    
    .. note::
        
        Experimentaly estimated diffusion tensors are, although non-physical, not always positive semidefinite (see e.g. :cite:t`Niethammer2006`).
        Also, in case of overlapping beads, the diffusion tensor calculated by PyRID is not necessarily positive semidefinite!
        
    See Also
    --------
    :func:`~pyrid.molecules.hydro_util`
        
    Raises
    ------
    Warning('Diffusion tensor should be a positive semidefinite matrix!')
        Warning is raised if the diffusion tensor and the mobility tensor respectively are not positive semidefinite.
    ValueError('Error: complex square root of the mobility tensor detected. Square root must be real valued (Diffusion tensor should be a real positive semidefinite matrix).')
        Raised in case the square root of the mobility tensor contains complex values.
    
    """
    
    # Note: Experimentaly estimated diffusion tensors are, although non-physical, not always positive semidefinite (see e.g. Niethammer et al. 2006, "On Diffusion Tensor Estimation", IEEE)
    
    # Diagonalize
    eVal, eVec = np.linalg.eig(Mu)
    # Test if positive definite:
    if not np.all(eVal.real >= 0):
        raise Warning('Diffusion tensor should be a positive semidefinite matrix!')
    
    rtol=1e-05
    
    mask = (np.abs(Mu_sqrt.real)>0.0)&(np.abs(Mu_sqrt.imag)>0.0)
    if np.any(Mu_sqrt[mask].imag/Mu_sqrt[mask].real > rtol):
        raise ValueError('Error: complex square root of the mobility tensor detected. Square root must be real valued (Diffusion tensor should be a real positive semidefinite matrix).')
       

@nb.njit
def is_diagonal(M):
    
    """Tests whether matrix x is diagonal.
    
    Parameters
    ----------
    M : `float[:,:]`
        Matrix
    
    
    """
    
    rtol=1e-05
    
    if np.sum(np.abs(M) - np.diag(np.diag(np.abs(M))))/np.sum(np.diag(np.abs(M))) > rtol:
        print('Matrix is not diagonal!')
        

#%%

# ---------------
# Mapping
# ---------------

@nb.njit
def cantor_pairing(k1,k2):
    
    """The Cantor pairing function maps a natural number (unsigned integer) pair k1,k2 to a unique natural number n such that the mapping is bijective, i.e. the original pair can always be recovered from n.
    
    Parameters
    ----------
    k1 : `uint64`
        Natural number 1
    k3 : `uint64`
        Natural number 2
    
    Notes
    -----
    This function is just needed when validating the reaction registry class against
    new implementations which may change the order in which reactions are evaluated. The Cantor pairing function enables to assign a unique id to each educt pair and thereby enables sorting the reactions by the educts! Educt order is not exchangeable!
    
    Returns
    -------
    `uint64`
        Result of the Cantor pairing
    
    """
    
    return 1/2*(k1+k2)*(k1+k2+1)+k2


@nb.njit
def unique_pairing(k1,k2):
    
    """The unique pairing function maps a natural number (unsigned integer) pair k1,k2 to a unique natural number n such that the mapping is bijective, i.e. the original pair can always be recovered from n.
    
    Parameters
    ----------
    k1 : `uint64`
        Natural number 1
    k3 : `uint64`
        Natural number 2
    
    Notes
    -----
    This function is just needed when validating the reaction handling functions against
    new implementations which may change the order in whcih reactions are resolved. The function enables to assign a unique id for each educt pair and thereby enables sorting the reactions by the educts! Educt order is exchangeable!
    
    from: https://math.stackexchange.com/questions/882877/produce-unique-number-given-two-integers
    
    Returns
    -------
    `uint64`
        Result of the unique pairing
        
    """
    
    max_ = max((k1,k2))
    return (max_*(max_+1))/2+min((k1,k2))
    

#%%

# ----------------------
# Some usefull functions
# ----------------------

@nb.njit
def sqrt_matrix(M):
    
    """Calculates the square root of a matrix by doing an eigenvalue decomposition and then taking the square root of the eigenvalues before reconstructing the matric form. Only works for diagonalzable matrices!
    
    Parameters
    ----------
    M : `float[:,:]`
        Matrix
    
    See Also
    --------
    :func:`~pyrid.math.transform_util.is_diagonal`
    
    
    Returns
    -------
    `float64[:,:]`
        Square root of matric M
    
    """
    
    def sqrtc(V):
        
        result = np.empty(V.shape, dtype = np.complex64)
        for i in range(V.shape[0]):
            result[i] = cmath.sqrt(V[i])
                
        return result
    
    # Diagonalize
    eVal, eVec = np.linalg.eig(M)
    # Test if positive definite:
    # if not np.all(eVal.real >= 0):
    #     raise ValueError('Unable to caclulate square root of mobility tensor! Only real positive semidefinite matrices are supported!')
    return eVec.dot(np.diag(sqrtc(eVal))).dot(np.linalg.inv(eVec)).real


@nb.njit
def normalize(v):
    
    """Normalizes a vector v.
    
    Parameters
    ----------
    v : `float64[:]`
        Vector
    
    
    Returns
    -------
    `float64[:]`
        Normalized vector
    
    """
    
    return v / np.linalg.norm(v)


@nb.njit
def normal_vector(p0,p1,p2):
    
    """Returns the normal vector of a triangle given its vertex vectors p0, p1 and p2.
    
    Parameters
    ----------
    p0 : `float64[3]`
        Vertex 1
    p1 : `float64[3]`
        Vertex 2
    p2 : `float64[3]`
        Vertex 3
    
    .. note::
        The direction of the triangle normal vector depends on the vertex order (clockwise or counter-clockwise)!
    
    Returns
    -------
    `float64[3]`
        Normal vector of the triangle.
    
    """
    
    e0 = p1 - p0
    e1 = p2 - p0
    return np.cross(e0, e1)

@nb.njit
def eij(p0,p1):
    
    """Normalizes the edge represented by the vertices p0 and p1. This edge can be used as the first coordinate vector when constructing a triangle's local coordinate frame.
    
    Parameters
    ----------
    p0 : `float64[3]`
        Vertex 1
    p1 : `float64[3]`
        Vertex 2
    
    
    Returns
    -------
    `float64[3]`
        Normalized edge vector
    
    """
    
    return normalize(p1-p0)

@nb.njit
def ek(p0,p2, ej):
    
    """Given the triangle vertices p0 and p2 and the normalized triangle edge norm(p1-p0), a vector that is orthogonal to the edge but lies in teh triangle plane is returned. Incombination with the triangle normal vector these 3 vector make up the triangle's local coordinate frame.
    
    Parameters
    ----------
    p0 : `float64[3]`
        Vertex 1
    p2 : `float64[3]`
        Vertex 3
    ej : `float64[3]`
        Normalized edge vector p1-p0
    
    
    Returns
    -------
    `float64[3]`
        Normalized vector orhogonal to ej that lies within the plane of triangle p0, p1, p2.
    
    """
    
    return normalize(p2-p0-ej*np.dot(ej,p2-p0))


@nb.njit
def local_coord(p0, p1, p2):
    
    """Returns the local coordinate system of the triangle represented by the three vertices p0,p1 and p2.
    
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
    `tuple(float64[3], float64[3], float64[3], float64[3])`
        Local coordinate system of the triangle (origin, ex,ey,ez).
    
    """

    origin = p0
    ex = eij(p0,p1)
    ey = ek(p0,p2, ex)
    ez = normalize(normal_vector(p0,p1,p2))
    
    return origin, ex,ey,ez

@nb.njit
def solid_angle(p0,p1,p2):
    
    """Calculates the solid angle defined by the tetraeder that is represented by the triangle with vertices p0, p1, and p2 and the coordinate system origin. Is used by PyRID to determine whether a point sited inside or outside a mesh.
    
    Parameters
    ----------
    p0 : `float64[3]`
        Vertex 1
    p1 : `float64[3]`
        Vertex 2
    p2 : `float64[3]`
        Vertex 3
    
    See Also
    --------
    :func:`pyrid.geometry.mesh_util.point_inside_mesh_test`
    
    Returns
    -------
    `float64`
        Solid angle
    
    """
    
    p0 = normalize(p0)
    p1 = normalize(p1)
    p2 = normalize(p2)
    
    numerator = np.dot(np.cross(p0,p1),p2)
    denomenator = 1 + np.dot(p0,p1) + np.dot(p1,p2) + np.dot(p2,p0)
    
    angle = 2 * math.atan2(numerator, denomenator)
    
    return abs(angle)

@nb.njit
def cross(v0, v1):
    
    """Calculates the cross product between vectors v1 and v2.
    
    Parameters
    ----------
    v1 : `float64[3]`
        Vector 1
    v2 : `float64[3]`
        Vector 2
    
    Returns
    -------
    `float64[3]`
        Cross product between v1 and v2.s
    
    """

    result = np.array([v0[1] * v1[2] - v0[2] * v1[1],
                            v0[2] * v1[0] - v0[0] * v1[2],
                            v0[0] * v1[1] - v0[1] * v1[0]])
    return result


@nb.njit
def half_angle(cos_phi):
    
    """Calculates the half angle cosine and sine functions (cos(phi/2), sin(phi/2)) given the full angle cosine function cos(phi).
    
    Parameters
    ----------
    cos_phi : `float64`
        cos(phi).
    
    
    Returns
    -------
    `tuple(float64, float64)`
        sin(phi/2), cos(phi/2)
    
    """
    
    if -1.0<=cos_phi<=1.0:
        # phi = np.arccos(cos_phi)
        cos_half = np.sqrt((1+cos_phi)/2)
        sin_half = np.sqrt((1-cos_phi)/2)
    elif cos_phi>1.0:
        # phi = 0.0
        sin_half = 0.0
        cos_half = 1.0
    else:
        # phi = pi
        sin_half = 1.0
        cos_half = 0.0
        
    return sin_half, cos_half


#%%

# -----------------
# Reflection
# -----------------

@nb.njit
def collision_response(v, normal, v_refl):
    
    """Calculates the collision response of a ray colliding with a mesh triangle.
    The collision is resolved via reflection along the triangle plane.
    
    Parameters
    ----------
    v : `float64[3]`
        Ray vector which to reflect
    normal : `float64[3]`
        Normal vector of the triangle plane
    v_refl : `float64[3]`
        Empty array to be filled with the values of the reflected vector
    
    Notes
    -----
    Reflection is calculated by
    
    .. math::
        
        \\vec{v}_{ref} = \\vec{v} - 2 (\\vec{v} \\cdot \\hat{n}) \\hat{n}
        
    where :math:`\\hat{n}` is the normal vector of the plane at which :math:`\\vec{v}` is reflected.
    
    
    """
    
    v_dot_n = v[0]*normal[0]+v[1]*normal[1]+v[2]*normal[2]
    
    v_refl[0] = v[0]-2*v_dot_n*normal[0]
    v_refl[1] = v[1]-2*v_dot_n*normal[1]
    v_refl[2] = v[2]-2*v_dot_n*normal[2]
    
    
#%%

# -------------------
# Quaternions
# -------------------



@nb.njit
def rot_quaternion(phi = 0, a_n = None):
    
    """Returns a quaternion, corresponding to a rotation about a given axis by a given angle. If no angle or axis is passed [1.0, 0.0, 0.0, 0.0] is returned, correpsonding to a rotation by zero degree.
    
    Parameters
    ----------
    phi : `float64`
        Angle, given in rad.
    a_n : `float64[3]`
        Rotation axis normal vector 
    
    Notes
    -----
    A rotation around the axis vector :math:`\\boldsymbol{u}` is represented in quaternion form:
        
    .. math::
        
        \\boldsymbol{q_{rot}} = \\cos(\\phi/2) + (u_x \\boldsymbol{i} + u_y \\boldsymbol{i} + u_z \\boldsymbol{i}) \\sin(\\phi/2),
                                                                                                                                                              
    """
    
    if a_n is not None:
        return np.array([np.cos(phi/2), a_n[0]*np.sin(phi/2), a_n[1]*np.sin(phi/2), a_n[2]*np.sin(phi/2)], dtype=np.float64)
    else:
        return np.array([1.0, 0.0, 0.0, 0.0])


@nb.njit
def quaternion_plane_to_plane(quaternion, n0, n1, a_n):
    
    """
    updates the quaternion corresponding to an orientation in plane n0 to the orientation in plane n1 and returns sin(phi) and cos(phi).
    
    Parameters
    ----------
    quaternion : `float64[4]`
        Quaternion
    normal : `float64[3]`
        Normal vector 1
    normal : `float64[3]`
        Normal vector 2
    a_n : `float64[3]`
        Empty rotation axis vector
    
    Returns
    -------
    `float64[4]`
        Rotation quaternion
        
    """
    
    # a_n = np.empty(3)
    
    #Rotation into the plane of the triangle:
        
    sin_phi, cos_phi = axis_angle_parameters(n0,n1,a_n)
    
    sin_half, cos_half = half_angle(cos_phi)
    
    quaternion0 = np.array([cos_half, a_n[0]*sin_half, a_n[1]*sin_half, a_n[2]*sin_half], dtype=np.float64)
    
    quaternion[:] = quat_mult(quaternion, quaternion0)     
        
    return sin_phi, cos_phi
    

@nb.njit
def quaternion_to_plane(normal, quaternion):
    
    """
    Creates a rotation quaternion representing the rotation of a vector [0.0, 0.0, 1.0] to a given plane normal vector.
    
    Parameters
    ----------
    normal : `float64[3]`
        Normal vector
    quaternion : `float64[4]`
        Quaternion
    
    
    Returns
    -------
    `float64[4]`
        Rotation quaternion
        
    """
    
    a_n = np.empty(3)
    
    #Rotation into the plane of the triangle:
    sin_half, cos_half = axis_halfangle_parameters(np.array([0.0,0.0,1.0]),normal,a_n)
    
    quaternion[:] = np.array([cos_half, a_n[0]*sin_half, a_n[1]*sin_half, a_n[2]*sin_half], dtype=np.float64)
    
    return quaternion
    

@nb.njit 
def quaternion_random_axis_rot(quaternion, a_n):
    
    """
    Updates an orientation quaternion by a random rotation around a given axis.
    """
    
    phi = np.random.rand()*2*np.pi
    quaternion0 = rot_quaternion(phi, a_n)
    quaternion[:] = quat_mult(quaternion, quaternion0)
    
            
    
@nb.njit
def quat_mult(q1, q2):
    
    """Executes the quaternion multiplication between two quaternions q1 and q2.
    
    Parameters
    ----------
    q1 : `float64[4]`
        Quaternion 1
    q2 : `float64[4]`
        Quaternion 2
    
    
    Returns
    -------
    `float64[4]`
        Quaternion product q1*q2
    
    """
    
    w0, x0, y0, z0 = q1
    w1, x1, y1, z1 = q2
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)





#%%

# ------------------------
# Barycentric Coordinates
# ------------------------
    

@nb.njit
def tri_area_2D(x1, y1, x2, y2, x3, y3):
    """
    Calculates the area of a triangle in 2D.
    
    Parameters
    ----------
    x1 : `float64`
        X value of vertex 1
    x2 : `float64`
        X value of vertex 2
    x3 : `float64`
        X value of vertex 3
    y1 : `float64`
        Y value of vertex 1
    y2 : `float64`
        Y value of vertex 2
    y3 : `float64`
        y value of vertex 3
        
    Returns
    -------
    `float64`
        Triangle area  
    """
    
    return (x1-x2)*(y2-y3) - (x2-x3)*(y1-y2)

@nb.njit
def barycentric_coord_projection_method(pos, p0, p1, p2, normal):
    
    """Calculates the barycentric coordinates of a position vector with respect to the triangle represented by the vertices p0, p1, and p2. The algorithm from :cite:t:`Ericson2004` is used. The algorithm takes advantage of the fact that the barycentric coordinates remain invariant under projections. It projects the vertices to either of the planes xy, yz, xz and then calculates the barycentric coordinates from the sub-triangle areas.
    
    Parameters
    ----------
    pos : `float64[3]`
        Position vector
    p0 : `float64[3]`
        Vertex 1
    p1 : `float64[3]`
        Vertex 2
    p2 : `float64[3]`
        Vertex 3
    normal : `float64[3]`
        Unnormalized triangle normal vector.
    
    
    Returns
    -------
    `tuple(float64, float64)`
        Barycentric coordinates u, v.
    
    """
    
    # Nominators and one-over-denominator for u and v ratios
    # float nu, nv, ood
    # Absolute components for determining projection plane
    x = abs(normal[0]) # Unnormalized triangle normal
    y = abs(normal[1])
    z = abs(normal[2])
    
    # Compute areas in plane of largest projection
    if (x >= y and x >= z):
        # x is largest, project to the yz plane
        nw = tri_area_2D(pos[1], pos[2], p1[1], p1[2], p2[1], p2[2]) # Area of PBC in yz plane
        nu = tri_area_2D(pos[1], pos[2], p2[1], p2[2], p0[1], p0[2]) # Area of PCA in yz plane
        ood = 1.0 / normal[0] # 1/(2*area of ABC in yz plane)

    elif (y >= x and y >= z):
        # y is largest, project to the xz plane
        nw = tri_area_2D(pos[0], pos[2], p1[0], p1[2], p2[0], p2[2])
        nu = tri_area_2D(pos[0], pos[2], p2[0], p2[2], p0[0], p0[2])
        ood = 1.0 / -normal[1]
    else:
        # z is largest, project to the xy plane
        nw = tri_area_2D(pos[0], pos[1], p1[0], p1[1], p2[0], p2[1])
        nu = tri_area_2D(pos[0], pos[1], p2[0], p2[1], p0[0], p0[1])
        ood = 1.0 / normal[2]
    
    w = nw * ood
    u = nu * ood
    v = 1.0 - w - u

    return u,v

    

@nb.njit
def barycentric_params(p0, p1, p2):
    
    """Precalculates the fixed parameters for the barycentric coordinates of a triangle represented by the vertices p0, p1, and p2. The algorithm is taken from :cite:t:`Ericson2004`.
    
    Parameters
    ----------
    p0 : `float64[3]`
        Vertex 1
    p1 : `float64[3]`
        Vertex 2
    p2 : `float64[3]`
        Vertex 3
        
    """
    
    e0 = p1 - p0
    e1 = p2 - p0
    d00 = np.dot(e0, e0)
    d01 = np.dot(e0, e1)
    d11 = np.dot(e1, e1)
    denom = d00 * d11 - d01 * d01
    
    return d00, d01, d11, denom
    
    
@nb.njit
def barycentric_coord(pos, p0, p1, p2, barycentric_params):

    """Calculates the barycentric coordinates of a position vector with respect to the triangle represented by the vertices p0, p1, and p2. The algorithm from :cite:t:`Ericson2004` is used.
    
    Parameters
    ----------
    pos : `float64[3]`
        Position vector
    p0 : `float64[3]`
        Vertex 1
    p1 : `float64[3]`
        Vertex 2
    p2 : `float64[3]`
        Vertex 3
    barycentric_params : `float64[4]`
        Precalculated parameters.
    
    
    Returns
    -------
    `tuple(float64, float64)`
        Barycentric coordinates u, v.
    
    """
    
    e0 = p1 - p0
    e1 = p2 - p0
    d00 = barycentric_params[0]
    d01 = barycentric_params[1]
    d11 = barycentric_params[2]
    denom = barycentric_params[3]
        
    pos_O = pos - p0
    d20 = np.dot(pos_O, e0)
    d21 = np.dot(pos_O, e1)
    
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    # w = 1.0 - u - v

    return u,v

    
@nb.njit
def barycentric_direction(dX, p0, p1, p2, barycentric_params):

    """Calculates the barycentric coordinates of a direction vector with respect to the triangle represented by the vertices p0, p1, and p2. The algorithm from :cite:t:`Ericson2004` is used.
    
    Parameters
    ----------
    pos : `float64[3]`
        Position vector
    p0 : `float64[3]`
        Vertex 1
    p1 : `float64[3]`
        Vertex 2
    p2 : `float64[3]`
        Vertex 3
    barycentric_params : `float64[4]`
        Precalculated parameters.
    
    
    Returns
    -------
    `tuple(float64, float64)`
        Barycentric coordinates u, v.
    
    """
    
    e0 = p1 - p0
    e1 = p2 - p0
    d00 = barycentric_params[0]
    d01 = barycentric_params[1]
    d11 = barycentric_params[2]
    denom = barycentric_params[3]
        
    d20 = np.dot(dX, e0)
    d21 = np.dot(dX, e1)
    
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    # w = 1.0 - u - v

    return u,v

    


@nb.njit
def cartesian_coord(u, v, p0,p1,p2):
    
    """Calculates the cartesian coordinates of a position vector given in barycentric coordinates.
    
    Parameters
    ----------
    u : `float64`
        Barycentric u coordinate
    v : `float64`
        Barycentric v coordinate
    p0 : `float64[3]`
        Vertex 1
    p1 : `float64[3]`
        Vertex 2
    p2 : `float64[3]`
    
    
    Returns
    -------
    `float64[3]`
        Position vector in cartesian coordiantes.
    
    """

    return p0[0] + u*(p1[0] - p0[0]) + v*(p2[0] - p0[0]), p0[1] + u*(p1[1] - p0[1]) + v*(p2[1] - p0[1]), p0[2] + u*(p1[2] - p0[2]) + v*(p2[2] - p0[2])



@nb.njit
def cartesian_direction(du, dv, p0,p1,p2):
    
    """Calculates the cartesian coordinates of a direction vector given in barycentric coordinates.
    
    Parameters
    ----------
    du : `float64`
        Barycentric u coordinate of the direction vector
    dv : `float64`
        Barycentric v coordinate of the direction vector
    p0 : `float64[3]`
        Vertex 1
    p1 : `float64[3]`
        Vertex 2
    p2 : `float64[3]`
    
    
    Returns
    -------
    `float64[3]`
        Direction vector in cartesian coordiantes.
    
    """

    return du*(p1[0] - p0[0]) + dv*(p2[0] - p0[0]), du*(p1[1] - p0[1]) + dv*(p2[1] - p0[1]), du*(p1[2] - p0[2]) + dv*(p2[2] - p0[2])

#%%

# -------------------
# Axis Agle Rotations
# -------------------

@nb.njit
def orthogonal_vector(v, v_perp):
    
    """
    Calculates a vector orthoginal to v.
    """
    
    if v[2] != 0 or v[1] != 0:
        v_perp[0] = 0.0
        v_perp[1] = v[2]
        v_perp[2] = -v[1]
    else:
        v_perp[0] = 0.0
        v_perp[1] = 0.0
        v_perp[2] = -v[0]

        

@nb.njit
def axis_angle_parameters(n0,n1,a_n):
    
    """Calculates the normalized rotation axis and angle (sin(phi), cos(phi)) between two normalized vectors. Accounts for the two special cases where n0=n1 and n0=-n1.
    
    Parameters
    ----------
    n0 : `float64[3]`
        Vector 1
    n1 : `float64[3]`
        Vector 2
    a_n : `float64[3]`
        Empty vector to be filled with values of the normalized rotation axis vector
    
    
    Returns
    -------
    `tuple(float64, float64)`
        sin(phi), cos(phi)
    
    """
    
    # Rotation axis is given by the cross product:
    a_n[0], a_n[1], a_n[2] = n0[1] * n1[2] - n0[2] * n1[1], n0[2] * n1[0] - n0[0] * n1[2], n0[0] * n1[1] - n0[1] * n1[0]
    
    a_n_sqrd = a_n[0]*a_n[0] + a_n[1]*a_n[1] + a_n[2]*a_n[2]
        
    if a_n_sqrd > 0.0:
        
        # Since we have already calculated the cross product, sin(phi) can efficiently be calculated. 
        a_norm = np.sqrt(a_n_sqrd)
        sin_phi = a_norm
        # And cos(phi) by:
        cos_phi = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2]
        # normalize the axis vector:
        a_n[:] /= a_norm
        
        return sin_phi, cos_phi
        
    else: # no and n1 are either parallel or antiparallel
        
        n0_dot_n1 = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2]
    
        if n0_dot_n1<0.0: # antiparallel
            
            # If n1 is rotated by 180 relative to n0 there is no unique rotation axis and the cross product is 0. As such, we may rotate about any axis that lies in the plane defined by either of the two vectors:
            
            orthogonal_vector(n0, a_n) # Arbitrary vector perpendicular to n0
            a_norm = np.sqrt(a_n[0]*a_n[0] + a_n[1]*a_n[1] + a_n[2]*a_n[2])
            
            sin_phi = 0.0 # sin(pi) = 0
            cos_phi = -1.0 # cos(pi) = -1
            # normalize the axis vector:
            a_n[:] /= a_norm
            
            return sin_phi, cos_phi
        
        else: # parallel (n0 == n1)
            
            cos_phi = 1.0 # cos(0) = 1
            sin_phi = 0.0 # sin(0) = 0
            
            return sin_phi, cos_phi
            
#%%

@nb.njit
def axis_halfangle_parameters(n0,n1,a_n):
    
    """Calculates the normalized rotation axis and half-angle (sin(phi/2), cos(phi/2)) between two normalized vectors.
    
    Parameters
    ----------
    n0 : `float64[3]`
        Vector 1
    n1 : `float64[3]`
        Vector 2
    a_n : `float64[3]`
        Empty vector to be filled with values of the normalized rotation axis vector
    
    
    Returns
    -------
    `tuple(float64, float64)`
        sin(phi), cos(phi)
    
    """
    
    # Rotation axis is given by the cross product:
    a_n[0], a_n[1], a_n[2] = n0[1] * n1[2] - n0[2] * n1[1], n0[2] * n1[0] - n0[0] * n1[2], n0[0] * n1[1] - n0[1] * n1[0]
    
    a_n_sqrd = a_n[0]*a_n[0] + a_n[1]*a_n[1] + a_n[2]*a_n[2]
        
    if a_n_sqrd > 0.0:
        
        cos_phi = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2]
        
        sin_half, cos_half = half_angle(cos_phi)
        
        a_norm = np.sqrt(a_n[0]*a_n[0] + a_n[1]*a_n[1] + a_n[2]*a_n[2])
        # normalize the axis vector:
        a_n[:] /= a_norm
        
        return sin_half, cos_half
        
    else: # no and n1 are either parallel or antiparallel
        
        n0_dot_n1 = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2]
    
        if n0_dot_n1<0.0: # antiparallel
            
            # If n1 is rotated by 180 relative to n0 there is no unique rotation axis and the cross product is [0,0,0]. We could rotate about any axis that lies in the plane defined by either of the two vectors:
            
            cos_half = 0.0 # cos(pi/2) = 0
            sin_half = 1.0 # sin(pi/2) = 1
            
            orthogonal_vector(n0, a_n) # We simply use an arbitrary vector perpendicular to n0 for a_n.
            a_norm = np.sqrt(a_n[0]*a_n[0] + a_n[1]*a_n[1] + a_n[2]*a_n[2])
            a_n[:] /= a_norm
            
            return sin_half, cos_half
        
        else: # parallel (n0 == n1)
            
            cos_half = 1.0 # cos(0) = 1
            sin_half = 0.0 # sin(0) = 0
            
            return sin_half, cos_half




@nb.njit
def rodrigues_rot(v, a_n, cos_phi, sin_phi):
    
    """Rotates vector v by an angle alpha around the axis vector a using the Rodriguez rotation formula.
    
    Parameters
    ----------
    v : `float64[3]`
        Vector which to rotate
    a_n : `float64[3]`
        Normalized rotation axis vector.
    cos_phi : `float64`
        Cosine of the angle phi by which to rotate vector v.
    sin_phi : `float64`
        Sine of the angle phi by which to rotate vector v.     
    
    
    Returns
    -------
    `float64[3]`
        Rotated vector
    
    """
    
    # vxa = cross(v, a_n)
    vxa_0 = v[1] * a_n[2] - v[2] * a_n[1]
    vxa_1 = v[2] * a_n[0] - v[0] * a_n[2]
    vxa_2 = v[0] * a_n[1] - v[1] * a_n[0]
    
    v_dot_a = v[0] * a_n[0] + v[1] * a_n[1] + v[2] * a_n[2]
    
    cos_v_dot_a = (1 - cos_phi) * v_dot_a

    return v[0] * cos_phi + a_n[0] * cos_v_dot_a - vxa_0 * sin_phi, v[1] * cos_phi + a_n[1] * cos_v_dot_a - vxa_1 * sin_phi, v[2] * cos_phi + a_n[2] * cos_v_dot_a - vxa_2 * sin_phi


