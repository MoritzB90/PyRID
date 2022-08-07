# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""


import numpy as np
import numba as nb

from ..math import transform_util as trf
from ..geometry.intersections_util import mesh_inside_box_test
from ..geometry.ray_march_util import ray_march_surface
from ..math import random_util as randu
from ..data_structures.cell_list_util import create_cell_list_points, reverse_cell_mapping

#%%

keytype = nb.types.int64
valuetype = nb.types.ListType(nb.types.float64)

listtype_W = nb.float64[:,:]

listtype_1 = nb.float64[:]
listtype_2 = nb.int64
listtype_3 = nb.types.ListType(nb.int64)
listtype_4 = nb.int64[:]
listtype_5 = nb.types.ListType(nb.int64[:])
    
#%%

@nb.njit
def random_point_in_triangle(p0,p1,p2):
    
    """Returns a random point uniformly distributed in a triangle.
    
    Parameters
    ----------
    p0 : `float64[3]`
        Triangle vertex 1
    p1 : `float64[3]`
        Triangle vertex 2
    p2 : `float64[3]`
        Triangle vertex 3
    
    Notes
    -----
    Random points, uniformly distributed inside a triangle can be sampled using an algorithm introduced by :cite:t:`Osada2002`:
    
    .. math::
        
        P(\\boldsymbol{r}) = (1-\\sqrt{\\mu_1})*\\boldsymbol{p}_0+(\\sqrt{\\mu_1}*(1-\\mu_2))*\\boldsymbol{p}_1+(\\mu_2*\\sqrt{\\mu_1})*\\boldsymbol{p}_2  ,
        
    where :math:`\\mu_1, \\mu_2` are random numbers between 0 and 1. :math:`\\boldsymbol{p}_0, \\boldsymbol{p}_1, \\boldsymbol{p}_2` are the three vertices of the triangle.
    
    Returns
    -------
    float64[3]
        Uniformly, random point in triangle
    
    """
    

    lamb = np.random.rand()
    mu = np.random.rand()
    
    pos = (1-np.sqrt(lamb))*p0+(np.sqrt(lamb)*(1-mu))*p1+(mu*np.sqrt(lamb))*p2  
    
    return pos

@nb.njit
def random_point_on_edge(p0,p1):
    
    """Returns a random point uniformly distributed on a line segment/edge between two vertices.
    
    Parameters
    ----------
    p0 : `float64[3]`
        Vertex 1
    p1 : `float64[3]`
        Vertex 2    
    
    Returns
    -------
    float64[3]
        Random point on edge
    
    """
    
    dl = np.random.rand()*(p1-p0)
    
    return p0+dl


@nb.njit
def point_on_sphere(radius):
    
    """Returns a random point uniformly distributed on the surface of a sphere with radius r.
    
    Parameters
    ----------
    radius : `float`
    
    
    Returns
    -------
    tuple(float, float, float)
        Uniformly distributed random point on sphere surface
    
    """
    
    u = 1-2*np.random.rand()
    
    phi = 2*np.pi*np.random.rand()
    
    x = radius*np.sqrt(1-u**2)*np.sin(phi)
    y = radius*np.sqrt(1-u**2)*np.cos(phi)
    z = radius*u
    return x,y,z

#%%

def evenly_on_sphere(n,r):
    
    """Approximately distributes n points evenly on the sphere surface.
    
    Parameters
    ----------
    n : `int64`
        Number of points.
    `r` : `float64`
        Sphere radius.
    
    Notes
    -----
    To distribute an arbitrary number of points evenly on a sphere surface, maximizing the minimum distance between all points, is not easily solved and only for a few cases exact theoretical solutions exist. For the cases n=1 to n=6 the known solutions to the Thompson problem are used, which is closesly related to the problem of distributing n points evenly on a sphere surface. N=1 and N=2 are trivial, for the other cases we find
    
    3. N = 3: points reside at vertices of equilateral triangle
    4. N = 4: points reside at the vertices of a regular tetrahedron.
    5. N = 5: points reside at vertices of a triangular dipyramid.
    6. N = 6: points reside at vertices of a regular octahedron.
    
    For n>6, points an approximation using the Fibonacci lattice is used (see e.g. http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/#more-3069, https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere). Thereby
    
    .. math::  
        
        \\begin{align*}
        x_i = R (\\cos(\\theta_i) \\sin(\\phi_i)) \\\\
        y_i = R (\\sin(\\theta_i) \\sin(\\phi_i)) \\\\
        z_i = R \\cos(\\phi_i) \\\\
        \\end{align*}
        
    where
        
    .. math::  

        \\begin{align*}
        G = \\frac{1 + \\sqrt{5}}{2} \\\\
        \\phi = \\arccos \\Big(1 - \\frac{2 i+1}{n} \\Big) \\\\
        \\theta = \\frac{2 \\pi i}{G}
        \\end{align*}
    
    Here, :math:`G` is the so called golden ratio, :math:`i = 1,2,...,n`, and :math:`R` is the sphere radius.
    
    Returns
    -------
    float64[n,3]
        Array of n points that are approximatels evenly distributed on a sphere surface.
    
    """
    
    
    if n==1:
        x=np.array([0.0])*r
        y=np.array([0.0])*r
        z=np.array([1.0])*r
    elif n==2: # points reside antipodal points.
        x=np.array([0.0, 0.0])*r
        y=np.array([0.0, 0.0])*r
        z=np.array([-1.0, 1.0])*r
    elif n==3: # points reside at vertices of equilateral triangle
        x=np.array([np.cos(np.pi/2),np.cos(np.pi*7/6),np.cos(np.pi*11/6)])*r
        y=np.array([np.sin(np.pi/2),np.sin(np.pi*7/6),np.sin(np.pi*11/6)])*r
        z=np.array([0.0,0.0,0.0])*r
    elif n==4: # points reside at the vertices of a regular tetrahedron.
        x=np.array([np.sqrt(8/9),-np.sqrt(2/9),-np.sqrt(2/9),0])*r
        y=np.array([0,np.sqrt(2/3),-np.sqrt(2/3),0])*r
        z=np.array([-1/3,-1/3,-1/3,1])*r
    elif n==5: # points reside at vertices of a triangular dipyramid.
        x=np.array([np.cos(np.pi/2),np.cos(np.pi*7/6),np.cos(np.pi*11/6),0,0])*r
        y=np.array([np.sin(np.pi/2),np.sin(np.pi*7/6),np.sin(np.pi*11/6),0,0])*r
        z=np.array([0.0,0.0,0.0,-1,1])*r
    elif n==6: # points reside at vertices of a regular octahedron.
        x=np.array([1,-1,0,0,0,0])*r
        y=np.array([0,0,1,-1,0,0])*r
        z=np.array([0,0,0,0,-1,1])*r
    else:
        
        goldenRatio = (1 + 5**0.5)/2
        i = np.arange(0, n)
        
        phi = np.arccos(1 - 2*(i+0.5)/n)
        theta = 2 *np.pi * i / goldenRatio
        
        x, y, z = (np.cos(theta) * np.sin(phi))*r, (np.sin(theta) * np.sin(phi))*r, np.cos(phi)*r
    
    x = np.hstack((0,x))
    y = np.hstack((0,y))
    z = np.hstack((0,z))
        
    return np.array([x,y,z]).T #x,y,z


#%%

@nb.njit
def release_molecules_boundary_2d(System, RBs, Particles):
    
    """Releases molecules into the volume at the simulation box boundary given an outside concentration of the different molecule types.
    
    Parameters
    ----------
    System : `object`
        Instance of System class
    RBs : `object`
       Instance of RBs class
    Particles : `object`
    
    
    Notes
    -----
    The average number of molecules that hit a boundary of edge length :math:`L` from one side within a time step :math:`\\Delta t` can be calculated from the molecule surface concentration :math:`C` and the average distance a diffusing molecule travels normal to an line :math:`l_{n}` within :math:`\\Delta t` :cite:p:`Kerr2008`:
    
    .. math::
    
       N_{hits} = \\frac{L l_{n}}{2 C},
    
    where
    
    .. math::
    
       l_{n} = \\sqrt{\\frac{4D\\Delta t}{\\pi}}.
       
    Here :math:`D = Tr(\\boldsymbol{D}_{xy}^{tt,b})/2` is the scalar translational diffusion constant.
    The boundary crossing of molecules can be described as a poisson process. As such, the number of molecules that cross the boundary each time step is drawn from a poisson distribution with a rate :math:`N_{hits}`.
    
    The normalized distance that a crossing molecule ends up away from the plane/boundary follows the following distribution :cite:p:`Kerr2008`:
        
    .. math::
        
        P(d\\tilde{x}) = 1-e^{-d\\tilde{x}^2}+\\sqrt{\\pi}*dx*\\text{erfc}(d\\tilde{x})
        
    The distance vector normal to the plane after the crossing can then be calculated from the diffusion length constant :math:`\\lambda` and the edges's normal vector :math:`\\hat{\\boldsymbol{n}}` by :math:`d\\boldsymbol{x} = \\lambda \\, d\\tilde{x} \\, \\hat{\\boldsymbol{n}} = \sqrt{4Dt} \\, d\\tilde{x} \\, \\hat{\\boldsymbol{n}}`.
    
    Now that the number of molecules and their distance away from the plane are determined, the molecules are distributed in the simualtion box. Since the diffusion along each dimension is independent we can simply pick a random point uniformly distributed on the respective edge:
    
    .. math::
        
        P(\\boldsymbol{r}) = p_0+\\mu*(p_1-p_0) ,
        
    where :math:`\\mu` is a random number between 0 and 1. :math:`\\boldsymbol{p}_0, \\boldsymbol{p}_1` are the two vertices of the edge.
    
    Note that, in general, here we neglect any interactions between the virtual molecules. Therefore, fixed concentration boundary conditions only result in the same inside and outside concentrations if no molecular in teractions are simulated.
    
    """
    
    
    quat = trf.rot_quaternion()
    
    for comp_id in range(1, System.n_comps+1):

        if System.Compartments[comp_id].box_overlap:

            for molecule in System.molecule_types:
                
                concentration = System.molecule_types[molecule].concentration_surface[comp_id]
                
                if concentration>0.0:
                    
                    l_perp = System.molecule_types[molecule].l_perp
                    length = System.Compartments[comp_id].border_length
                    
                    hits_in_dt = length*l_perp/2*concentration
                    
                    D = System.molecule_types[molecule].Dtrans_2d
                    lamb = np.sqrt(4*D*System.dt)
                    
                    
                    N_crossed = np.random.poisson(hits_in_dt)
                    
                    for i in range(N_crossed):
                            
                        tri_id_0 = randu.random_choice(cum_weights = System.Compartments[comp_id].border_2d['cum_edge_length'])
                        
                        
                        tri_id = System.Compartments[comp_id].border_2d[tri_id_0]['triangle_ids']
                        tri_direction = System.Compartments[comp_id].border_2d[tri_id_0]['direction_normal']
                        
                        edge_id = System.Compartments[comp_id].border_2d[tri_id_0]['edge_id']
                        
                        e0 = System.Mesh[tri_id]['edges'][edge_id][0]
                        e1 = System.Mesh[tri_id]['edges'][edge_id][1]
                        
                        p0 = System.vertices[e0]
                        p1 = System.vertices[e1] 
                        
                        xm = System.dx_rand.call(np.random.rand())*lamb
                        
                        pos = random_point_on_edge(p0,p1)
                        dX = xm*tri_direction
                        
                        
                        face_normal = System.Mesh[tri_id]['triangle_coord'][3]
                        
                        #Rotation into the plane of the triangle:
                        trf.quaternion_to_plane(face_normal, quat)
                        
                        # Random orientation in plane
                        trf.quaternion_random_axis_rot(quat, face_normal)
                        
                        
                        add_molecule_single(comp_id, System, RBs, Particles, pos, quat, tri_id, molecule)
                        
                        
                        RBs[RBs.slot-1]['dX'][:] = dX
                        
                        RBs.update_particle_pos_2D(System, Particles, RBs.slot-1)



#%%

@nb.njit#(cache=True)
def add_molecules(comp_id, System, RBs, Particles, pos, quaternion, triangle_ids, mol_type_ids, Types):

    """Places new molecules on teh surface of a compartment, given their positions, orientations, triangle ids and types.
    
    Parameters
    ----------
    Compartment_id : `int64`
        id of the compartment
    System : `object`
        Instance of System class
    RBs : `object`
        Instance of RBs class
    Particles : `object`
        Instance of Particles class   
    pos : `array_like`
        Positions of the molecule centers   
    quaternion : `array_like`
        Rotation quaternions of the molecules
    triangle_ids : `array_like`
        Triangle ids of the molecules
    mol_type_ids : `array_like`
        Type ids of the molecules
    Types : `array_like`
        Array of strings of the molecule names
    
    """
    
    for i in range(len(pos)):
        
        RBs.add_RB(str(Types[mol_type_ids[i]]), comp_id, System, Particles, 1)

        RBs.set_triangle(triangle_ids[i], RBs.slot-1)

        RBs.set_pos(Particles, System, pos[i], RBs.slot-1) 
        
        RBs.set_orientation_quat(Particles, quaternion[i], System, RBs.slot-1)
                
#%%

@nb.njit#(cache=True)
def add_molecule_single(Compartment_id, System, RBs, Particles, pos, quaternion, triangle_id, mol_type):
    
    """Places a single new molecule into a compartment, given its position, orientation, triangle id and type.
    
    Parameters
    ----------
    Compartment_id : `int64`
        id of the compartment
    System : `object`
        Instance of System class
    RBs : `object`
        Instance of RBs class
    Particles : `object`
        Instance of Particles class   
    pos : `array_like`
        Position of the molecul center   
    quaternion : `array_like`
        Rotation quaternion of the molecule
    triangle_ids : `array_like`
        Triangle ids of the molecules
    mol_type : `string`
        Molecule name
    
    """
    
    RBs.add_RB(str(mol_type), Compartment_id, System, Particles, 1)
    
    RBs.set_triangle(triangle_id, RBs.slot-1)
    
    RBs.set_pos(Particles, System, pos, RBs.slot-1) 
    
    RBs.set_orientation_quat(Particles, quaternion, System, RBs.slot-1)
    
    

#%%

@nb.njit
def normal(triangle_id, jitter, number, mol_types, System, center = None):
    
    """Distributes molecules spherically around a point on a mesh surface by a gaussian distribution and using a ray marching algorithm.
    
    Parameters
    ----------
    triangle_id : `int64`
        triangle id around whose centroid the molecules are distributed
    jitter : `float64`
        Standard deviation of the normal distribution
    number : `int64[:]`
        Number of molecules to distribute per type
    mol_types : `array_like`
        Array of strings of molecule names
    System : `object`
        Instance of System class
    
    
    Returns
    -------
    tuple(float64[N,3], float64[N,4], int64[N])
        Positions, Quaternions, and Molecule ids
    
    """
    
    # We pick a point on a triangle and then randomly draw distance vectors from a gaussian distribution.
    # Next, we simply raymarch along that vector on the mesh surface!
    
    if center is not None:
        origin = center
    else:
        origin = np.copy(System.Mesh[triangle_id]['triangle_centroid'])
    
    W_inv = System.Mesh[triangle_id]['triangle_coord'][1:].T
        
    positions = np.ones((np.sum(number),3))*origin
    triangle_ids = np.empty(np.sum(number))
    molecules_id = np.empty(np.sum(number), dtype = np.int64)
    
    quaternions = np.empty((np.sum(number), 4))
    
    j = 0
    dX = np.empty((np.sum(number),3)) 
    for _ in range(np.sum(number)):
        x_r = np.random.normal(0, jitter)
        y_r = np.random.normal(0, jitter)
        dX[j][0] = W_inv[0][0]*x_r + W_inv[0][1]*y_r
        dX[j][1] = W_inv[1][0]*x_r + W_inv[1][1]*y_r
        dX[j][2] = W_inv[2][0]*x_r + W_inv[2][1]*y_r
        
        tri_id = ray_march_surface(positions[j], quaternions[j], dX[j], triangle_id, System, update_quat = False)
        
        if tri_id == -1: 
            # distribution was not successfull (molecule got absorbed due to fixed concentration boundary condition)
            continue
        
        triangle_ids[j] = tri_id
    
        face_normal = System.Mesh[tri_id]['triangle_coord'][3]
        
        # Rotation into the plane of the triangle:
        trf.quaternion_to_plane(face_normal, quaternions[j])
        
        # Random orientation in plane
        trf.quaternion_random_axis_rot(quaternions[j], face_normal)
        
        j += 1
    
    
    N0 = 0
    for i,N in enumerate(number):
        molecules_id[N0:N0+N] = System.molecule_types[str(mol_types[i])].type_id
        N0 += N

    return positions[0:j], quaternions[0:j], molecules_id[0:j], triangle_ids[0:j]

#%%

@nb.njit
def point_in_disc(triangle_id, radius, System):
    
    """Returns a random point uniformly distributed inside a disc with radius r that lies within the plane of given mesh triangle.
    
    Parameters
    ----------
    triangle_id : `int64`
        Index of the triangle from where the point is distributed
    radius : `float`
        Radius of the disc
    System : `object`
        Instance of System class    
    
    
    Returns
    -------
    `float64[3]`
        Random point uniformly distributed in a disc.
    
    """
    
    W_inv = System.Mesh[triangle_id]['triangle_coord'][1:].T
    
    dX = np.empty(3) 

    phi = 2*np.pi*np.random.rand()
    r = radius*np.sqrt(np.random.rand())
    x_r = r*np.cos(phi)
    y_r = r*np.sin(phi)
    
    dX[0] = W_inv[0][0]*x_r + W_inv[0][1]*y_r
    dX[1] = W_inv[1][0]*x_r + W_inv[1][1]*y_r
    dX[2] = W_inv[2][0]*x_r + W_inv[2][1]*y_r
    
    return dX


@nb.njit
def trace_direction_vector(dX, origin, triangle_id, System):
    
    """Traces the path along a direction vector, and, depending on the boundary conditions and the presence of mesh compartments updates the direction vector. If the direction vector hits an absorptive boundary (fixed concentration boundary conditions) triangle id -1 is returned.
    
    Parameters
    ----------
    dX : `float64[3]`
        Direction vector
    origin : `float64[3]`
        Origin of the molecule
    triangle_id : `int64`
        Triangle id of the starting position/origin. 
    System : `object`
        Instance of System class

    
    Returns
    -------
    tuple(float64[3], float64[4], int64)
        Returns position, quaternion, and triangle_id of the new position. If path hit an absorptive boundary triangle_id = -1 is returned.
    
    
    """
    
    # We pick a point on a triangle and then randomly draw a distance vector.
    # Next, we simply raymarch along that vector on the mesh surface!
    
    position = np.copy(origin)
    
    # create a quaternion
    quaternion =  trf.rot_quaternion()
    
    # Ray march on surface without updating the molecules quaternion (not necessary since we assign a random orietnetation afterwards anyway):
    triangle_id = ray_march_surface(position, quaternion, dX, triangle_id, System, update_quat=False)
    
    #Rotation into the plane of the triangle:
    face_normal = System.Mesh[triangle_id]['triangle_coord'][3]
    trf.quaternion_to_plane(face_normal, quaternion)
                        
    # Random orientation in plane
    trf.quaternion_random_axis_rot(quaternion, face_normal)
        
    return position, quaternion, triangle_id
        
#%%


@nb.njit
def pds(Compartment, System, mol_types, number, facegroup = None, multiplier = 50):

    """Uniformly distributes molecules of different types on the mesh surface of a compartment using a blue noise/poisson disc sampling algorithm originaly introduced by :cite:t:`Corsini2012`.
    
    Parameters
    ----------
    Compartment : `object`
        Instance of Compartment class
    System : `object`
        Instance of System class
    mol_types : `array_like`
        Array of strings of molecule names
    number : `float64[:]`
        Number of molecules to distribute per given type
    facegroup : `string`
        Name of the facegroup to which to restrict the molecule distribution. Default = None
    multiplier : `int64`
        Specifies the multiplication factor by which the algorithm upscales the target number for the Monte Carlo sampling. Default = 50
    
    
    Returns
    -------
    tuple(float64[:,3], int64[:], float64[:,4])
        Returns arrays of the positions, molecule types, and rotation quaternions of the distributed moelcules.
    
    
    """
    
    radii = np.empty(len(mol_types))
    mol_type_ids = np.empty(len(mol_types), dtype = np.int64)
    for i, moltype in enumerate(mol_types):
        radii[i] = System.molecule_types[str(moltype)].radius_2d
        mol_type_ids[i] = System.molecule_types[str(mol_types[i])].type_id
        
    weights = radii**2/max(radii**2)
    
    points, points_type, face_ids, quaternion, count = poisson_disc_sampling_2D(Compartment, radii, mol_type_ids, weights, number, System, facegroup, multiplier)
    
    for i, moltype in enumerate(mol_types):
        print(str(count[i])+' molecules '+moltype+' distributed on surface of compartment '+Compartment.name)
    print('-------------------------------')
    
    return points, points_type, quaternion, face_ids      


#%%

@nb.njit
def mc(System, Compartment, mol_types, N, face_group = None):

    N_total = np.sum(N)
    
    mol_type_ids = np.empty(len(mol_types), dtype = np.int64)
    for i, moltype in enumerate(mol_types):
        mol_type_ids[i] = System.molecule_types[str(mol_types[i])].type_id
    
    points_type = np.empty(N_total, dtype = np.int64)
    N0 = 0
    for i,mol_id in enumerate(mol_type_ids):
        points_type[N0:N0+N[i]] = mol_id
        N0+=N[i]
        
        
    points, face_ids = monte_carlo_distribution_2D(Compartment, System, N_total, facegroup = face_group)
    
    
    quaternion = np.empty((N_total,4))
    for i in range(N_total):
        
        #Rotation into the plane of the triangle:
        face_normal = System.Mesh[face_ids[i]]['triangle_coord'][3]
        # create a quaternion
        q =  trf.rot_quaternion()
        # Rotation into the plane of the triangle:
        trf.quaternion_to_plane(face_normal, q)
        
        # Random orientation in plane
        trf.quaternion_random_axis_rot(q, face_normal)

        quaternion[i][:] = q
                
        
    print('Molecules distributed on surface of compartment '+Compartment.name)
    for j, name in enumerate(mol_types):
        print(name, ': ', N[j])
    print('-------------------------------')
    
    return points, points_type, quaternion, face_ids 


#%%


@nb.njit
def monte_carlo_distribution_2D(Compartment, System, N_total, facegroup = None):
    
    """Does Monte Carlo sampling for N points inside a compartment.
    
    Parameters
    ----------
    Compartment : `object`
        Instance of Compartment class
    System : `object`
        Instance of System class
    N_total : `int64`
        Total number of points to distribute
    facegroup : `string`
        Name of the facegroup to which to restrict the molecule distribution. Default = None    
    
    Returns
    -------
    float64[:,3]
        Array of positions
    
    """
    
    if facegroup is None:
        triangle_ids = Compartment.triangle_ids_exclTransp
    else:
        triangle_ids = Compartment.groups[str(facegroup)]['triangle_ids']
        
    A_max = max(System.Mesh[triangle_ids]['triangle_area'])
    
    points_MC = nb.typed.List.empty_list(listtype_1)
    face_ids_MC = nb.typed.List.empty_list(listtype_2)
    count = 0
    
    while count<N_total:
        
        face_id0 = np.random.randint(len(triangle_ids))
        face_id = triangle_ids[face_id0]
        
        val = np.random.rand()
        
        if val<System.Mesh[face_id]['triangle_area']/A_max:
        
            X = System.Mesh[face_id]['triangles']
            
            p0 = System.vertices[X[0]]
            p1 = System.vertices[X[1]]
            p2 = System.vertices[X[2]]
        
            lamb = np.random.rand()
            mu = np.random.rand()
            # Osada et al. 2002, Shape Distributions
            pos = (1-np.sqrt(lamb))*p0+(np.sqrt(lamb)*(1-mu))*p1+(mu*np.sqrt(lamb))*p2  
            
            outside = False
            for i in range(3):
                if pos[i]>System.box_lengths[i]/2:
                    outside = True
                if pos[i]<-System.box_lengths[i]/2:
                    outside = True
            
            if outside == False:
                                                        
                points_MC.append(pos)           
                face_ids_MC.append(face_id)
                
                count += 1
            
    return points_MC, face_ids_MC


@nb.njit
def poisson_disc_sampling_2D(Compartment, radii, mol_type_ids, weights, N, System, facegroup = None, multiplier = 50):
    
    """Uniformly distributes spheres of different radii inside a compartment using a blue noise/poisson disc sampling algorithm originaly introduced by :cite:t:`Corsini2012` to distribute points on triangulated mesh surfaces but that has been adapted here for volume distributions. PyRID also uses a variant of the original algorithm to distribute molecules on mesh surfaces.
    
    Parameters
    ----------
    Compartment : `object`
        Instance of Compartment class
    radii : `int64[:]`
        radii of the different sphere types
    mol_type_ids : `array_like`
        Array of molecule type ids
    weights : `float64[:]`
        Weights specifying how often the different sphere types are selected for distribution. Weights should be higher for large spheres since the probability of a successfull trial is lower than for small spheres.
    N : `float64[:]`
        Number of molecules to distribute per given type
    System : `object`
        Instance of System class
    facegroup : `string`
        Name of the facegroup to which to restrict the molecule distribution. Default = None
    multiplier : `int64`
        Specifies the multiplication factor by which the algorithm upscales the target number for the Monte Carlo sampling. Default = 100
    
    
    Raises
    ------
    ValueError('There is no overlap of the Compartment with the Simulation box! Unable to distribute points in Compartment!')
        Raised if the selected compartment does not overlap with the simulation box.
    
    Returns
    -------
    tuple(float64[:,3], int64[:], float64[:,4], int64)
        Returns arrays of the positions, types and rotations quaternions. Also return the total count of distributed spheres.
    
    Notes
    -----
    The algorithm is based on Corsini et al. 2012, Efficient and Flexible Sampling with Blue Noise Properties of Triangular Meshes.
    
    The algorithm consists of 10 steps:
        
    1. Generate a sample pool S using monte_carlo_distribution_3D().
    2. Divide space into cells and count the number of samples in each cell.
    3. Randomly select a cell weighted by the number of active samples in each cell (active sample: sample that is not yet occupied or deleted).
    4. Randomly select a sample from the selected cell.
    5. Randomly choose a particle type of radius Ri (weighted by the relative number of each type that we want to distribute).
    6. Check whether the distance of the selected sample to the neighboring samples that are already occupied is larger or equal to Ri+Rj.
    7. If True, accept the sample and add the molecule type and position to an occupied sample list. Next, delete all other samples within radius Ri, as these wont ever become occupied anyway.
    8. Update the number count of samples for the current cell.
    9. While the desired number of molecules is not reached, return to 3. However, set a maximum number of trials.
    10. If their are no active samples left before we reach the desired molecule number and the maximum number of trials, generate a new sample pool.
    
    """
    
    # Check if any part of the compartment is actually in the simulation box so we dont end
    # up in an infinite loop!
    Compartment_in_box = mesh_inside_box_test(Compartment, System)
    if Compartment_in_box == False:
        # print('Warning: There is no overlap of the Compartment with the Simulation box!')
        # return None, None, None, None, None
        raise ValueError('There is no overlap of the Compartment with the Simulation box! Unable to distribute points on Compartment Surface!')
    
    
    points = nb.typed.List.empty_list(listtype_1)
    quaternion = nb.typed.List.empty_list(listtype_1)
    points_type = nb.typed.List.empty_list(listtype_2)
    face_ids_psd = nb.typed.List.empty_list(listtype_2)
    count = np.zeros(len(radii), dtype = np.int64)
    
    #1. -----------------------------------------------------
    N_total = np.sum(N)*multiplier
    sample_points, face_ids = monte_carlo_distribution_2D(Compartment, System, N_total, facegroup)
    
    #2. -----------------------------------------------------
    rc = max(np.max(radii)*2, np.max(System.box_lengths)/25) # np.max(radii)*10
    
    active_cell_list, N_samples_cell, nonempty_cells, Ncell, cells_per_dim = create_cell_list_points(rc, sample_points, System)
    
    occupied_cell_list = nb.typed.List.empty_list(listtype_5)
    for _ in range(Ncell):

        occupied_cell_list.append(nb.typed.List.empty_list(listtype_4))
        
    #3.1 -----------------------------------------------------
    
    N_Trials = np.zeros(len(radii), dtype = np.int64)
    max_Trials = np.empty(len(radii), dtype = np.int64)
    for i in range(len(radii)):
        max_Trials[i]=int(50*N[i]/weights[i])
    
    pos_shift = np.empty(3, dtype = np.float64)
    cx_shift = 0
    cy_shift = 0
    cz_shift = 0
        
    while not np.all(count>=N) and np.any(N_Trials<max_Trials):
        
        #3.2 -----------------------------------------------------
        # Randomly select a cell weighted by the number of active samples. Yes, this can be done better...
        if len(nonempty_cells)>0:
            selected = False
            while selected==False:
                cell_id = nonempty_cells[np.random.randint(len(nonempty_cells))]
                val = np.random.rand()

                if val<N_samples_cell[cell_id]/np.max(N_samples_cell):            
                    
                    selected = True
                    
                    #3.3 Randomly select a sample from the selected cell.
                    sample_nr = np.random.randint(N_samples_cell[cell_id])
                    sample_id = active_cell_list[cell_id][sample_nr]
                    
                    pos_i = sample_points[sample_id]
        else:
            # generate new samples
            break
                
        #4. -----------------------------------------------------
        # Randomly choose a particle type of radius Ri (weighted by the relative number of each 
        # type that we want to distribute).
        dN = N-count #difference between goal and current number of molecules for each type
        dN_weight = dN/np.max(dN)
        
        selected = False
        while selected==False:
            type_id = np.random.randint(len(radii))

            if N_Trials[type_id]<max_Trials[type_id]:
                N_Trials[type_id] += 1
                val = np.random.rand()
                if val<dN_weight[type_id]:
                    val = np.random.rand()
                    if val<weights[type_id]:
                        selected = True
            
            if np.all(N_Trials>=max_Trials):
                selected = True
                     
        #5. -----------------------------------------------------
        # Check whether the distance of the selected sample to the neighboring samples that are 
        # already occupied is larger or equal to Ri+Rj.
        
        boundary_check = True
        if System.boundary_condition == 'repulsive':
            boundary_check = System.box_lengths[0]/2-radii[type_id]>pos_i[0]>=-(System.box_lengths[0]/2-radii[type_id]) and System.box_lengths[1]/2-radii[type_id]>pos_i[1]>=-(System.box_lengths[1]/2-radii[type_id]) and System.box_lengths[2]/2-radii[type_id]>pos_i[2]>=-(System.box_lengths[2]/2-radii[type_id])

        
        if boundary_check:
                
            fits = True
            cx,cy,cz = reverse_cell_mapping(cell_id, cells_per_dim)
            for cz_N in range(cz - 1, cz + 2):
                cz_shift = 0 + cells_per_dim[2] * (cz_N < 0) - cells_per_dim[2] * (cz_N >= cells_per_dim[2])
                pos_shift[2] = 0.0 - System.box_lengths[2] * (cz_N < 0) + System.box_lengths[2]*(cz_N >= cells_per_dim[2])
                for cy_N in range(cy - 1, cy + 2):
                    cy_shift = 0 + cells_per_dim[1] * (cy_N < 0) - cells_per_dim[1] * (cy_N >= cells_per_dim[1])
                    pos_shift[1] = 0.0 - System.box_lengths[1] * (cy_N < 0) + System.box_lengths[1] * (cy_N >= cells_per_dim[1])
                    for cx_N in range(cx - 1, cx + 2):
                        cx_shift = 0 + cells_per_dim[0] * (cx_N < 0) - cells_per_dim[0] * (cx_N >= cells_per_dim[0])
                        pos_shift[0] = 0.0 - System.box_lengths[0] * (cx_N < 0) + System.box_lengths[0] * (cx_N >= cells_per_dim[0])
                        
                        cell_temp = (cx_N + cx_shift) + (cy_N + cy_shift) * cells_per_dim[0] + (cz_N + cz_shift) * cells_per_dim[0] * cells_per_dim[1]
                        
                        for neighbour in occupied_cell_list[cell_temp]:
                            sample_id_j, type_id_j = neighbour
                            pos_j = sample_points[sample_id_j] 
                            
                            dx = pos_i[0]-(pos_j[0]+pos_shift[0])
                            dy = pos_i[1]-(pos_j[1]+pos_shift[1])
                            dz = pos_i[2]-(pos_j[2]+pos_shift[2])
                            
                            distance = np.sqrt(dx**2+dy**2+dz**2)
                                    
                            if distance<radii[type_id]+radii[type_id_j]:
                                fits=False
            
            # 6. -----------------------------------------------------
            # If True, accept the sample and add the molecule type and position to an occupied sample list. 
            # Next, delete all other samples within radius Ri, as these wont ever become occupied anyway.
            if fits==True:
                
                count[type_id] += 1
                
                occupied_cell_list[cell_id].append(np.array([sample_id, type_id], dtype = np.int64))
                
                points.append(pos_i)
                points_type.append(mol_type_ids[type_id])
                face_ids_psd.append(face_ids[sample_id])
                
                #Rotation into the plane of the triangle:
                face_normal = System.Mesh[face_ids[sample_id]]['triangle_coord'][3]
                # create a quaternion
                q =  trf.rot_quaternion()
                # Rotation into the plane of the triangle:
                trf.quaternion_to_plane(face_normal, q)
                
                # Random orientation in plane
                trf.quaternion_random_axis_rot(q, face_normal)
    
                quaternion.append(q)
                
                #7. -----------------------------------------------------
                # Update the number count of samples for the current cell.
            
    # print('N_Trials: ', N_Trials)
    # print('max_Trials: ', max_Trials)
    
    return points, points_type, face_ids_psd, quaternion, count
                
                

