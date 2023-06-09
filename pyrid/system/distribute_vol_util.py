# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numba as nb
import numpy as np
from ..geometry.mesh_util import point_triangle_distance
from ..geometry.intersections_util import point_inside_AABB_test, mesh_inside_box_test, point_inside_mesh_test
from ..geometry.ray_march_util import ray_march_volume
from ..math import random_util as randu
import math
from ..system.distribute_surface_util import random_point_in_triangle
from ..data_structures.cell_list_util import create_cell_list_mesh, create_cell_list_points, reverse_cell_mapping


#%%

@nb.njit
def release_molecules_boundary(System, RBs, Particles):
    
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
    The average number of molecules that hit a boundary of area :math:`A` from one side within a time step :math:`\\Delta t` can be calculated from the molecule concentration :math:`C` and the average distance a diffusing molecule travels normal to a plane :math:`l_{n}` within :math:`\\Delta t` :cite:p:`Kerr2008`:
    
    .. math::
    
       N_{hits} = \\frac{A l_{n}}{2 C},
    
    where
    
    .. math::
    
       l_{n} = \\sqrt{\\frac{4D\\Delta t}{\\pi}}.
       
    Here :math:`D = Tr(\\boldsymbol{D}^{tt,b})/3` is the scalar translational diffusion constant.
    The boundary crossing of molecules can be described as a poisson process. As such, the number of molecules that cross the boundary each time step is drawn from a poisson distribution with a rate :math:`N_{hits}`.
    
    The normalized distance that a crossing molecule ends up away from the plane/boundary follows the following distribution :cite:p:`Kerr2008`:
        
    .. math::
        
        P(d\\tilde{x}) = 1-e^{-d\\tilde{x}^2}+\\sqrt{\\pi}*dx*\\text{erfc}(d\\tilde{x})
        
    The distance vector normal to the plane after the crossing can then be calculated from the diffusion length constant :math:`\\lambda` and the plane's normal vector :math:`\\hat{\\boldsymbol{n}}` by :math:`d\\boldsymbol{x} = \\lambda \\, d\\tilde{x} \\, \\hat{\\boldsymbol{n}} = \sqrt{4Dt} \\, d\\tilde{x} \\, \\hat{\\boldsymbol{n}}`.
    
    In the case that a molecule enters the simulation box near to another boundary, e.g. of a mesh compartment, we may also want to account for the distance traveled parallel to the plane in order to correctly resolve collision with the mesh. However, currently PyRID does not account for this. For small intregration time steps and meshes that are further than :math:`\sqrt{4Dt}` away from the simulation box border, the error introduced should however be negligable.
    
    Now that the number of molecules and their distance away from the plane are determined, the molecules are distributed in the simualtion box. Since the diffusion along each dimension is independent we can simply pick a random point uniformly distributed on the respective plane. For triangulated mesh surfaces, triangles are picked randomly, weighted by their area. Sampling a uniformly distributed random point in a triangle is done by :cite:p:`Osada2002`
    
    .. math::
        
        P(\\boldsymbol{r}) = (1-\\sqrt{\\mu_1})*\\boldsymbol{p}_0+(\\sqrt{\\mu_1}*(1-\\mu_2))*\\boldsymbol{p}_1+(\\mu_2*\\sqrt{\\mu_1})*\\boldsymbol{p}_2  ,
        
    where :math:`\\mu_1, \\mu_2` are random numbers between 0 and 1. :math:`\\boldsymbol{p}_0, \\boldsymbol{p}_1, \\boldsymbol{p}_2` are the three vertices of the triangle.
    
    Note that, in general, here we neglect any interactions between the virtual molecules. Therefore, fixed concentration boundary conditions only result in the same inside and outside concentrations if no molecular in teractions are simulated.
    
    """
    
    
    if System.mesh == False:
        comp_id = 0
        pos = np.empty(3)
        axes = [0,1,2]
        sides = np.array([-1,1])
        for molecule in System.molecule_types:
            
            concentration = System.molecule_types[molecule].concentration[comp_id]
            
            if concentration>0.0:
                l_perp = System.molecule_types[molecule].l_perp
                area = System.box_area
                
                hits_in_dt = area*l_perp/2*concentration
                
                D = System.molecule_types[molecule].Dtrans
                lamb = np.sqrt(4*D*System.dt)
                
                N_crossed = np.random.poisson(hits_in_dt)
                
                for i in range(N_crossed):
                        
                    dim = randu.random_choice(cum_weights = System.box_border['cum_area'])
                    side = sides[np.random.randint(2)]
                    xm = System.dx_rand.call(np.random.rand())*lamb
                        
                    axes_ = [x for x in axes if x!=dim]
                    
                    pos[dim] = side*(System.box_lengths[dim]/2-xm)
                    pos[axes_[0]] = (1-2*np.random.rand())*System.box_lengths[axes_[0]]/2
                    pos[axes_[1]] = (1-2*np.random.rand())*System.box_lengths[axes_[1]]/2
                    quat = random_quaternion()
                        
                    add_molecule_single(comp_id, System, RBs, Particles, pos, quat, molecule)
                    
                    
    else:
        
        comp_id = 0
        axes = [0,1,2]
        sides = np.array([-1,1])
        for molecule in System.molecule_types:
            
            concentration = System.molecule_types[molecule].concentration[comp_id]
            
            if concentration>0.0:
                l_perp = System.molecule_types[molecule].l_perp
                area = System.box_area
                
                hits_in_dt = area*l_perp/2*concentration
                
                D = System.molecule_types[molecule].Dtrans
                lamb = np.sqrt(4*D*System.dt)
                
                
                N_crossed = np.random.poisson(hits_in_dt)
                
                for i in range(N_crossed):
                        
                    tri_id_0 = randu.random_choice(cum_weights = System.box_border['cum_area'])
                    tri_id = System.box_border[tri_id_0]['triangle_ids']
                    
                    triangle = System.vertices[System.Mesh[tri_id]['triangles']]
                    tri_normal = System.Mesh[tri_id]['triangle_coord'][3]
                    
                    p0 = triangle[0]
                    p1 = triangle[1]
                    p2 = triangle[2]
                    
                    xm = System.dx_rand.call(np.random.rand())*lamb
                    
                    pos = random_point_in_triangle(p0,p1,p2)
                    dX = -xm*tri_normal
                    
                    quat = random_quaternion()
                    
                    
                    add_molecule_single(comp_id, System, RBs, Particles, pos, quat, molecule)
                    
                    
                    RBs[RBs.slot-1]['dX'][:] = dX
                    
                    RBs.update_particle_pos(Particles, System, RBs.slot-1)
        
        for comp_id in range(1, System.n_comps+1):

            if System.Compartments[comp_id].box_overlap:

                for molecule in System.molecule_types:
                    
                    concentration = System.molecule_types[molecule].concentration[comp_id]
                    
                    if concentration>0.0:
                        
                        l_perp = System.molecule_types[molecule].l_perp
                        area = System.Compartments[comp_id].border_area
                        
                        hits_in_dt = area*l_perp/2*concentration
                        
                        D = System.molecule_types[molecule].Dtrans
                        lamb = np.sqrt(4*D*System.dt)
                        
                        N_crossed = np.random.poisson(hits_in_dt)
                        
                        for i in range(N_crossed):
                                
                            tri_id_0 = randu.random_choice(cum_weights = System.Compartments[comp_id].border_3d['cum_area'])
                            tri_id = System.Compartments[comp_id].border_3d[tri_id_0]['triangle_ids']
                            
                            triangle = System.vertices[System.Mesh[tri_id]['triangles']]
                            tri_normal = System.Mesh[tri_id]['triangle_coord'][3]
                            
                            p0 = triangle[0]
                            p1 = triangle[1]
                            p2 = triangle[2]
                            
                            xm = System.dx_rand.call(np.random.rand())*lamb
                            
                            pos = random_point_in_triangle(p0,p1,p2)
                            dX = -xm*tri_normal
                            
                            quat = random_quaternion()
                            
                            
                            add_molecule_single(comp_id, System, RBs, Particles, pos, quat, molecule)
                            
                            
                            RBs[RBs.slot-1]['dX'][:] = dX
                            
                            RBs.update_particle_pos(Particles, System, RBs.slot-1)

#%%

@nb.njit#(cache=True)
def random_quaternion():
    
    """Returns a unit quaternion representing a uniformly distributed random rotation.
    
    Notes
    -----
    The method can be found in K. Shoemake, Graphics Gems III by D. Kirk, pages 124-132 :cite:p:`Kirk2012` and on http://planning.cs.uiuc.edu/node198.html:
        
        .. math::
            
            \\boldsymbol{q} = [\\sqrt{1-u_1} \\, \\sin(2 \\pi u_2), \\sqrt{1-u_1} \\, \\cos(2 \\pi u_2), \\sqrt{u_1} \\, \\sin(2 \\pi u_3), \\sqrt{u_1} \\, \\cos(2 \\pi u_3)] ,
    
            
    where :math:`u_1,u_2,u_3 \\in [0,1]` are uniformly distributed random numbers. :math:`\\boldsymbol{q}` is a uniformly distributed, random rotation quaternion.
                                                                                                                                                              
    Returns
    -------
    `float64[4]`
        Uniformly distributed, random rotation quaternion
    
    """
    
    
    u1,u2,u3 = np.random.rand(), np.random.rand(), np.random.rand()
    
    return np.array([np.sqrt(1-u1)*np.sin(2*np.pi*u2), np.sqrt(1-u1)*np.cos(2*np.pi*u2), np.sqrt(u1)*np.sin(2*np.pi*u3), np.sqrt(u1)*np.cos(2*np.pi*u3)])


@nb.njit#(cache=True)
def random_quaternion_tuple():
    
    """Returns a unit quaternion representing a uniformly distributed random rotation.
    
    Notes
    -----
    The method can be found in K. Shoemake, Graphics Gems III by D. Kirk, pages 124-132 :cite:p:`Kirk2012` and on http://planning.cs.uiuc.edu/node198.html:
        
        .. math::
            
            \\boldsymbol{q} = [\\sqrt{1-u_1} \\, \\sin(2 \\pi u_2), \\sqrt{1-u_1} \\, \\cos(2 \\pi u_2), \\sqrt{u_1} \\, \\sin(2 \\pi u_3), \\sqrt{u_1} \\, \\cos(2 \\pi u_3)] ,
    
            
    where :math:`u_1,u_2,u_3 \\in [0,1]` are uniformly distributed random numbers. :math:`\\boldsymbol{q}` is a uniformly distributed, random rotation quaternion.
                                                                                                                                                              
    Returns
    -------
    `tuple[4]`
        Uniformly distributed, random rotation quaternion
    
    """
    
    # Based on K. Shoemake. Uniform random rotations. In D. Kirk, editor, Graphics Gems III, pages 124-132. Academic, New York, 1992. http://planning.cs.uiuc.edu/node198.html
    
    u1,u2,u3 = np.random.rand(), np.random.rand(), np.random.rand()
    
    return np.sqrt(1-u1)*np.sin(2*np.pi*u2), np.sqrt(1-u1)*np.cos(2*np.pi*u2), np.sqrt(u1)*np.sin(2*np.pi*u3), np.sqrt(u1)*np.cos(2*np.pi*u3)


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

@nb.njit
def point_in_sphere_simple(radius):
    
    """Returns a random point uniformly distributed in the volume of a sphere with radius r.
    
    Parameters
    ----------
    radius : `float`
    
    
    Returns
    -------
    tuple(float, float, float)
        Uniformly distributed random point in sphere volume
    
    """
    
    u = 1-2*np.random.rand()
    
    phi = 2*np.pi*np.random.rand()
    
    D = radius*np.random.rand()**(1/3)
    
    x = D*np.sqrt(1-u**2)*np.sin(phi)
    y = D*np.sqrt(1-u**2)*np.cos(phi)
    z = D*u
    return x,y,z

#%%

@nb.njit#(cache=True)
def add_molecules(Compartment_id, System, RBs, Particles, pos, quaternion, mol_type_ids, mol_types):
    
    """Places new molecules into a compartment, given their positions, orientations and types.
    
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
    mol_type_ids : `array_like`
        Type ids of the molecules
    mol_types : `array_like`
        Array of strings of the molecule names
    
    """

    for i in range(len(pos)):
        
        RBs.add_RB(str(mol_types[mol_type_ids[i]]), Compartment_id, System, Particles, 0)
        
        RBs.set_pos(Particles, System, pos[i], RBs.slot-1) 
        
        RBs.set_orientation_quat(Particles, quaternion[i], System, RBs.slot-1)



@nb.njit#(cache=True)
def add_molecule_single(Compartment_id, System, RBs, Particles, pos, quaternion, mol_type):
    
    """Places a single new molecule into a compartment, given its position, orientation and type.
    
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
    mol_type : `string`
        Molecule name
    
    """
    
    RBs.add_RB(str(mol_type), Compartment_id, System, Particles, 0)
    
    RBs.set_pos(Particles, System, pos, RBs.slot-1) 
    
    RBs.set_orientation_quat(Particles, quaternion, System, RBs.slot-1)
        

#%%


@nb.njit
def normal(origin, jitter, number, mol_types, System):
    
    """Distributes molecules spherically around a point by a gaussian distribution.
    
    Parameters
    ----------
    origin : `float64[3]`
        Origin of the distribution
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
    
    pos = np.ones((np.sum(number),3))*origin
    dX = np.random.normal(0, jitter, (np.sum(number),3))
    quaternion = np.empty((np.sum(number),4))
    
    j = 0
    for _ in range(np.sum(number)):
        
        if System.mesh:
            
            success = ray_march_volume(pos[j], dX[j], System)
            
        
        else:
            
            if System.boundary_condition_id == 0:
                
                pos[j][0] += dX[j][0]
                pos[j][1] += dX[j][1]
                pos[j][2] += dX[j][2]
                
            
            elif System.boundary_condition_id == 2:
                
                pos[j][0] += dX[j][0]
                pos[j][1] += dX[j][1]
                pos[j][2] += dX[j][2]
                
                success = point_inside_AABB_test(pos[j], System.AABB)
                
            
            if System.boundary_condition_id == 1:
                
                
                for dim in range(3):
                    if pos[j][dim]+dX[j][dim]>System.box_lengths[dim]/2:
                        
                        boundary = System.box_lengths[dim]/2
                        t = (boundary - pos[j][dim])/dX[j][dim]
                        pos[j][:] += t*dX[j]
                        dX[j][:] -= t*dX[j]
                        dX[j][0] *= -1
                        
                    elif pos[j][dim]+dX[j][dim]<-System.box_lengths[dim]/2:
                        
                        boundary = -System.box_lengths[dim]/2
                        t = (boundary - pos[j][dim])/dX[j][dim]
                        pos[j][:] += t*dX[j]
                        dX[j][:] -= t*dX[j]
                        dX[j][0] *= -1
                
                pos[j][0] += dX[j][0]
                pos[j][1] += dX[j][1]
                pos[j][2] += dX[j][2]
                        
        if success:
            
            q1,q2,q3,q4 = random_quaternion_tuple()
            quaternion[j][0] = q1
            quaternion[j][1] = q2
            quaternion[j][2] = q3
            quaternion[j][3] = q4
            
            j += 1
    
    
    molecules_id = np.empty(np.sum(number), dtype = np.int64)
    N0 = 0
    for i,N in enumerate(number):
        molecules_id[N0:N0+N] = System.molecule_types[str(mol_types[i])].type_id
        N0 += N

    return pos[0:j], quaternion[0:j], molecules_id[0:j]
                

@nb.njit
def random_direction_sphere(sphere_radius):
    
    """Returns a random direction vector uniformly distributed in the volume of a sphere with radius r.
    
    Parameters
    ----------
    sphere_radius : `float64`
        Radius of the sphere
    
    
    Returns
    -------
    float64[3]
        Direction vector
    
    """
    
    u = 1-2*np.random.rand()
    
    phi = 2*np.pi*np.random.rand()
    
    D = sphere_radius*np.random.rand()**(1/3)
    
    dX = np.empty(3)
    
    dX[0] = D*np.sqrt(1-u**2)*np.sin(phi)
    dX[1] = D*np.sqrt(1-u**2)*np.cos(phi)
    dX[2] = D*u
    
    return dX


@nb.njit
def random_direction_Halfsphere(sphere_radius, normal):
    
    """Returns a random direction vector uniformly distributed in the volume of a half-sphere with radius r, splitted in half in direction of a given plane normal vector.
    
    Parameters
    ----------
    sphere_radius : `float64`
        Radius of the sphere
    normal : `float64[3]`
        Normal vector of the plane splitting the sphere in half
    
    
    Notes
    -----
    First a uniformly distributed random vector :math:`d\\boldsymbol{X}` that sites somewhere within the full sphere is drawn.
    Next, we test whether the vector points into the same direction as the plane normal vector :math:`\\hat{\\boldsymbol{n}}` via their dot product.
    If the dot product is negative, the direction vector is reflected at the plane by :math:`d\\boldsymbol{X}_{refl} = d\\boldsymbol{X} -2 d\\boldsymbol{X} \cdot \\hat{\\boldsymbol{n}}`.
    
    Returns
    -------
    float64[3]
        Direction vector
    
    """
    
    u = 1-2*np.random.rand()
    
    phi = 2*np.pi*np.random.rand()
    
    D = sphere_radius*np.random.rand()**(1/3)
    
    dX = np.empty(3)
    
    dX[0] = D*np.sqrt(1-u**2)*np.sin(phi)
    dX[1] = D*np.sqrt(1-u**2)*np.cos(phi)
    dX[2] = D*u
    
    dX_dot_n = np.dot(dX, normal)
    if dX_dot_n<0.0:
        return dX-2*dX_dot_n*normal
    
    return dX


@nb.njit
def trace_direction_vector(dX, origin, pos, System):
    
    """Traces the path along a direction vector, and, depending on the boundary conditions and the presence of mesh compartments updates the direction vector. If the direction vector hits an absorptive boundary (fixed concentration boundary conditions), returns False, indicating that the corresponding molecule needs to be deleted.
    
    Parameters
    ----------
    dX : `float64[3]`
        Direction vector
    origin : `float64[3]`
        Origin of the molecule
    pos : `float64[3]`
        New position/origin of the moelcule is saved in this array. 
    System : `object`
        Instance of System class

    
    Returns
    -------
    bool
        Indicates whether the position tracing was successful or the molecule hit an absorptive boundary.
    
    """
    
    
    if System.mesh:
        
        pos[0] = origin[0]
        pos[1] = origin[1]
        pos[2] = origin[2]
        
        success = ray_march_volume(pos, dX, System)
        
        return success
    
    else:
        
        if System.boundary_condition_id == 0:
            
            pos[0] = origin[0] + dX[0]
            pos[1] = origin[1] + dX[1]
            pos[2] = origin[2] + dX[2]
            
            return True
        
        elif System.boundary_condition_id == 2:
            
            pos[0] = origin[0] + dX[0]
            pos[1] = origin[1] + dX[1]
            pos[2] = origin[2] + dX[2]
            
            inside_box = point_inside_AABB_test(pos, System.AABB)
            
            return inside_box
        
        if System.boundary_condition_id == 1:
            
            pos[0] = origin[0]
            pos[1] = origin[1]
            pos[2] = origin[2]
            
            for dim in range(3):
                if pos[dim]+dX[dim]>System.box_lengths[dim]/2:
                    
                    boundary = System.box_lengths[dim]/2
                    t = (boundary - pos[dim])/dX[dim]
                    pos[:] += t*dX
                    dX[:] -= t*dX
                    dX[0] *= -1
                    
                elif pos[dim]+dX[dim]<-System.box_lengths[dim]/2:
                    
                    boundary = -System.box_lengths[dim]/2
                    t = (boundary - pos[dim])/dX[dim]
                    pos[:] += t*dX
                    dX[:] -= t*dX
                    dX[0] *= -1
            
            pos[0] += dX[0]
            pos[1] += dX[1]
            pos[2] += dX[2]
                    
            
            return True
        
        
            
#%%


@nb.njit#(cache=True)
def pds(Compartment, System, mol_types, number, clustering_factor=1, max_trials=100):
    
    """Distributes a number of molecules of different types inside the volume of a compartment using a poisson disc sampling algorithm.
    
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
    clustering factor : `int64`
        Determines how much the molecules are spread inside the compartment. Higher values are necessary for uniform distributions at low density. Default = 1
    max_trials : `int64`
        Maximum number of trials to find a new position around a given active point in the poisson disc sampling algorithm.
    
    Returns
    -------
    tuple(float64[:,3], int64[:], float64[:,4])
        Returns arrays of the positions, molecule types, and rotation quaternions of the distributed moelcules.
    
    """
    
    radii = np.empty(len(mol_types))
    mol_type_ids = np.empty(len(mol_types), dtype = np.int64)
    for i, moltype in enumerate(mol_types):
        radii[i] = System.molecule_types[str(moltype)].radius
        mol_type_ids[i] = System.molecule_types[str(mol_types[i])].type_id
        
    weights = radii/max(radii)
    points, points_type, quaternion, count = poisson_disc_sampling(Compartment, System, radii, mol_type_ids, number, weights, clustering_factor, max_trials)
    
    
    print('Molecules distributed in volume of compartment '+Compartment.name)
    for j, name in enumerate(mol_types):
        print(name, ': ', count[j])
    print('-------------------------------')
    
    return points, points_type, quaternion


    

@nb.njit#(cache=True)    
def poisson_disc_sampling(Compartment, System, radii, mol_type_ids, N, weights, clustering_factor, max_trials):
    
    """Poisson disc sampling algorithm for different sized spheres, accounting for different boundary conditions and mesh compartment boundaries. The algorithm is based on :cite:t:`Bridson2007`. 
    
    Parameters
    ----------
    Compartment : `object`
        Instance of Compartment class
    System : `object`
        Instance of System class
    radii : `float64[:]`
        Radii of the different sphere types
    mol_type_ids : `array_like`
        Type ids of the different spheres
    N : `float64[:]`
        Number of spheres to distribute per given type
    clustering factor : `int64`
        Determines how much the spheres are spread inside the compartment. Higher values are necessary for uniform distributions at low density. Default = 1
    max_trials : `int64`
        Maximum number of trials to find a new position around a given active point in the poisson disc sampling algorithm.
    
    Raises
    ------
    ValueError('There is no overlap of the Compartment with the Simulation box! Unable to distribute points in Compartment!')
        Raised if the selected compartment does not overlap with the simulation box.
    
    Returns
    -------
    tuple(float64[:,3], int64[:], float64[:,4], int64)
        Returns arrays of the positions, types and rotations quaternions. Also return the total count of distributed spheres.
    
    """
    

    
    if System.mesh:
        # Check if any part of the compartment is actually in the simulation box so we dont end
        # up in an infinite loop!
        Compartment_in_box = mesh_inside_box_test(Compartment, System)
        if Compartment_in_box == False:

            raise ValueError('There is no overlap of the Compartment with the Simulation box! Unable to distribute points in Compartment!')
            
    
    pos_tri = np.empty(3, dtype = np.float64)
    
    if System.mesh==True: 
        print("debugging Cell division <3")
        rc = max(np.max(radii)*2, np.max(System.box_lengths)/25)
        cells_per_dim = (Compartment.box_lengths / rc).astype(np.int64)
        print(radii)
        print(System.box_lengths)
        print(rc)
        print(Compartment.box_lengths)
        print(cells_per_dim)
        if np.any(cells_per_dim<3):
            print('error: Cell division <3')
        cell_length_per_dim = Compartment.box_lengths / cells_per_dim
        
        triangle_ids = np.arange(System.N_triangles)
        
        CellList, AABB_centers = create_cell_list_mesh(System, Compartment, cells_per_dim, cell_length_per_dim, triangle_ids, False, True) # np.max(System.box_lengths)/50
    
    dim = System.dim
    
    origin_x = Compartment.AABB[0][0]
    origin_y = Compartment.AABB[0][1]
    origin_z = Compartment.AABB[0][2]
    
    width = Compartment.AABB[1][0]-Compartment.AABB[0][0]
    if width > System.box_lengths[0]:
        width = System.box_lengths[0]
    height = Compartment.AABB[1][1]-Compartment.AABB[0][1]
    if height > System.box_lengths[1]:
        height = System.box_lengths[1]
    depth = Compartment.AABB[1][2]-Compartment.AABB[0][2]
    if depth > System.box_lengths[2]:
        depth = System.box_lengths[2]
        
    number_types = len(radii)
    diameter = 2*np.min(radii)
    searchRadius = clustering_factor*(max((width, height, depth))/2)
    if searchRadius<diameter:
        searchRadius = diameter
    # print('searchRadius: ', searchRadius)
    w0 = diameter/np.sqrt(dim)
    cells_per_dim_pds = int(width/w0)
    w = width/cells_per_dim_pds

    
    #Step 0
    columns = int(width/w)
    rows = int(height/w)
    aisles = int(depth/w)
    grid = -np.ones(rows*columns*aisles, dtype = np.int64)
    grid_type = -np.ones(rows*columns*aisles, dtype = np.int64)
    active = nb.typed.List()
    points = nb.typed.List()
    quaternion = nb.typed.List()
    ptype = nb.typed.List()
    type_list = np.arange(len(radii))
    points_ptype = nb.typed.List()
    count = np.zeros(len(radii), dtype = np.int64)
    #Step 1
    
    
    seed_found = False
    count_loop = 0
    while seed_found==False:
        count_loop+=1
        
        if count_loop>10000:
            raise ValueError('Canceled while loop: Not able to place moelcule seed in poisson_disc_sampling()')
            
        x = (np.random.rand())*width
        y = (np.random.rand())*height
        z = (np.random.rand())*depth
        i = int(x/w)
        j = int(y/w)
        k = int(z/w)
        pos = np.array([x,y,z])
        type_1_idx = np.random.randint(number_types)
        
        if System.mesh==True and Compartment.name == 'System':
            point_on_correct_side = True
            for comp_id in System.Compartments:
                Compartment_temp = System.Compartments[comp_id]
                point_on_correct_side_temp = point_inside_mesh_test(Compartment_temp.triangle_ids, System.Mesh, System.vertices, pos+Compartment_temp.origin, 1e0)==False
                if point_on_correct_side_temp == False:
                    point_on_correct_side = False
        elif System.mesh==True and Compartment.name != 'System':
            point_on_correct_side = point_inside_mesh_test(Compartment.triangle_ids, System.Mesh, System.vertices, pos+Compartment.origin, 1e0)
        else:
            point_on_correct_side = True        
        
        boundary_check = False
        if System.boundary_condition == 'repulsive':
            boundary_check = width-radii[type_1_idx]>pos[0]>=radii[type_1_idx] and height-radii[type_1_idx]>pos[1]>=radii[type_1_idx] and depth-radii[type_1_idx]>pos[2]>=radii[type_1_idx]
        if System.boundary_condition == 'periodic' or System.boundary_condition == 'fixed concentration':
            boundary_check = width>pos[0]>=0.0 and height>pos[1]>=0.0 and depth>pos[2]>=0.0
        
        if boundary_check and point_on_correct_side:         
            if System.mesh==True:              
                # !!! Here we use a loose cell grid!
                cx = int((pos[0]) / cell_length_per_dim[0])
                cy = int((pos[1]) / cell_length_per_dim[1])
                cz = int((pos[2]) / cell_length_per_dim[2])
        
                # Determine cell in 3D volume for i-th particle
                cell = cx + cy * cells_per_dim[0] + cz * cells_per_dim[0] * cells_per_dim[1]
            
                min_dist = 1e6
                triangles_list = CellList.get_triangles(cell)
                if len(triangles_list)>0:
                    for Tri_idx in triangles_list:
                        
                        triangle = System.Mesh[Tri_idx]['triangles']
                    
                        p0 = System.vertices[triangle[0]]
                        p1 = System.vertices[triangle[1]]
                        p2 = System.vertices[triangle[2]]
        
                        dist, region = point_triangle_distance(p0,p1,p2,pos+Compartment.origin, pos_tri)
                        if dist<min_dist:
                            min_dist = dist
                no_overlap = min_dist >= radii[type_1_idx]
            else:
                no_overlap = True    
                
            if no_overlap:
                seed_found = True
    
    
    active.append(nb.typed.List([pos,np.ones(len(radii))]))
    points.append(pos)
    quaternion.append(random_quaternion())
    ptype.append(type_1_idx)
    points_ptype.append(mol_type_ids[type_1_idx])
    grid[i+j*columns+k*columns*rows] = len(points)-1
    grid_type[i+j*columns+k*columns*rows] = type_1_idx
    count[type_1_idx] += 1
    

    trials = 0
    while not np.all(count>=N) and len(active)>0:
        trials += 1
        
        rand_idx = np.random.randint(0,len(active))
        pos = active[rand_idx][0]
        type_1_idx = ptype[rand_idx]
        
        #====================================================
        idx = randu.random_choice(weights = weights[active[rand_idx][1]==1])
        
        # probDensity = np.zeros(len(weights[active[rand_idx][1]==1])+1)
        # sort_indices = randu.probDens_2(weights[active[rand_idx][1]==1], probDensity)
        # idx = randu.binary_search(probDensity, sort_indices, np.random.rand())
        #====================================================
        
        type_2_idx = type_list[active[rand_idx][1]==1][idx]
        
        found = False
        if count[type_2_idx]<N[type_2_idx]:
            for n in range(max_trials):
                a = np.random.rand()*2*np.pi
                b = np.arccos(1-2*np.random.rand())
                m = np.random.uniform(radii[type_1_idx]+radii[type_2_idx],(radii[type_1_idx]+radii[type_2_idx])+searchRadius)
                
                offsetX = m*np.cos(a)*np.sin(b)
                offsetY = m*np.sin(a)*np.sin(b)
                offsetZ = m*np.cos(b)
                sample = np.array([offsetX, offsetY, offsetZ])
                sample += pos
                

                if System.mesh==True and Compartment.name == 'System':
                    point_on_correct_side = True
                    for comp_id in System.Compartments:
                        Compartment_temp = System.Compartments[comp_id]
                        point_on_correct_side_temp = point_inside_mesh_test(Compartment_temp.triangle_ids, System.Mesh, System.vertices, sample+Compartment_temp.origin, 1e0)==False
                        if point_on_correct_side_temp == False:
                            point_on_correct_side = False
                elif System.mesh==True and Compartment.name != 'System':
                    point_on_correct_side = point_inside_mesh_test(Compartment.triangle_ids, System.Mesh, System.vertices, sample+Compartment.origin, 1e0)
                else:
                    point_on_correct_side = True  
                
                
                boundary_check = False
                if System.boundary_condition == 'repulsive':
                    boundary_check = width-radii[type_2_idx]>sample[0]>=radii[type_2_idx] and height-radii[type_2_idx]>sample[1]>=radii[type_2_idx] and depth-radii[type_2_idx]>sample[2]>=radii[type_2_idx]
                if System.boundary_condition == 'periodic' or System.boundary_condition == 'fixed concentration':
                    boundary_check = width>sample[0]>=0.0 and height>sample[1]>=0.0 and depth>sample[2]>=0.0
                
                if boundary_check and point_on_correct_side: 


                    if System.mesh==True: 
                        # !!! Here we use a loose cell grid!
                        cx = int((sample[0]) / cell_length_per_dim[0])
                        cy = int((sample[1]) / cell_length_per_dim[1])
                        cz = int((sample[2]) / cell_length_per_dim[2])
                
                        # Determine cell in 3D volume for i-th particle
                        cell = cx + cy * cells_per_dim[0] + cz * cells_per_dim[0] * cells_per_dim[1]
                        
                    
                        min_dist = 1e6
                        triangles_list = CellList.get_triangles(cell)
                        if len(triangles_list)>0:
                            for Tri_idx in triangles_list:
                                
                                triangle = System.Mesh[Tri_idx]['triangles']
                            
                                p0 = System.vertices[triangle[0]]
                                p1 = System.vertices[triangle[1]]
                                p2 = System.vertices[triangle[2]]
                                
                                dist, region = point_triangle_distance(p0,p1,p2,sample+Compartment.origin, pos_tri)
                                if dist<min_dist:
                                    min_dist = dist
                        no_overlap = min_dist >= radii[type_2_idx]
                    else:
                        no_overlap = True
                        
                            
                    if no_overlap:
                        
                        col = int(sample[0]/w)
                        row = int(sample[1]/w)
                        aisle = int(sample[2]/w)
                        
                        ok = True
                        NNrange = math.ceil((np.max(radii)+radii[type_2_idx])/w) 
                        
                        for k in range(aisle-NNrange,aisle+NNrange+1):
                            k_shift = 0 + aisles * (k < 0) - aisles * (k >= aisles)
                            k_pos_shift = 0.0 - depth * (k < 0) + depth*(k >= aisles)
                            for j in range(row-NNrange,row+NNrange+1):
                                j_shift = 0 + rows * (j < 0) - rows * (j >= rows)
                                j_pos_shift = 0.0 - height * (j < 0) + height*(j >= rows)
                                for i in range(col-NNrange,col+NNrange+1):
                                    i_shift = 0 + columns * (i < 0) - columns * (i >= columns)
                                    i_pos_shift = 0.0 - width * (i < 0) + width*(i >= columns)
                            
                                
                                    index = (i + i_shift) + (j + j_shift) * columns + (k + k_shift) * columns * rows
                                    neighbour = grid[index]
                                    neighbour_type = grid_type[index]
                                    if neighbour!=-1:
                                        dx = sample[0] - (points[neighbour][0] + i_pos_shift)
                                        dy = sample[1] - (points[neighbour][1] + j_pos_shift)
                                        dz = sample[2] - (points[neighbour][2] + k_pos_shift)
                                        d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                                        if d<(radii[neighbour_type]+radii[type_2_idx]):
                                            ok = False

                                        
                        if ok:
                            found = True
                            points.append(sample)
                            points_ptype.append(mol_type_ids[type_2_idx])
                            active.append(nb.typed.List([sample,np.ones(len(radii))]))
                            ptype.append(type_2_idx)
                            count[type_2_idx] += 1
                            grid[col+row*columns+aisle*columns*rows] = len(points)-1
                            grid_type[col+row*columns+aisle*columns*rows] = type_2_idx
                            
                            q = random_quaternion()
                            quaternion.append(q)
                            
                            break
                        
                    
        if not found:
            active[rand_idx][1][type_2_idx] = 0
            
        if np.sum(active[rand_idx][1]) == 0:
            del active[rand_idx]
            del ptype[rand_idx]
        
    
    for i in range(len(points)):
        points[i][0] += origin_x
        points[i][1] += origin_y
        points[i][2] += origin_z
        
    return points, points_ptype, quaternion, count


#%%

@nb.njit
def mc(System, Compartment, mol_types, N):

    """Does Monte Carlo sampling for N molecules inside a compartment returning vectors for the molecule positions, types and orientations.
    
    Parameters
    ----------
    Compartment : `object`
        Instance of Compartment class
    mol_types : `list of strings`
        List of molecule types to distribute.
    N : `int64[:]`
        Total number of molecules to distribute per molecule type.
    
    
    Returns
    -------
    tuple(float64[:,3], int64[:], float64[:,4])
        molecule positions, molecule types, molecule orientations in quaternion representation
    
    """

    mol_type_ids = np.empty(len(mol_types), dtype = np.int64)
    for i, moltype in enumerate(mol_types):
        mol_type_ids[i] = System.molecule_types[str(mol_types[i])].type_id
        
    AABB = np.copy(Compartment.AABB)
    for dim in range(3):
        if AABB[0][dim]<System.AABB[0][dim]:
            AABB[0][dim] = System.AABB[0][dim]
        if AABB[1][dim]>System.AABB[1][dim]:
            AABB[1][dim] = System.AABB[1][dim]
    
    box_lengths = AABB[1]-AABB[0]
    
    points = nb.typed.List()
    points_type = nb.typed.List()
    quaternion = nb.typed.List()
    
    count = np.zeros(len(mol_type_ids))
    for i,mol_id in enumerate(mol_type_ids):
        
        while count[i]<N[i]:
            
            pos_MC = AABB[0] + np.random.rand(3)*box_lengths
        
            if System.mesh==True and Compartment.name == 'System':
                point_on_correct_side = True
                for comp_id in System.Compartments:
                    Compartment_temp = System.Compartments[comp_id]
                    point_on_correct_side_temp = point_inside_mesh_test(Compartment_temp.triangle_ids, System.Mesh, System.vertices, pos_MC, 1e0)==False
                    if point_on_correct_side_temp == False:
                        point_on_correct_side = False
            elif System.mesh==True and Compartment.name != 'System':
                point_on_correct_side = point_inside_mesh_test(Compartment.triangle_ids, System.Mesh, System.vertices, pos_MC, 1e0)
            else:
                point_on_correct_side = True   
                
    
            if point_on_correct_side: 
                count[i] += 1
                points.append(pos_MC)
                points_type.append(mol_id)
                quaternion.append(random_quaternion())
                
    print('Molecules distributed in volume of compartment '+Compartment.name)
    for j, name in enumerate(mol_types):
        print(name, ': ', count[j])
    print('-------------------------------')
    
    return points, points_type, quaternion
    
    
    
#%%


listtype_1 = nb.float64[:]
listtype_2 = nb.int64
listtype_3 = nb.types.ListType(nb.int64)
listtype_4 = nb.int64[:]
listtype_5 = nb.types.ListType(nb.int64[:])


@nb.njit
def pds_uniform(Compartment, System, mol_types, N, multiplier = 100):
    
    """Uniformly distributes molecules of different types inside a compartment using a blue noise/poisson disc sampling algorithm originaly introduced by :cite:t:`Corsini2012` to distribute points on triangulated mesh surfaces but that has been adapted here for volume distributions. PyRID also uses a variant of the original algorithm to distribute molecules on mesh surfaces.
    
    Parameters
    ----------
    Compartment : `object`
        Instance of Compartment class
    System : `object`
        Instance of System class
    mol_types : `array_like`
        Array of strings of molecule names
    N : `float64[:]`
        Number of molecules to distribute per given type
    multiplier : `int64`
        Specifies the multiplication factor by which the algorithm upscales the target number for the Monte Carlo sampling. Default = 100
    
    
    Returns
    -------
    tuple(float64[:,3], int64[:], float64[:,4])
        Returns arrays of the positions, molecule types, and rotation quaternions of the distributed moelcules.
    
    """

    radii = np.empty(len(mol_types))
    mol_type_ids = np.empty(len(mol_types), dtype = np.int64)
    for i, moltype in enumerate(mol_types):
        radii[i] = System.molecule_types[str(moltype)].radius
        mol_type_ids[i] = System.molecule_types[str(mol_types[i])].type_id
        
    weights = radii**2/max(radii**2)
    
    points, points_type, quaternion, count = poisson_disc_sampling_uniform(Compartment, radii, mol_type_ids, weights, N, System, multiplier = multiplier)
    
    for i, moltype in enumerate(mol_types):
        print(str(count[i])+' molecules '+moltype+' distributed in volume of compartment '+Compartment.name)
    print('-------------------------------')
    
    return points, points_type, quaternion



@nb.njit
def monte_carlo_distribution_3D(Compartment, System, N_total):
    
    """Does Monte Carlo sampling for N points inside a compartment.
    
    Parameters
    ----------
    Compartment : `object`
        Instance of Compartment class
    System : `object`
        Instance of System class
    N_total : `int64`
        Total number of points to distribute
    
    
    Returns
    -------
    float64[:,3]
        Array of positions
    
    """
    
    
    AABB = np.copy(Compartment.AABB)
    for dim in range(3):
        if AABB[0][dim]<System.AABB[0][dim]:
            AABB[0][dim] = System.AABB[0][dim]
        if AABB[1][dim]>System.AABB[1][dim]:
            AABB[1][dim] = System.AABB[1][dim]
    
    box_lengths = AABB[1]-AABB[0]
    
    
    points_MC = AABB[0] + np.random.rand(N_total, 3)*box_lengths
                
    return points_MC


@nb.njit
def poisson_disc_sampling_uniform(Compartment, radii, mol_type_ids, weights, N, System, multiplier = 100):
    
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
    
    
    if System.mesh==True: 
        
        rc = max(np.max(radii)*2, np.max(Compartment.box_lengths)/25)
        cells_per_dim = (Compartment.box_lengths / rc).astype(np.int64)
        print("Debugging Cell division <3 2")
        print(radii)
        print(System.box_lengths)
        print(rc)
        print(Compartment.box_lengths)
        print(cells_per_dim)
        if np.any(cells_per_dim<3):
            print('error: Cell division <3')
        cell_length_per_dim = Compartment.box_lengths / cells_per_dim
        
        triangle_ids = np.arange(System.N_triangles)
        
        CellList, AABB_centers = create_cell_list_mesh(System, Compartment, cells_per_dim, cell_length_per_dim, triangle_ids, False, True) # np.max(System.box_lengths)/50
        
        
        # Check if any part of the compartment is actually in the simulation box so we dont end
        # up in an infinite loop!
        Compartment_in_box = mesh_inside_box_test(Compartment, System)
        if Compartment_in_box == False:
    
            raise ValueError('There is no overlap of the Compartment with the Simulation box! Unable to distribute points inside Compartment!')
    
    
    points = nb.typed.List.empty_list(listtype_1)
    points_type = nb.typed.List.empty_list(listtype_2)
    quaternion = nb.typed.List()
    count = np.zeros(len(radii), dtype = np.int64)
    
    #1. -----------------------------------------------------
    N_total = np.sum(N)*multiplier#50
    sample_points = monte_carlo_distribution_3D(Compartment, System, N_total)
    
    #2. -----------------------------------------------------
    rc = max(np.max(radii)*2, np.max(Compartment.box_lengths)/25) # np.max(radii)*10
    
    active_cell_list, N_samples_cell, nonempty_cells, Ncell, cells_per_dim = create_cell_list_points(rc, sample_points, Compartment)
    
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
        
        #3.3 -----------------------------------------------------
        # Check if point is inside the compartment
        if System.mesh==True and Compartment.name == 'System':
            point_on_correct_side = True
            for comp_id in System.Compartments:
                Compartment_temp = System.Compartments[comp_id]
                point_on_correct_side_temp = point_inside_mesh_test(Compartment_temp.triangle_ids, System.Mesh, System.vertices, pos_i, 1e0)==False
                if point_on_correct_side_temp == False:
                    point_on_correct_side = False
        elif System.mesh==True and Compartment.name != 'System':
            point_on_correct_side = point_inside_mesh_test(Compartment.triangle_ids, System.Mesh, System.vertices, pos_i, 1e0)
        else:
            point_on_correct_side = True   
            

        if point_on_correct_side: 
                
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
                
                if System.mesh==True and radii[type_id]>0.0: 
                    # !!! Here we use a loose cell grid!
                    cx = int((pos_i[0]-Compartment.origin[0]) / cell_length_per_dim[0])
                    cy = int((pos_i[1]-Compartment.origin[1]) / cell_length_per_dim[1])
                    cz = int((pos_i[2]-Compartment.origin[2]) / cell_length_per_dim[2])
            
                    # Determine cell in 3D volume for i-th particle
                    cell = cx + cy * cells_per_dim[0] + cz * cells_per_dim[0] * cells_per_dim[1]
                
                    min_dist = 1e6
                    triangles_list = CellList.get_triangles(cell)
                    if len(triangles_list)>0:
                        for Tri_idx in triangles_list:
                            
                            triangle = System.Mesh[Tri_idx]['triangles']
                        
                            p0 = System.vertices[triangle[0]]
                            p1 = System.vertices[triangle[1]]
                            p2 = System.vertices[triangle[2]]
                            
                            dist, region = point_triangle_distance(p0,p1,p2,pos_i)
                            if dist<min_dist:
                                min_dist = dist
                    no_overlap = min_dist >= radii[type_id]
                else:
                    no_overlap = True
                    
                    
                if no_overlap:   
                
                    fits = True
                    cx,cy,cz = reverse_cell_mapping(cell_id, cells_per_dim)
                    
                    cz_start, cz_end = cz - 1, cz + 2
                    cy_start, cy_end = cy - 1, cy + 2
                    cx_start, cx_end = cx - 1, cx + 2
                    
                    if System.boundary_condition_id != 0:
                        
                        if cz_start < 0:
                            cz_start = 0 
                        elif cz_end > cells_per_dim[2]:
                            cz_end = cells_per_dim[2]
                            
                        if cy_start < 0:
                            cy_start = 0 
                        elif cy_end > cells_per_dim[1]:
                            cy_end = cells_per_dim[1]
                            
                        if cx_start < 0:
                            cx_start = 0 
                        elif cx_end > cells_per_dim[0]:
                            cx_end = cells_per_dim[0]
                            
                    for cz_N in range(cz_start, cz_end):
                        cz_shift = 0 + cells_per_dim[2] * (cz_N < 0) - cells_per_dim[2] * (cz_N >= cells_per_dim[2])
                        pos_shift[2] = 0.0 - Compartment.box_lengths[2] * (cz_N < 0) + Compartment.box_lengths[2]*(cz_N >= cells_per_dim[2])
                        for cy_N in range(cy_start, cy_end):
                            cy_shift = 0 + cells_per_dim[1] * (cy_N < 0) - cells_per_dim[1] * (cy_N >= cells_per_dim[1])
                            pos_shift[1] = 0.0 - Compartment.box_lengths[1] * (cy_N < 0) + Compartment.box_lengths[1] * (cy_N >= cells_per_dim[1])
                            for cx_N in range(cx_start, cx_end):
                                cx_shift = 0 + cells_per_dim[0] * (cx_N < 0) - cells_per_dim[0] * (cx_N >= cells_per_dim[0])
                                pos_shift[0] = 0.0 - Compartment.box_lengths[0] * (cx_N < 0) + Compartment.box_lengths[0] * (cx_N >= cells_per_dim[0])
                                
                                cell_temp = (cx_N + cx_shift) + (cy_N + cy_shift) * cells_per_dim[0] + (cz_N + cz_shift) * cells_per_dim[0] * cells_per_dim[1]
                                
                                for neighbour in occupied_cell_list[cell_temp]:
                                    sample_id_j, type_id_j = neighbour
                                    pos_j = sample_points[sample_id_j] 
                                    
                                    if System.boundary_condition_id == 0:
                                        dx = pos_i[0]-(pos_j[0]+pos_shift[0])
                                        dy = pos_i[1]-(pos_j[1]+pos_shift[1])
                                        dz = pos_i[2]-(pos_j[2]+pos_shift[2])
                                    else:
                                        dx = pos_i[0] - pos_j[0]
                                        dy = pos_i[1] - pos_j[1]
                                        dz = pos_i[2] - pos_j[2]
                                    
                                    distance = np.sqrt(dx**2+dy**2+dz**2)
                                            
                                    if distance<radii[type_id]+radii[type_id_j]:
                                        fits=False
                                        
                else:
                    fits = False
                
                # 6. -----------------------------------------------------
                # If True, accept the sample and add the molecule type and position to an occupied sample list. 
                # Next, delete all other samples within radius Ri, as these wont ever become occupied anyway.
                if fits==True:
                    
                    count[type_id] += 1
                    
                    occupied_cell_list[cell_id].append(np.array([sample_id, type_id], dtype = np.int64))
                    
                    points.append(pos_i)
                    points_type.append(mol_type_ids[type_id])
                    q = random_quaternion()
                    quaternion.append(q)
                    
                    #7. -----------------------------------------------------
                    # Update the number count of samples for the current cell.
            
    # print('N_Trials: ', N_Trials)
    # print('max_Trials: ', max_Trials)
    
    return points, points_type, quaternion, count



