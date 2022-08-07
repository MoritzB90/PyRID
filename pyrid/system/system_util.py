# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numpy as np
import numba as nb
from numba.experimental import jitclass
from ..reactions import reactions_registry_util as rru
from ..geometry.mesh_util import triangle_centroid, triangle_area, triangle_volume_signed
from ..math.transform_util import local_coord, normal_vector, isclose, barycentric_params
from ..math import random_util as rnd
from ..data_structures.cell_list_util import CellListMesh, create_cell_list_mesh

#%%

@nb.njit
def string_in_array(string, strings_array):
    
    in_array = False
    
    for string_i in strings_array:
        if str(string_i) == str(string):
            in_array = True
            
    return in_array
        
    
#%%

item_t_border_2d = np.dtype([('edge_id', np.int64), ('triangle_ids', np.int64), ('direction_normal', np.float64, (3,)), ('edge_length', np.float64), ('cum_edge_length', np.float64),],  align=True)
border_2d_lt = nb.typeof(np.zeros(1, dtype = item_t_border_2d))

item_t_border_3d = np.dtype([('triangle_ids', np.int64), ('cum_area', np.float64),],  align=True)
border_3d_lt = nb.typeof(np.zeros(1, dtype = item_t_border_3d))

item_t_group= np.dtype([('triangle_ids', np.int64),],  align=True)
group_lt = nb.typeof(np.zeros(1, dtype = item_t_group))

key_typeGroup = nb.types.string
value_typeGroup = group_lt

# item_t_border_3d = np.dtype([('triangle_ids', np.int64, (3,)),],  align=True)

# border_3d_lt = nb.typeof(np.zeros(1, dtype = item_t_border_2d))

spec = [
    ('triangle_ids', nb.int64[:]),
    ('triangle_ids_exclTransp', nb.int64[:]),
    ('name', nb.types.string),
    ('id', nb.int64),
    ('AABB', nb.float64[:,:]),
    ('origin', nb.float64[:]),
    ('box_lengths', nb.float64[:]),
    ('centroid', nb.float64[:]),
    ('border_2d', border_2d_lt),
    ('border_length', nb.float64),
    ('border_3d', border_3d_lt),
    ('border_area', nb.float64),
    ('groups', nb.types.DictType(key_typeGroup, value_typeGroup)),
    ('area', nb.float64),
    ('volume', nb.float64),
    ('box_overlap', nb.types.boolean),
]

@jitclass(spec)
class Compartment(object):
    
    """
    Compartment class contains attributes and methods used to describe a triangulated mesh compartment.
    
    Attributes
    ----------
    triangle_ids : `int64[:]`
        triangle indices of the compartment mesh.
    triangle_ids_exclTransp : `int64[:]`
        triangle indices of the compartment mesh, escluding triangles marked as transparent.
    name : `str`
        Name of the compartment.
    AABB : `float64[:,:]`
        Axis Aligned Bounding Box of the compartment
    origin : `float64[:]`
        coordinates of AABB origin (AABB[0])
    box_lengths : `float64[:]`
        length of AABB in each dimension
    centroid : `float64[:]`
        centroid of mesh
    border_2d : `array_like 1-D`
        Information about edge of intersection between mesh and simualtion box
        
        `data-type: np.dtype([('edge_id', np.int64), ('triangle_ids', np.int64),('direction_normal', np.float64, (3,)), ('edge_length', np.float64),('cum_edge_length', np.float64),],  align=True)`
    border_length : `float64`
        length of edge of intersection between mesh and simualtion box
    border_3d : `array_like 1-D`
        Information about faces of intersection between mesh and simulation box
        
        `data-type: np.dtype([('triangle_ids', np.int64), ('cum_area', np.float64),],  align=True)`
    border_area : float64
        area of intersection between mesh and simualtion box
    groups : `array_like 1-D`
        triangle ids of face groups
        
        data-type: `np.dtype([('triangle_ids', np.int64),],  align=True)`
    area : `float64`
        Total area of mesh surface
    volume : `float64`
        Total mesh volume
    box_overlap : `bool`
        True is any intersection between mesh and simulation box
        
    
    
    Methods
    -------
    calc_centroid_radius_AABB(self, System):
        calculates some general properties of the mesh compartment
    add_border_2d(triangle_ids_border, System):
        Registers the edge of intersection (if exists) between compartment mesh and simulation box.
    
    """

    def __init__(self, triangle_ids, name, comp_id, System):
        
        if len(System.triangle_ids_transparent)>0:
            self.triangle_ids_exclTransp = np.array([tri_id for tri_id in triangle_ids if tri_id not in System.triangle_ids_transparent], dtype = np.int64)
        else:
            self.triangle_ids_exclTransp = triangle_ids
            
        self.triangle_ids = triangle_ids
        self.name = name
        self.id = comp_id
        self.groups = nb.typed.Dict.empty(key_typeGroup, value_typeGroup)
        
        for tri_id in triangle_ids:
            System.Mesh[tri_id]['comp_id'] = comp_id
        
        self.calc_centroid_radius_AABB(System) 
        
        self.box_overlap = False
        
    def calc_centroid_radius_AABB(self, System):
        """Calculates some general properties of the mesh compartment
        
             - area
             - volume
             - centroid
             - AABB (Axis Aligned Bounding Box), origin, box_lengths
             
        The centroid (geometric center) of the mesh is calculated by averaging over
        all triangles centroids, weighted by their area.

        Parameters
        ----------
        System : `obj`
            Instance of System class
        
        
        """
        
        self.area = 0.0
        for tri_id in self.triangle_ids_exclTransp:
            
            triangle = System.Mesh[tri_id]['triangles']
            p0 = System.vertices[triangle[0]]
            p1 = System.vertices[triangle[1]]
            p2 = System.vertices[triangle[2]]
            
            self.area += triangle_area(p0,p1,p2)
            
        self.volume = 0.0
        self.centroid = np.array([0.0,0.0,0.0])
        area_sum = 0.0
        
        for tri_id in self.triangle_ids:
            triangle = System.Mesh[tri_id]['triangles']
            p0 = System.vertices[triangle[0]]
            p1 = System.vertices[triangle[1]]
            p2 = System.vertices[triangle[2]]
            
            self.volume += triangle_volume_signed(p0, p1, p2);
            
            center = (p0 + p1 + p2) / 3 # center of tetrahedron
            area = 0.5*np.linalg.norm(np.cross(p1-p0,p2-p0))  # signed volume of tetrahedron
            self.centroid += area*center
            area_sum += area
        
        self.centroid /= area_sum
        
        ###############################
        self.AABB = np.zeros((2,3))
        for d in range(3):
            self.AABB[0][d] = self.centroid[d]
            self.AABB[1][d] = self.centroid[d]
        # Since we iterate over all triangles, vertices are accounted for multiple times.
        # However, this should not be a huge performance issue here. Still, I should probably come up
        # with a better solution at some point.
        for i in range(3):
            # for vert_idx in self.triangles[:,i]:
            for tri_id in self.triangle_ids:
                vert_idx = System.Mesh[tri_id]['triangles'][i]
                
                for d in range(3):
                    if System.vertices[vert_idx][d]<self.AABB[0][d]:
                        self.AABB[0][d] = System.vertices[vert_idx][d]
                    if System.vertices[vert_idx][d]>self.AABB[1][d]:
                        self.AABB[1][d] = System.vertices[vert_idx][d]
                        

        self.origin = np.empty(3)
        self.origin[0] = self.AABB[0][0]
        self.origin[1] = self.AABB[0][1]
        self.origin[2] = self.AABB[0][2]
                
        
        self.box_lengths = np.empty(3)
        self.box_lengths[0] = self.AABB[1][0]-self.AABB[0][0] #1.0 #
        self.box_lengths[1] = self.AABB[1][1]-self.AABB[0][1] #1.0 #
        self.box_lengths[2] = self.AABB[1][2]-self.AABB[0][2] #1.0 #
        
    
    def add_border_2d(self, triangle_ids_border, System):
        
        """Registers the edge of intersection (if exists) between compartment mesh and simulation box. The intersection line is needed for fixed concdentration simulations as new molecules enter the simualtion box along this intersection line.
        
        .. admonition:: Note
        
            See naming conventions for face groups!
        
        Parameters
        ----------
        triangle_ids_border : `int64[:]`
            triangle ids of the mesh triangle along the border
        System : `obj`
            Instance of System class
        
        
        """
    
        # eps = 1e-3
        
        System.boundary_2d = True
        
        
        edge_ids = []
        direction_normals = []
        edge_lengths = []
        triangle_ids = []
        
        # Edges = []
        
        for i in range(len(triangle_ids_border)):

            tri_id = triangle_ids_border[i]
        
            # Find the edge that aligns with one of the box faces:
            edges = System.Mesh[tri_id]['edges']
            for j in range(3):
                
                """
                edge:   [0,1]
                        [1,2]
                        [0,2]
                """
                edge = edges[j]
                v0 = edge[0]
                v1 = edge[1]
                
                p0 = System.vertices[v0]
                p1 = System.vertices[v1]
                
                for dim in range(3):
                    if isclose(p0[dim],p1[dim]) and isclose(abs(p1[dim]),abs(System.box_lengths[dim]/2)):
                        
                        # Change the vertex position such that the edge aligns
                        # exactly with the simulation box border:
                        p0[dim] = np.sign(p0[dim])*System.box_lengths[dim]/2
                        System.update_mesh(vertex = v0)
                        
                        p1[dim] = np.sign(p1[dim])*System.box_lengths[dim]/2
                        System.update_mesh(vertex = v1)
                        ######
                        
                        edge_ids.append(j)
                      
                        # Calculate the direction normal vector:
                        v_edge = p0-p1
                        normal =  System.Mesh[tri_id]['triangle_coord'][3]
                        direction = np.cross(v_edge, normal)
                        direction /= np.linalg.norm(direction)
                        
                        if np.sign(direction[dim]) == np.sign(p0[dim]):
                            direction *= -1
                        
                        direction_normals.append(direction)
                        
                        edge_lengths.append(np.linalg.norm(v_edge))
                        
                        triangle_ids.append(tri_id)
                        
                        System.Mesh[tri_id]['border_edge'][j] = 1
                        System.Mesh[tri_id]['border_dim'][j] = dim
                        System.Mesh[tri_id]['border_normal'][j][:] = direction
         
        #---------
        
        
        self.border_2d = np.zeros(len(edge_ids), dtype = item_t_border_2d)
        
        for i in range(len(edge_ids)):
            
            self.border_2d[i]['edge_id'] = edge_ids[i]
            self.border_2d[i]['direction_normal'][:] = direction_normals[i]
            self.border_2d[i]['edge_length'] = edge_lengths[i]
            self.border_2d[i]['triangle_ids'] = triangle_ids[i]
            
        self.border_2d['cum_edge_length'][:] = np.cumsum(self.border_2d['edge_length'])
        
        self.border_length = np.sum(self.border_2d['edge_length'])
            
    def add_border_3d(self, triangle_ids_border, System):
        
        """Registers the area of intersection (if exists) between compartment mesh and simulation box. The intersection area is needed for fixed concdentration simulations as new molecules enter the simualtion box along the faces of the intersection.
        
        .. admonition:: Note
        
            See naming conventions for face groups!
        
        Parameters
        ----------
        triangle_ids_border : `int64[:]`
            triangle ids of the mesh triangle along the border
        System : `obj`
            Instance of System class
        
        
        """
        
        # eps = 1e-3
        
        self.border_3d = np.zeros(len(triangle_ids_border), dtype = item_t_border_3d)
        
        for i in range(len(triangle_ids_border)):

            tri_id = triangle_ids_border[i]
            self.border_3d[i]['triangle_ids'] = tri_id
            
            # Make sure the box vertices lie exactly on the simulation box border:
            triangle = System.Mesh[tri_id]['triangles']
            for vert in range(3):
                for dim in range(3):
                    if isclose(abs(System.vertices[triangle[vert]][dim]),abs(System.box_lengths[dim]/2)):
                        
                        System.vertices[triangle[vert]][dim] = np.sign(System.vertices[triangle[vert]][dim])*System.box_lengths[dim]/2
                        
                        System.update_mesh(vertex = triangle[vert])
            
        
        self.border_3d['cum_area'][:] = np.cumsum(System.Mesh[triangle_ids_border]['triangle_area'])
        
        self.border_area = np.sum(System.Mesh[triangle_ids_border]['triangle_area'])
        
        self.box_overlap = True
            
    def add_group(self, group_name, triangle_ids_group, System):
        
        """Registers any triangle/face group that has been defined in the mesh ob file that is not a 2d or 3d border of intersectioj with the simulation box 
        
        
        .. admonition:: Note
        
            See naming conventions for face groups!
        
        Parameters
        ----------
        group_name : `str`
            name of the group
        triangle_ids_group : `int64[:]`
            triangle ids of the mesh triangle group
        System : `obj`
            Instance of System class
        
        """
        
        self.groups[group_name] = np.zeros(len(triangle_ids_group), dtype = item_t_group)
        
        for i in range(len(triangle_ids_group)):

            tri_id = triangle_ids_group[i]
            self.groups[group_name][i]['triangle_ids'] = tri_id
        

#%%


#TODO: molecule_type does not need to be a class, could simply use a struct instead!

item_t_UMR = np.dtype([('rate', np.float64), ('id', np.int64),],  align=True)

spec_moltype = [
    ('pos', nb.float64[:,:]),
    ('h_membrane', nb.float64),
    ('radii', nb.float64[:]),
    ('pos_rb', nb.float64[:,:]),
    ('radii_rb', nb.float64[:]),
    ('types', nb.typeof(np.array(['type'], dtype = np.dtype('U20')))),
    ('radius', nb.float64),
    ('radius_2d', nb.float64),
    ('Dtrans', nb.float64),
    ('Dtrans_2d', nb.float64),
    ('Drot', nb.float64),
    ('Drot_2d', nb.float64),
    ('mu_rb', nb.float64[:,::1]),
    ('mu_tb', nb.float64[:,::1]),
    ('mu_rb_sqrt', nb.float64[:,::1]),
    ('mu_tb_sqrt', nb.float64[:,::1]),
    ('D_rr', nb.float64[:,::1]),
    ('D_tt', nb.float64[:,::1]),
    ('name', nb.types.string),
    ('type_id', nb.int64),
    ('r_mean', nb.float64),
    ('r_mean_pair', nb.float64),
    ('volume', nb.float64),
    ('collision_type', nb.int64),
    ('transition_rate_total', nb.float64),
    ('number_umr', nb.int64),
    ('um_reaction', nb.boolean),
    ('concentration', nb.float64[:]),
    ('concentration_surface', nb.float64[:]),
    ('l_perp', nb.float64),
    ('h', nb.int64),
    ]

@jitclass(spec_moltype)
class MoleculeType(object):
    
    """
    A brief summary of the classes purpose and behavior
    
    Attributes
    ----------
    name : `str`
        Name of molecule    
    type_id : `int64`
        ID of molecule type    
    collision_type : `int64` {0,1}
        collision type of molecule. (Default = 0)
    pos : `float64[N,3]`
        positions of each particle 
    radii : `float64[N]`
        radii of particles
    pos_rb : `float64[N,3]`
        position of each particle with exculded volume radius > 0.0
    radii_rb : `float64[N]`
        radii of particles with exculded volume radius > 0.0
    types : `array_like 1-D`
        array of names of particle types
        `dtype : np.dtype('U20')`
    radius : `float64`
        Total radius of molecule
    radius_2d : `float64`
        Total radius of molecule in plane (needed when distributing molecules on surface)  
    Dtrans : `float64`
        Translational diffusion coefficient 
    Drot : `float64`
        Rotational diffusion coefficient 
    mu_rb : `float64[3,3]`
        Rotational mobility tensor
    mu_tb : `float64[3,3]`
        Translational mobility tensor
    mu_rb_sqrt : `float64[3,3]`
        Rotational mobility tensor
    mu_tb_sqrt : `float64[3,3]`
        Square root of translational mobility tensor
    mu_rb_sqrt : `float64[3,3]`
        Square root of rotational mobility tensor
    D_tt : `float64[3,3]`
        Translational diffusion tensor
    D_rr : `float64[3,3]`
        Rotational diffusion tensor
    r_mean : `float64`
        Average radial displacement of diffusing molecule
    volume : `float64`
        Volume of molecule
    transition_rate_total : `float64`
        Total transition rate of unimolecular reactions
    um_reaction : `bool`
        True if unimolecular reaction has been defined for molecule type
    number_umr : `int64`
        Number of unimolecular reactions
    concentration : `float64[:]`
        concentration of molecule in volume of each compartment outside simulation box (Needed for simulation with fixed concentration boundary condition)
    concentration_surface : `float64[:]`
        concentration of molecule on surface of each compartment outside simulation box (Needed for simulation with fixed concentration boundary)
    l_perp : `float64`
        Mean diffusive displacement of molecule perpendicular to a plane.
    
    
    
        
    
    Methods
    -------
    update_um_reaction(rate)
        Updates the the total transitaion rate for unimolecular reactions of this molecule type.
    
    """
    
    def __init__(self, pos, types, radii, name, collision_type, System, h_membrane = None):
        
        self.pos = pos
        self.types = types
        self.radii = radii
        self.volume = 0
        self.name = name
        self.type_id = System.ntypes_rb
        self.collision_type = collision_type
        
        if h_membrane is None:
            self.h_membrane = 0.0
        else:
            self.h_membrane = h_membrane
        
        self.concentration = np.zeros(len(System.Compartments)+1, dtype = np.float64)
        self.concentration_surface = np.zeros(len(System.Compartments)+1, dtype = np.float64)
        
        self.um_reaction = False
        self.number_umr = 0
        
        self.pos_rb = pos[radii>0]
        self.radii_rb = radii[radii>0]
        
        self.radius = 0.0
        self.radius_2d = 0.0
        distance_from_center = 0.0
        distance_from_center_2d = 0.0
        for i,p in enumerate(pos):
            if self.radii[i]>0:
                
                self.volume += 4/3*np.pi*self.radii[i]**3
                
                distance_from_center = np.linalg.norm(p)+radii[i]
                distance_from_center_2d = np.linalg.norm(p[0:2])+radii[i]
                
                if distance_from_center > self.radius:
                    self.radius = distance_from_center
                    
                if distance_from_center_2d > self.radius_2d:
                    self.radius_2d = distance_from_center_2d
                    
        
        if self.radius == 0:
            raise Warning('Molecules must not have a total radius of 0!')
            
        # self.update_mobility_matrix(System)
        
        self.Dtrans=System.kbt/(6*np.pi*System.eta*self.radius) #in mum^2/s
        self.Drot=System.kbt/(8*np.pi*System.eta*self.radius**3) #in 1/s
        
        
        
    def update_um_reaction(self, rate):
        
        """Updates the the total transitaion rate for unimolecular reactions of this molecule type.
        
        Parameters
        ----------
        rate : float64
            Reaction rate 
        
        Notes
        -----
        We cannot tell when a bimolecular reaction will occur, beacause we do not know when 
        two diffusing molecules meet. However, for unimolecular reactions the expected 
        lifetime distribution is known in case there is no interruption by any bimolecular reaction: 
        
        .. math::
            :label: rho
            
            \\rho(t) = k*e^{-k t},
            
        where :math:`k = \sum_i^n(k_i)`.
        As such, we can schedule a reaction event by using the upper equation :cite:t:`Stiles2001`, :cite:t:`Erban2007`. 
        For a unimolecular reaction occuring between t+dt we add this reaction to the reaction list 
        evaluated at time t+dt. The reaction is then processed as any other bimolecular reaction that may 
        occur within that timestep. If the unimolecular reaction is picked, it needs to be checked if different 
        unimolecular reaction pathways exist. The fractional probablity for each of the n 
        transitions is:
        
        .. math::
            :label: pi
            
            p_i = k_i/\sum_n(k_j)
            
        Based on the latter equation one of the n unimolecular transitions will be picked.
        As for bimolecular reactions, all other reactions that exist for the educt will be deleted.
        In case a bimolecular reaction is picked before the unimolecular reaction and in case it is successfull, the unimolecular reaction will be deleted from the reaction list and a new reaction time is drawn for the product molecule of the bimolecular reaction if any unimolecular reaction has been defined for the product type.
        
        """
        
        self.um_reaction = True
        
        self.number_umr += 1
        
        self.transition_rate_total += rate
        
#%%

item_t_bpr = np.dtype([('ids', np.int64, (100,)), ('n_reactions', np.int64),],  align=True) 

item_t_ptype = np.dtype([('id', np.int64), ('radius', np.float64), ('cutoff', np.float64), ('h', np.int64), ('transition_rate_total', np.float64), ('UP_reaction', bool), ('bond_educt_id', np.int64),],  align=True) 

key_typePT = nb.types.string
value_typePT = nb.typeof(np.empty(1, dtype = item_t_ptype))

key_mt = nb.types.string 
value_mt = MoleculeType.class_type.instance_type

keytype_RD = nb.int64
valuetype_RD = rru.Reaction.class_type.instance_type

item_t_inter = np.dtype([('parameters', (np.float64, (5,))), ('type', 'U20'), ('id', np.int64), ('global', bool), ('cutoff', np.float64), ('breakable', bool),],  align=True)

item_t_react = np.dtype([('defined', bool), ('bond', bool), ('id', np.int64), ('enzyme', np.int64), ('rate', np.float64), ('radius', np.float64), ('type', 'U20'), ('type_id', np.int64), ('type_BP_BM', 'U2'),],  align=True)


item_t_barostat = np.dtype([('active', bool),('mu', np.float64),('mu_tensor', np.float64, (3,3)),('Tau_P', np.float64),('P0', np.float64),('start', np.int64)],  align=True)   
barostat_lt = nb.typeof(np.zeros(1, dtype = item_t_barostat))

# item_t_baryc = np.dtype([('d00', np.float64),('d01', np.float64),('d11', np.float64),('denom', np.float64)],  align=True)

item_t_mesh = np.dtype([('triangles', np.int64, (3,)), ('edges', np.int64, (3,2)), ('border_edge', np.int64, (3,)), ('border_dim', np.int64, (3,)), ('border_normal', np.float64, (3,3)), ('neighbours', np.int64, (3,)), ('triangle_coord', np.float64, (4,3)), ('normal', np.float64, (3,)), ('barycentric_params', np.float64, (4,)), ('triangle_distance', np.float64), ('triangle_centroid', np.float64, (3,)), ('triangle_area', np.float64), ('stamp', np.int64), ('comp_id', np.int64), ('group_id', np.int64),],  align=True)

mesh_lt = nb.typeof(np.zeros(1, dtype = item_t_mesh))

# TODO: Might be usefull to add a struct for mesh edges, so we can better save edge properties, e.g. for borders. Also, we would have less memory overhead compared to saving edge data for each triangle, which introduces some redudance!
item_t_edges = np.dtype([('edge', np.int64, (2,)), ('border_edge', bool),],  align=True)
edges_lt = nb.typeof(np.zeros(1, dtype = item_t_edges))


Comp_keytype = nb.int64
Comp_valuetype = Compartment.class_type.instance_type

item_t_border_box = np.dtype([('triangle_ids', np.int64), ('cum_area', np.float64),],  align=True)
border_box_lt = nb.typeof(np.zeros(1, dtype = item_t_border_box))

list_int64 = nb.types.ListType(nb.int64)

spec = [
    ('N', nb.int64),
    ('Np', nb.int64),
    ('Nmol', nb.int64[:]),
    ('dim', nb.int64),
    ('dt', nb.float64),
    ('box_lengths', nb.float64[:]),
    ('volume', nb.float64),
    ('Epot', nb.float64[:]),
    ('AABB', nb.float64[:,:]), # AABB of the simulation box
    ('AABB_all', nb.float64[:,:]), # AABB covering all mesh compartments
    ('origin', nb.float64[:]),
    ('empty', nb.int64),
    ('vertices', nb.float64[:,::1]),
    ('vertex_triangles', nb.types.ListType(list_int64)),
    ('Mesh', mesh_lt),
    ('N_triangles', nb.int64),
    ('triangle_ids', nb.int64[:]),
    ('box_border', border_box_lt),
    ('box_area', nb.float64),
    ('boundary_2d', nb.types.boolean),
    ('triangle_ids_transparent', nb.int64[:]),
    ('Compartments', nb.types.DictType(Comp_keytype, Comp_valuetype)),
    ('n_comps', nb.int64),
    ('area', nb.float64),
    ('cells_per_dim', nb.int64[:]),
    ('cell_length_per_dim', nb.float64[:]),
    ('Ncell', nb.int64),
    ('AABB_centers', nb.float64[:,:]),
    ('CellList', CellListMesh.class_type.instance_type),
    ('particle_types', nb.types.DictType(key_typePT, value_typePT)),
    ('particle_id_to_name', nb.typeof(np.zeros(10, dtype = 'U20'))),
    ('ntypes', nb.int64),
    ('ntypes_rb', nb.int64),
    ('kB', nb.float64),
    ('Temp', nb.float64),
    ('eta', nb.float64),
    ('max_reactions_per_particle', nb.int64),
    ('reactions_left', nb.int64),
    ('molecule_types', nb.types.DictType(key_mt, value_mt)),
    ('Reactions_Dict', nb.types.DictType(keytype_RD, valuetype_RD)),
    ('reaction_id', nb.int64),
    ('bp_reaction_ids', nb.typeof(np.zeros(1, dtype = item_t_bpr))),
    ('up_reaction_id', nb.int64[:]),
    ('um_reaction_id', nb.int64[:]),
    ('interaction_args', nb.typeof(np.zeros((1,1), dtype = item_t_inter))),
    ('interaction_IDs', nb.types.DictType(nb.types.string, nb.int64)),
    ('reaction_args', nb.typeof(np.zeros((1,1), dtype = item_t_react))),
    ('molecule_id_to_name', nb.typeof(np.zeros(1, dtype = 'U20'))),
    ('mesh_scale', nb.float64),
    ('mesh', nb.types.boolean),
    ('boundary_condition', nb.types.string),
    ('boundary_condition_id', nb.int64),
    ('interactions', nb.types.boolean),
    ('avogadro', nb.float64),
    ('kbt', nb.float64),
    ('wall_force', nb.float64),
    ('barostat', barostat_lt),
    ('seed', nb.int64),
    ('mol_type_id', nb.types.DictType(nb.types.string, nb.int64)),
    ('react_type_id', nb.types.DictType(nb.types.string, nb.int64)),
    ('virial_scalar', nb.float64[:]),
    ('virial_scalar_Wall', nb.float64[:]),
    ('virial_tensor', nb.float64[:,:,:]),
    ('virial_tensor_Wall', nb.float64[:,:,:]),
    ('Pressure', nb.float64[:]),
    ('N_bonds', nb.int64[:,:]),
    ('name', nb.types.string),
    ('id', nb.int64), # This is the id to identify the compartment. For Wolrd it is 0.
    ('count', nb.int64),
    ('current_step', nb.int64),
    ('time_stamp', nb.int64),
    ('interaction_defined', nb.types.boolean[:,:]),
    ('pair_interaction', nb.types.boolean[:]),
    ('length_unit', nb.types.string),
    ('time_unit', nb.types.string),
    ('energy_unit', nb.types.string),
    ('length_units_prefix', nb.types.DictType(nb.types.string, nb.float64)),
    ('time_units_prefix', nb.types.DictType(nb.types.string, nb.float64)),
    # ('energy_units', nb.types.DictType(nb.types.string, nb.float64)),
    ('max_cells_per_dim', nb.int64),
    ('dx_rand', rnd.Interpolate.class_type.instance_type),
    ('compartments_id', nb.types.DictType(nb.types.string, nb.int64)),
    ('compartments_name', nb.types.ListType(nb.types.string)),
]

@jitclass(spec)
class System(object):
    
    """
    A brief summary of the classes purpose and behavior
    
    Attributes
    ----------
    N : `nb.int64`
        Total number of molecules in simulation
    Np : `nb.int64`
        Total number of particles in simulation
    Nmol : `nb.int64[:]`
        Number of molecules per molecule type
    dim : `nb.int64 (3)`
        System dimension
    dt : `nb.float64`
        Integration time step
    box_lengths : `nb.float64[:]`
        Length of simulation box in each dimension
    volume : `nb.float64`
        Simulation box volume
    Epot : `nb.float64[:]`
        Total potential energy
    AABB : `nb.float64[2,3]`
        Axis Aligned Bounding Box of simulation box
    AABB_all : `nb.float64[2,3]`
        Axis Aligned Bounding Box of simulation box plus mesh compartments (only different from simulation box if compartments overlap with box)
    origin : `nb.float64[3]`
        origin of simulation box (AABB[0])
    empty : `nb.int64 (-1)`
        empty flag
    vertices : `nb.float64[N,3]`
        Position of all vertices in each dimension
    Mesh : `array_like`
        Mesh data

        `dtype : np.dtype([('triangles', np.int64, (3,)), ('edges', np.int64, (3,2)), ('border_edge', np.int64, (3,)), ('border_dim', np.int64, (3,)), ('border_normal', np.float64, (3,3)), ('neighbours', np.int64, (3,)), ('triangle_coord', np.float64, (4,3)), ('normal', np.float64, (3,)), ('triangle_distance', np.float64), ('triangle_centroid', np.float64, (3,)), ('triangle_area', np.float64), ('stamp', np.uint64), ('comp_id', np.uint64), ('group_id', np.uint64),],  align=True)`
    N_triangles : `nb.int64`
        Total number of mesh triangles
    triangle_ids : `nb.int64[:]`
        triangle ids of all compartments, excluding transparent faces (borders)
    box_border : `border_box_lt`
        Triangle faces of the simulation box border (needed for distributing molecules in case of fixed concentration boundary condition)
    box_area : `nb.float64`
        Area of triangle faces of the simulation box border
    boundary_2d : `nb.types.boolean`
        True if intersection of compartment with simulation box exists.
    triangle_ids_transparent : `nb.int64[:]`
        Triangle ids of transparent faces (particles do not collide)
    Compartments : `nb.types.DictType(Comp_keytype, Comp_valuetype)`
        Dictionary containing the Compartment instances
    n_comps : `nb.int64`
        Total number of compartments
    area : `nb.float64`
        Total surface area of all compartments
    cells_per_dim : `nb.int64[3]`
        Number of cells in each dimension of system grid
    cell_length_per_dim : `nb.float64[3]`
        Cell length in each dimension of system grid
    Ncell : `nb.int64`
        Total number of cells of system grid
    AABB_centers : `nb.float64[:,3]`
        Center of each system grid cell
    CellList : `CellListMesh.class_type.instance_type`
        Linked cell list of mesh triangles
    particle_types : `nb.types.DictType(key_typePT, value_typePT)`
        Particle types
    particle_id_to_name : `nb.typeof(np.zeros(10, dtype = 'U20'))`
        Returns the name of particle type by its id
    ntypes : `nb.uint64`
        Total number of particle types
    ntypes_rb : `nb.uint64`
        Total number of molecule types
    kB : `nb.float64`
        Boltzmann constant
    Temp : `nb.float64`
        Temperature of the system
    eta : `nb.float64`
        Viscosity
    max_reactions_per_particle : `nb.int64`
        Maximum number of reactions that can be defined per particle (Default = 20)
    reactions_left : `nb.int64`
        Number of reactions left
    molecule_types : `nb.types.DictType(key_mt, value_mt)`
        Information about each molecule type
    Reactions_Dict : `nb.types.DictType(keytype_RD, valuetype_RD)`
        Dictionary of reactions defined
    reaction_id : `nb.int64`
        Current reaction id
    up_reaction_id : `int64[:]`
        Reaction ids of all unimolecular particle reactions
    um_reaction_id : `int64[:]`
        Reaction ids of all unimolecular molecule reactions
    interaction_args : `nb.typeof(np.zeros((1,1), dtype = item_t_inter))`
        Arguments for each partcile-particle interaction
    interaction_IDs : `nb.types.DictType(nb.types.string, nb.int64)`
        Interaction ids of all partcile-particle interactions
    reaction_args : `nb.typeof(np.zeros((1,1), dtype = item_t_react))`
        Arguments for each particle-particle reaction (including fusion and enzymatic reactions)
    molecule_id_to_name : `nb.typeof(np.zeros(10, dtype = 'U20'))`
        Returns the name of molecule type by its id
    mesh_scale : `nb.float64`
        Scaling factor for the compartment meshes
    mesh : `nb.types.boolean`
        True if mesh compartments have been defined
    boundary_condition : `nb.types.string ('periodic', 'repulsive', 'fixed concentration')`
        Type of boundary condition
    boundary_condition_id : `nb.int64`
        Type id of boundary condition
    interactions : `nb.types.boolean`
        True if any interaction between particles has been defined
    avogadro : `nb.float64 (6.02214076e23)`
        Avogadros constant
    kbt : `nb.float64`
        Boltzmann constant * Temperature/Avogadros constant
    wall_force : `nb.float64`
        Force constant for repulsive walls (particle-boundary and particle-triangle interactions)
    barostat : `barostat_lt`
        Berendsen barostat
    seed : `nb.int64`
        Random seed of simulation if defined by user
    mol_type_id : `nb.types.DictType(nb.types.string, nb.int64)`
        Returns the id of molecule type by its name
    react_type_id : `nb.types.DictType(nb.types.string, nb.int64)`
        Returns the id of reaction type by its name
    virial_scalar : `nb.float64[:]`
        Virial (scalar)
    virial_scalar_Wall : `nb.float64[:]`
        Virial (scalar) of particle-wall interaction
    virial_tensor : `nb.float64[:,:,:]`
        Virial tensor
    virial_tensor_Wall : `nb.float64[:,:,:]`
        Virial tensor of particle-wall interaction
    Pressure : `nb.float64[:]`
        Current pressure
    N_bonds : `nb.int64[:,:]`
        Current number of bonds per particle type pair
    name : `nb.types.string ('System')`
        Name of compartment
    id : `nb.int64 (0)`
        ID of compartment
    count : `nb.int64`
        Counter attribute (for debugging purposes)
    current_step : `nb.int64`
        Current simulation time step
    time_stamp : `nb.uint64`
        Gives the current number of rays that have been cast for collision detection
    interaction_defined : `nb.types.boolean[:,:]`
        True if any interaction or reaction between the two particle types has been defined
    pair_interaction : `nb.types.boolean[:]`
        True if any pair-interaction has been defined for this particle type
    length_units_prefix : `nb.types.DictType(nb.types.string, nb.float64)`
        Length unit prefix (Default = 1e6 ('micrometer'))
    time_units_prefix : `nb.types.DictType(nb.types.string, nb.float64)`
        Time unit prefix (Default = 1e0 ('s'))
    max_cells_per_dim : `nb.int64`
        Maximum number of cells of system grid in each dimension (Default = 50)
    dx_rand : `rnd.Interpolate.class_type.instance_type`
        Interpolation of inverse cumulative probability function describing the displacement of molecules along a planes normal after crossing by diffusion.
    compartments_id : `nb.types.DictType(nb.types.string, nb.int64)`
        Returns the compartment id by its name
    compartments_name : `nb.types.ListType(nb.types.string)`
        Returns the compartment name by its id

    
    Methods
    -------
    add_system_grid()
        Calculates properties of the system cell grid.
    add_barostat_berendsen(Tau_P, P0, start)
        Adds a berendsen barostat to the system.
    register_molecule_type(molecule_name, particle_pos, particle_types, collision_type = 0)
        Regsiters a molecule type.
    set_diffusion_tensor(molecule_name, D_tt, D_rr, mu_tb, mu_rb, mu_tb_sqrt, mu_rb_sqrt)
        Sets the diffusion tensor for a molecule type.
    fixed_concentration_at_boundary(molecule_name, C, comp_name, location)
        Sets the fixed concentration boundary given a concentartion for each defined molecule type and for each compartents volume and surface.
    register_particle_type(Type, radius)
        Registers a particle type.
    set_compartments(comp_name, triangle_ids)
        Sets a compartment given its triangle ids.
    add_border_3d(triangle_ids_border)
        Registers the border faces of an intersection between compartments and the simulation box (Needed for fixed concentration boundary condition)
    add_mesh(vertices, triangles, mesh_scale = 1.0, box_triangle_ids = None, triangle_ids_transparent = None)
        Set up the mesh for the mesh compartments.
    add_edges()
        Finds the edges of each triangle and adds them to the Mesh struct.
    add_neighbours()
        Finds the neighbours of each triangle and adds them to the Mesh struct.
    get_AABB_centers()
        Calculates the center of each cell of the system grid.
    create_CellList()
        Creates the linked cell list for the triangulated meshes (Used during collision detection).
    
    ..
        **Particle interactions**
    
    add_interaction(self, interaction_type, type1, type2, parameters, bond = False, breakable = False)
    
    ..
        **Uniparticle reactions**
    
    add_up_reaction(self, reaction_type, educt_type, rate, product_types, product_loc, product_direction, radius)
     
    ..
        **Unimolecular reactions**
    
    add_um_reaction(self, reaction_type, educt_type, rate, product_types, product_loc, product_direction, radius)
    
    ..
        **Biparticle reactions**
        
    add_bp_reaction(self, reaction_type, educt_types, product_types, rate, radius, interaction_type = None, interaction_parameters = None)
        
    ..
        **Bimolecular reactions**
    
    add_bm_reaction(self, reaction_type, educt_types, product_types, particle_pairs, pair_rates, pair_Radii)
    
    """
    
    def __init__(self, box_lengths = np.array([50.0,50.0,50.0]), dt = 0.1, Temp = 293.15, eta = 1e-21, boundary_condition = 'periodic', wall_force = 100.0, seed = None, length_unit = 'nanometer', energy_unit = 'kJ/mol', time_unit = 'ns', cells_per_dim = None, max_cells_per_dim = 50):
        
        self.current_step = 0
        
        self.length_unit = length_unit
        self.time_unit = time_unit
        self.energy_unit = energy_unit
        
        self.length_units_prefix = nb.typed.Dict.empty(nb.types.string, nb.float64)
        self.length_units_prefix['micrometer'] = 1e6
        self.length_units_prefix['nanometer'] = 1e9
        
        self.time_units_prefix = nb.typed.Dict.empty(nb.types.string, nb.float64)
        self.time_units_prefix['s'] = 1.0
        self.time_units_prefix['ms'] = 1e3
        self.time_units_prefix['mus'] = 1e6
        self.time_units_prefix['ns'] = 1e9
        
        self.name = 'System'
        self.id = 0
        self.N = 0
        self.Np = 0
        self.dim = 3
        self.box_lengths = box_lengths
        self.max_cells_per_dim = max_cells_per_dim
        self.volume = self.box_lengths.prod()
        # self.Epot = 0.0
        
        self.boundary_condition = boundary_condition
        if boundary_condition == 'periodic':
            self.boundary_condition_id = 0
        elif boundary_condition == 'repulsive':
            self.boundary_condition_id = 1
        elif boundary_condition == 'fixed concentration':
            self.boundary_condition_id = 2
        else:
            raise NameError("Unknown boundary condition!") 
            
        # if boundary_condition == 'repulsive':
        if wall_force is not None:
            self.wall_force = wall_force
        else:
            self.wall_force = 1e20/self.length_units_prefix[length_unit]**2
            
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
            
        self.count = 0
        self.time_stamp = 0
        
        
        self.mesh_scale = 1.0
        self.mesh = False
        
        self.AABB = np.empty((2,3))
        self.AABB[0] = -box_lengths/2
        self.AABB[1] = box_lengths/2
        self.origin = np.empty(3)
        self.origin[0] = self.AABB[0][0]
        self.origin[1] = self.AABB[0][1]
        self.origin[2] = self.AABB[0][2]
        
        self.AABB_all = np.zeros((2,3), dtype = np.float64)
        
        self.empty = -50
        self.ntypes = 0
        self.ntypes_rb = 0
        self.max_reactions_per_particle = 20
        self.reactions_left = 0
        
        
        self.particle_types = nb.typed.Dict.empty(key_typePT, value_typePT)#
        self.particle_id_to_name = np.zeros(self.ntypes, dtype = 'U20')
        self.molecule_id_to_name = np.zeros(self.ntypes, dtype = 'U20')
        
        self.avogadro = 6.02214076e23
        self.dt = dt
        self.kB = 1.380649e-23*self.length_units_prefix[length_unit]**2/self.time_units_prefix[time_unit]**2 # J/K = N*m/K = kg*m^2/(s^2*K)
        if energy_unit == 'kJ/mol':
            self.kbt = 1.380649e-23*Temp*self.avogadro*1e-3 # kJ/mol = N*m/(mol*1000) = kg*m^2/(s^2*mol*1000)
        else:
            self.kbt = 1.380649e-23*Temp*self.avogadro*1e-3
            print('The only currently supported unit for energy is kJ/mol !')
            
        self.Temp = Temp
        self.eta = eta
        
        self.molecule_types = nb.typed.Dict.empty(key_mt, value_mt)
        
        self.Reactions_Dict = nb.typed.Dict.empty(keytype_RD, valuetype_RD)
        self.reaction_id = 0
        
        
        self.interaction_IDs = nb.typed.Dict.empty(nb.types.string, nb.int64)
        self.interaction_IDs['harmonic_repulsion'] = 0
        self.interaction_IDs['harmonic_attraction'] = 1
        self.interaction_IDs['screened_electrostatics'] = 2
        self.interaction_IDs['Lennard_Jones'] = 3
        self.interaction_IDs['Weeks_Chandler_Anderson'] = 4
        self.interaction_IDs['repulsive_membrane'] = 5
        self.interaction_IDs['CSW'] = 6
        self.interaction_IDs['PHS'] = 7
        self.interactions = False
        
        
        self.barostat = np.zeros(1, dtype = item_t_barostat)
        
        self.virial_scalar = np.zeros(1, dtype=np.float64)
        self.virial_tensor = np.zeros((1,3,3), dtype=np.float64)
        
        self.mol_type_id = nb.typed.Dict.empty(nb.types.string, nb.int64)
        self.mol_type_id['Volume'] = 0
        self.mol_type_id['Surface'] = 1
    
        self.react_type_id = nb.typed.Dict.empty(nb.types.string, nb.int64)
        self.react_type_id['bind'] = 0 # Particle only
        self.react_type_id['conversion'] = 1
        self.react_type_id['decay'] = 2 # Molecule only
        self.react_type_id['conversion_mol'] = 3
        self.react_type_id['enzymatic'] = 4
        self.react_type_id['fusion'] = 5 # Molecule only
        self.react_type_id['enzymatic_mol'] = 6
        self.react_type_id['production'] = 7
        self.react_type_id['fission'] = 8
        self.react_type_id['absorption'] = 9
        self.react_type_id['release'] = 10
        
        x = np.linspace(0,6,100) 
        cum_p = rnd.dx_cum_prob(x)
        ii = np.where(cum_p == 1.0)[0]
        
        self.dx_rand = rnd.Interpolate(rnd.dx_cum_prob(x[0:ii[1]]), x[0:ii[1]])
        
        self.n_comps = 0
        self.Compartments = nb.typed.Dict.empty(Comp_keytype, Comp_valuetype)
        self.compartments_id = nb.typed.Dict.empty(nb.types.string, nb.int64)
        self.compartments_name = nb.typed.List.empty_list(nb.types.string)
        
        self.boundary_2d = False
        
        print('System initialized')
        
    
        
    def add_box_compartment(self):
        
        self.compartments_id['Box'] = 0
        self.compartments_name.append('Box')
        
    def add_system_grid(self):
        
        """Calculates properties of the system cell grid.
        
        
        """
        
        eps = 1.0 + 1e-8 # We scale the overall grid size by this factor relative to the simulation box size. This way, the cell grid is slightly larger than the simulation box. If we don't do this, we can get out of bounds errors for points that lie exactly in the plane of a positive box boundary!
        
        self.cells_per_dim = np.empty(3, dtype = np.int64)
        
        
        radii = [self.particle_types[key][0]['radius'] for key in self.particle_types if self.particle_types[key][0]['radius']>0]
        if len(radii)>0:
            min_radius = min(radii)
            
            self.cells_per_dim[0] = int(self.box_lengths[0]/min_radius)
            self.cells_per_dim[1] = int(self.box_lengths[1]/min_radius)
            self.cells_per_dim[2] = int(self.box_lengths[2]/min_radius)
            
            if np.any(self.cells_per_dim>self.max_cells_per_dim):
                min_radius = max(self.box_lengths)/self.max_cells_per_dim
                
                self.cells_per_dim[0] = int(self.box_lengths[0]/min_radius)
                self.cells_per_dim[1] = int(self.box_lengths[1]/min_radius)
                self.cells_per_dim[2] = int(self.box_lengths[2]/min_radius)
        else:
                min_radius = max(self.box_lengths)/self.max_cells_per_dim
                
                self.cells_per_dim[0] = int(self.box_lengths[0]/min_radius)
                self.cells_per_dim[1] = int(self.box_lengths[1]/min_radius)
                self.cells_per_dim[2] = int(self.box_lengths[2]/min_radius)

        
        self.cell_length_per_dim = self.box_lengths*eps / self.cells_per_dim
        self.Ncell = self.cells_per_dim.prod()
        self.AABB_centers = self.get_AABB_centers()
    
    def add_barostat_berendsen(self, Tau_P, P0, start):
        
        """Adds a berendsen barostat to the system.
        
        Parameters
        ----------
        Tau_P : `float`
            Time constant of berendsen barostat
        P0 : `float`
            Target pressure of berendsen barostat
        start : `int`
            Simulation step from which to start simulating in NPT ensemble.
            
        Notes
        -----
        
        The Berendsen barostat is frequently used in molecular dynamics simulations due to its simplicity and fast equilibration. While the Berendsen barostat results in the correct particle density, it does not correctly sample from the (NPT) thermodynamical ensemble, i.e. the fluctuation in the pressure are not 100% accurate. Therefore, one needs to be carefull when to use this Barostat. However, if the barostat is used in the preparation step but not during sampling, e.g. when doing LLPs simualtions where the barostat is used to relax the system's shape but afterwards one simulates in the NVT ensemble :cite:t:`Muller2020`, this should not be a problem. 
        Note: :cite:t:`Baruffi2019` introduced a barostat for overdamped Langevin dynamics that settles to the correct thermodynamic equilibrium distribution.
        
        """
        
        self.barostat[0]['active']=True
        self.barostat[0]['Tau_P']=Tau_P
        self.barostat[0]['P0']=P0
        self.barostat[0]['start']=start
        
        
    def update_rb_barostat(self, RBs, Particles):
        
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
        
        for i0 in range(RBs.occupied.n):
            i = RBs.occupied[i0] 
            
            RBs.rescale_pos(Particles, self, self.barostat[0]['mu'],i)
            
            
        
    def register_molecule_type(self, molecule_name, particle_pos, particle_types, collision_type = 0, h_membrane = None):
        
        """Regsiters a molecule type.
        
        Parameters
        ----------
        molecule_name : `str`
            Name of the molecule.
        particle_pos : `int[:,3]`
            Positions of each of the molecule's particles.
        particle_types : `array_like (dtype = 'U20')`
            Array of names for each of the molecule's particles.
        collision_type : `int {0,1}`
            Collision type of the molecule (Default = 0). 0: Collisions with compartments are handled via interaction force. 1: Collisions with compartments are handled by raytracing algorithm (recommended for molcules whose diffusion speed and/or interaction radius is small compared to the chosen integration time step).
        
        """
        
        if collision_type == 0 and (self.mesh == True or self.boundary_condition == 'repulsive'):
            self.interactions = True
        
        particle_radii = np.empty(len(particle_types), dtype = np.float64)
        for i, ptype in enumerate(particle_types):
            particle_radii[i] = self.particle_types[str(ptype)][0]['radius']
        
        self.molecule_types[molecule_name] = MoleculeType(particle_pos, particle_types, particle_radii, molecule_name, collision_type, self, h_membrane)

        self.um_reaction_id = np.empty(self.ntypes, dtype = np.int64)
        self.um_reaction_id[:] = -1
        
        
        molecule_id_to_name0 = np.empty(self.ntypes_rb+1, dtype = 'U20')
        for i,name in enumerate(self.molecule_id_to_name):
            molecule_id_to_name0[i] = name
        molecule_id_to_name0[self.ntypes_rb] = molecule_name
        self.molecule_id_to_name = molecule_id_to_name0
        
        self.virial_scalar = np.zeros(self.ntypes_rb+1, dtype = np.float64)
        self.virial_scalar_Wall = np.zeros(self.ntypes_rb+1, dtype = np.float64)
        
        self.virial_tensor = np.zeros((self.ntypes_rb+1,3,3), dtype = np.float64)
        self.virial_tensor_Wall = np.zeros((self.ntypes_rb+1,3,3), dtype = np.float64)
        
        self.Pressure = np.zeros(self.ntypes_rb+1, dtype = np.float64)
        self.Nmol = np.zeros(self.ntypes_rb+1, dtype = np.int64)
        
        self.Epot = np.zeros(self.ntypes_rb+1, dtype = np.float64)
        
        self.ntypes_rb += 1
        
        print('Molecule type '+molecule_name+' added to system')
        
        
    def set_diffusion_tensor(self, molecule_name, D_tt, D_rr, mu_tb, mu_rb, mu_tb_sqrt, mu_rb_sqrt):
        
        """A brief description of what the function (method in case of classes) is and what it’s used for
        
        Parameters
        ----------
        molecule_name : `str`
            Some Information
        D_tt : `float[3,3]`
            Translational diffusion tensor ()
        D_rr : `float[3,3]`
            Rotational diffusion tensor
        mu_tb : `float[3,3]`
            Translational mobility tensor
        mu_rb : `float[3,3]`
            Rotational mobility tensor
        mu_tb_sqrt : `float[3,3]`
            Square root of translational mobility tensor
        mu_rb_sqrt : `float[3,3]`
            Square root of rotational mobility tensor
        
        Notes
        -----
        PyRID offers a module for the calculation of the diffusion tensors of rigid bead models (see molecules_util.hydro_util). It uses the Kirkwood-Riseman calculation with modified Oseen tensor as introduced by :cite:t:`Carrasco1999`, :cite:t:`Carrasco1999a`. The same method has been implemented in the `HYDRO++ <http://leonardo.inf.um.es/macromol/programs/hydro++/hydro++.htm>`_ tool, which is free to use but not open source! As such, you may also use HYDRO++ for the calculation of the diffusion tensors. Or even better, use both and tell me if you you get inconsistent results ;) !
        
        .. admonition:: Note
        
            Diffusion tensor should be a real positive semidefinite matrix!
        
        """
      
        # is_diagonal(D_tt)
        # is_diagonal(D_rr)
        
        self.molecule_types[molecule_name].D_tt = np.ascontiguousarray(D_tt)
        self.molecule_types[molecule_name].D_rr = np.ascontiguousarray(D_rr)
        
        self.molecule_types[molecule_name].mu_tb = mu_tb
        self.molecule_types[molecule_name].mu_rb = mu_rb
        
        self.molecule_types[molecule_name].mu_tb_sqrt = mu_tb_sqrt
        self.molecule_types[molecule_name].mu_rb_sqrt = mu_rb_sqrt
        
        self.molecule_types[molecule_name].r_mean = 2*np.sqrt(4*np.trace(D_tt)/3*self.dt/np.pi)
        self.molecule_types[molecule_name].r_mean_pair = 2*np.sqrt(8*np.trace(D_tt)/3*self.dt/np.pi)
        
        self.molecule_types[molecule_name].Dtrans = np.trace(D_tt)/3 #in mum^2/s
        self.molecule_types[molecule_name].Drot = np.trace(D_rr)/3 #in 1/s
        
        self.molecule_types[molecule_name].Dtrans_2d = np.trace(D_tt[0:2,0:2])/2
        self.molecule_types[molecule_name].Drot_2d = np.trace(D_rr[0:2,0:2])/2
        
    def fixed_concentration_at_boundary(self, molecule_name, C, comp_name, location):
        
        """Sets the fixed concentration boundary given a concentartion for each defined molecule type and for each compartents volume and surface.
        
        Parameters
        ----------
        molecule_name : `str`
            Name of the molecule
        C : `float`
            concentration of molecule type outside simulation box ('virtual molecules')
        comp_name : `str`
            Name of the compartment
        location : `int {0,1}`
            Location of the molecule. 0: Volume, 1: Surface
        
        Notes
        -----
        fixed_concentration_at_boundary() calculates some properties that are necessary to properly distribute molecules inside the simulation box that hit the simulation box boundary from the outside (Thereby, 'virtual molecules' become `real` molecules in our simualtion). The number of hits per time step a boundary of area A experiences is :math:`N = l_{perp}*A*C`. Where :math:`C` is the concentration in molecules per volume and :math:`l_{perp}` is the average net displacement in one tiem step towards or away from any plane, where :math:`l_{perp} = \sqrt{(4*D*\Delta t/\pi)}` :cite:t:`Stiles2001`.
        
        """
        
        comp_id = self.compartments_id[str(comp_name)]
        
        if location == 0:
            D = self.molecule_types[molecule_name].Dtrans
        else:
            D = self.molecule_types[molecule_name].Dtrans_2d
        
        if self.mesh == False:
            
            self.box_border = np.zeros(3, dtype = item_t_border_box)
                    
            area =  np.array([self.box_lengths[1]*self.box_lengths[2],  self.box_lengths[0]*self.box_lengths[2], self.box_lengths[0]*self.box_lengths[1]])
            
            self.box_border['cum_area'][:] = np.cumsum(area)
            
            self.box_area = 2*np.sum(area)
            
            # A_x = self.box_lengths[1]*self.box_lengths[2]
            # A_y = self.box_lengths[0]*self.box_lengths[2]
            # A_z = self.box_lengths[0]*self.box_lengths[1]
            
            
        # A = self.box_area
            
        l_perp = np.sqrt(4*D*self.dt/np.pi)
        
        self.molecule_types[molecule_name].l_perp = l_perp
        
        if location == 0:
            self.molecule_types[molecule_name].concentration[comp_id] = C
        else:
            self.molecule_types[molecule_name].concentration_surface[comp_id] = C
        

        
    def register_particle_type(self, Type, radius):
        
        """Registers a particle type.
        
        Parameters
        ----------
        Type : `str`
            Name of the particle type.
        radius : `float`
            radius of the particle.
        
        """

        self.particle_types[Type] = np.zeros(1, dtype = item_t_ptype)#np.array([(self.ntypes, radius)], dtype = item_t_ptype)
        self.particle_types[Type][0]['id'] = self.ntypes
        self.particle_types[Type][0]['radius'] = radius
        
        particle_id_to_name0 = np.empty(self.ntypes+1, dtype = 'U20')
        for i,types in enumerate(self.particle_id_to_name):
            particle_id_to_name0[i] = types
        particle_id_to_name0[self.ntypes] = Type
        self.particle_id_to_name = particle_id_to_name0
        
        self.ntypes += 1
        
        self.up_reaction_id = np.empty(self.ntypes, dtype = np.int64)
        self.up_reaction_id[:] = -1
        self.bp_reaction_ids = np.zeros(self.ntypes, dtype = item_t_bpr)
        
        self.interaction_args = np.zeros((self.ntypes, self.ntypes), dtype = item_t_inter)
        
        self.reaction_args = np.zeros((self.ntypes, self.ntypes), dtype = item_t_react)
        self.reaction_args['id'][:,:] = -1
        
        self.N_bonds = np.zeros((self.ntypes, self.ntypes), dtype = np.int64)
        
        self.interaction_defined  = np.zeros((self.ntypes, self.ntypes), dtype = nb.types.bool_)
        self.pair_interaction = np.zeros(self.ntypes, dtype = nb.types.bool_)
        
        # self.force_measure = np.zeros(self.ntypes, dtype = np.int64)
        
        print('Particle type '+Type+' added to system')
        

    
    def set_compartments(self, comp_name, triangle_ids):
        
        """Sets a compartment given its triangle ids.
        
        Parameters
        ----------
        comp_name : `str`
            Name of the compartment.
        triangle_ids : `int64[:]`
            IDs of the triangles making up the compartment mesh.
        
        """
        
        self.n_comps += 1
        
        self.Compartments[self.n_comps] = Compartment(triangle_ids, comp_name, self.n_comps, self)
                
        self.compartments_id[str(comp_name)] = self.n_comps
        self.compartments_name.append(str(comp_name))
        
        # self.Compartments[self.n_comps].calc_centroid_radius_AABB(self)    
        
        self.volume -= self.Compartments[self.n_comps].volume
        
        for dim in range(3):
            
            if self.Compartments[self.n_comps].AABB[0][dim] < self.AABB_all[0][dim]:
                self.AABB_all[0][dim] = self.Compartments[self.n_comps].AABB[0][dim]
                
            if self.Compartments[self.n_comps].AABB[1][dim] > self.AABB_all[1][dim]:
                self.AABB_all[1][dim] = self.Compartments[self.n_comps].AABB[1][dim]
        
        
        print('Compartment '+ comp_name +' set!') 
        

    def add_border_3d(self, triangle_ids_border):
        
        """Registers the border faces of an intersection between compartments and the simulation box (Needed for fixed concentration boundary condition)
        
        Parameters
        ----------
        triangle_ids_border : `int64[:]`
            IDs of the triangles making up the compartment-simulation box border.
        
        """
        
        eps = 1e-3
        
        self.box_border = np.zeros(len(triangle_ids_border), dtype = item_t_border_box)
        
        for i in range(len(triangle_ids_border)):

            tri_id = triangle_ids_border[i]
            self.box_border[i]['triangle_ids'] = tri_id
            
            # Make sure the box vertices lie exactly on the simulation box border:
            triangle = self.Mesh[tri_id]['triangles']
            for vert in range(3):
                for dim in range(3):
                    if 1-eps<abs(self.vertices[triangle[vert]][dim]/(self.box_lengths[dim]/2))<1+eps:
                        self.vertices[triangle[vert]][dim] = np.sign(self.vertices[triangle[vert]][dim])*self.box_lengths[dim]/2
                        self.update_mesh(vertex = triangle[vert])
                        
                    
        self.box_border['cum_area'][:] = np.cumsum(self.Mesh[triangle_ids_border]['triangle_area'])
        
        self.box_area = np.sum(self.Mesh[triangle_ids_border]['triangle_area'])
        
        
    def add_mesh(self, vertices, triangles, mesh_scale = 1.0, box_triangle_ids = None, triangle_ids_transparent = None):
        
        """Set up the mesh for the mesh compartments.
        
        Parameters
        ----------
        vertices : `float64[:,3]`
            Position of mesh vertices in each dimension.
        triangles : `int64[:,3]`
            Indices of vertices that make up each triangle (pointers to vertices)
        mesh_scale : `float`
            Scaling factor to scale up or down the mesh (Default = 1.0).
        box_triangle_ids : `int64[:]`
            Triangle ids of the faces that make up the simulation box border (To be excluded from self.triangle_ids)
        triangle_ids_transparent : `int64[:]`
            Triangle ids of any transparent faces (To be excluded from self.triangle_ids)
        
        Notes
        -----
        
        A Compartment is defined by a triangulated manifold mesh (a mesh without holes and disconnected vertices or edges),
        i.e. it has no gaps and separates the space on the inside of the surface from the space outside. It also looks like a surface everywhere on the mesh :cite:t:`Shirley2009`.
        The mesh is stored as a shared vertex mesh. This data structure has triangles which point to vertices which contain the vertex data:
            
            | triangles: Three vectors per triangle
            | 0 | (0,1,2)
            | 1 | (1,3,2)
            | 2 | (0,3,1)
            | 
            | vertices: One vector per vertex
            | 0 | (a_x,a_y,a_z)
            | 1 | (b_x,b_y,b_z)
            | 2 | (c_x,c_y,c_z)
            | 3 | (d_x,d_y,d_z)
        
        During raymarching on a mesh surface, we need to find the neighbor triangle given some edge i that has been crossed.
        We may save, for each triangle, the corresponding edges in an array:
            
            | 0 | (0,1), (0,2), (1,2)
            | 1 | ...
            
        In addition, we save, for each edge of a triangle the corresponding neighbor in another array:
            
            | 0 | (1, 4, 2)
            | 1 | ...
        
        Also, we need to rotate the particles between local coordinate systems of neighbouring triangles, as particles cross an edge.
        We may precalculate the necessary parameters and save them in a datastructure similar to the above:
            
            | 0 | (params_0->1, params_0->4, params_0->2)
            | 1 | ...
            | 
            | params_i->j : [sinPhi, cosPhi, a_n]
            
        We may also want to prcompute the rotation parameters for each triangle pait to speed up the coordiante transformation 
        when a particle hops between two triangles. We may take the same data structure as before:
            
            | 0 | (rot_params[0->1], (rot_params[0->4], (rot_params[0->2])
            | 1 | ...
            | 
            | rot_params: array[sin, cos, an]
        
        """
        
        vertices*=mesh_scale
        self.mesh_scale = mesh_scale
        
        self.mesh = True
        self.vertices = vertices
        self.Mesh = np.zeros(len(triangles), dtype = item_t_mesh)
        self.N_triangles = len(triangles)
        
        if triangle_ids_transparent is not None:
            self.triangle_ids_transparent = triangle_ids_transparent
        
        triangle_ids_all = np.arange(self.N_triangles, dtype = np.int64)
        if box_triangle_ids is not None and triangle_ids_transparent is not None:
            self.triangle_ids = np.array([tri_id for tri_id in triangle_ids_all if (tri_id not in box_triangle_ids and tri_id not in triangle_ids_transparent)], dtype = np.int64)
            
        elif box_triangle_ids is None and triangle_ids_transparent is None:
            self.triangle_ids = triangle_ids_all
        
        elif box_triangle_ids is not None:
            self.triangle_ids = np.array([tri_id for tri_id in triangle_ids_all if tri_id not in box_triangle_ids], dtype = np.int64) 
        elif triangle_ids_transparent is not None:
            self.triangle_ids = np.array([tri_id for tri_id in triangle_ids_all if tri_id not in triangle_ids_transparent], dtype = np.int64)
            
            
        
        #Assign values:
        for tri_id in range(self.N_triangles):
            self.Mesh[tri_id]['triangles'][:] = triangles[tri_id]
            p0,p1,p2 = vertices[triangles[tri_id][0]], vertices[triangles[tri_id][1]], vertices[triangles[tri_id][2]]
            origin, ex,ey,ez = local_coord(p0,p1,p2)
            self.Mesh[tri_id]['triangle_coord'][0,:] = origin
            self.Mesh[tri_id]['triangle_coord'][1,:] = ex
            self.Mesh[tri_id]['triangle_coord'][2,:] = ey
            self.Mesh[tri_id]['triangle_coord'][3,:] = ez # normalied triangle normal
            
            d00, d01, d11, denom = barycentric_params(p0, p1, p2)
            self.Mesh[tri_id]['barycentric_params'][0] = d00
            self.Mesh[tri_id]['barycentric_params'][1] = d01
            self.Mesh[tri_id]['barycentric_params'][2] = d11
            self.Mesh[tri_id]['barycentric_params'][3] = denom
            
            self.Mesh[tri_id]['normal'][:] = normal_vector(p0,p1,p2) # unnormalied triangle normal
            
            self.Mesh[tri_id]['triangle_centroid'][:] = triangle_centroid(p0,p1,p2)
            self.Mesh[tri_id]['triangle_area'] = triangle_area(p0,p1,p2)
            self.area += self.Mesh[tri_id]['triangle_area']
            
            self.Mesh[tri_id]['triangle_distance'] = np.dot(self.Mesh[tri_id]['triangle_coord'][3], self.vertices[self.Mesh[tri_id]['triangles'][0]])
            
        self.add_edges() #Given an edge, which two triangles share it?
        self.add_neighbours() #Given a triangle, what are the (three) adjacent triangles?
        self.add_vertex_triangles()
        
        if self.wall_force == 0.0:
            print('Warning: No wall_force set, molecules wont get repelled by the mesh!')
        
        
            
            
            
#%%

    def update_mesh(self, vertex = None, triangle_ids_list = None):

        """
        Updates mesh properties. Should be called whever the mesh topology changes, i.e. the position of a vertex is changed after the mesh has been initialized (System.add_mesh()).
        
        Parameters
        ----------
        vertex : `int64`
            The index of the vertex that has been changed. (Default = None)
        triangle_ids_list : `int64[:]`
            List of triangle ids which to update. (Default = None)
            
        """
        
        if vertex is not None:
            triangle_ids = self.vertex_triangles[vertex]
        elif triangle_ids_list is not None:
            triangle_ids = triangle_ids_list
            
        #Assign values:
        for tri_id in triangle_ids:
            
            triangle = self.Mesh[tri_id]['triangles']
            
            p0 = self.vertices[triangle[0]]
            p1 = self.vertices[triangle[1]]
            p2 = self.vertices[triangle[2]] 
            
            origin, ex,ey,ez = local_coord(p0,p1,p2)
            self.Mesh[tri_id]['triangle_coord'][0,:] = origin
            self.Mesh[tri_id]['triangle_coord'][1,:] = ex
            self.Mesh[tri_id]['triangle_coord'][2,:] = ey
            self.Mesh[tri_id]['triangle_coord'][3,:] = ez # normalied triangle normal
            
            d00, d01, d11, denom = barycentric_params(p0, p1, p2)
            self.Mesh[tri_id]['barycentric_params'][0] = d00
            self.Mesh[tri_id]['barycentric_params'][1] = d01
            self.Mesh[tri_id]['barycentric_params'][2] = d11
            self.Mesh[tri_id]['barycentric_params'][3] = denom
            
            self.Mesh[tri_id]['normal'][:] = normal_vector(p0,p1,p2) # unnormalied triangle normal
            
            self.Mesh[tri_id]['triangle_centroid'][:] = triangle_centroid(p0,p1,p2)
            self.Mesh[tri_id]['triangle_area'] = triangle_area(p0,p1,p2)
            self.area += self.Mesh[tri_id]['triangle_area']
        
            self.Mesh[tri_id]['triangle_distance'] = np.dot(self.Mesh[tri_id]['triangle_coord'][3], self.vertices[self.Mesh[tri_id]['triangles'][0]])
            
            

    def add_edges(self):
        
        """Finds the edges of each triangle and adds them to the Mesh struct.
        
        Notes
        -----

        Edges are sorted in counter clockwise order:
        
            | edge:   [0,1]
            |         [1,2]
            |         [0,2]
        
        """
                
        #Given triangle i and the jth edge of that triangle, which is the neighboring triangle?
        
        for tri_id in range(self.N_triangles):
            self.Mesh[tri_id]['edges'][0]=np.array([self.Mesh[tri_id]['triangles'][0], self.Mesh[tri_id]['triangles'][1]])
            self.Mesh[tri_id]['edges'][1]=np.array([self.Mesh[tri_id]['triangles'][1], self.Mesh[tri_id]['triangles'][2]])
            self.Mesh[tri_id]['edges'][2]=np.array([self.Mesh[tri_id]['triangles'][0], self.Mesh[tri_id]['triangles'][2]])
        
        
    def add_vertex_triangles(self):
        
        self.vertex_triangles = nb.typed.List.empty_list(list_int64)
        for _ in range(len(self.vertices)):
            self.vertex_triangles.append(nb.typed.List.empty_list(nb.int64))
        
        for tri_id in range(self.N_triangles):
            
            triangle = self.Mesh[tri_id]['triangles']
            
            for vertex_id in triangle:
                self.vertex_triangles[vertex_id].append(tri_id)
            
            
    def add_neighbours(self):
        
        """Finds the neighbours of each triangle and adds them to the Mesh struct.
        
        """
        
        Vertex = {}
        for tri_id in range(self.N_triangles):
            triangle = self.Mesh[tri_id]['triangles']
            for j in range(3):
                if triangle[j] not in Vertex:
                    Vertex[triangle[j]] = nb.typed.List([tri_id])
                else:
                    Vertex[triangle[j]].append(tri_id)
        
        
        for tri_id in range(self.N_triangles):
            triangle = self.Mesh[tri_id]['triangles']
            for j, k1, k2 in [[0, triangle[0], triangle[1]], [1, triangle[1], triangle[2]], [2, triangle[0], triangle[2]]]:
                
                a = set(Vertex[k1])-set([tri_id])
                b = set(Vertex[k2])-set([tri_id])
                neighb = list(a-(a-b))[0]
                
                self.Mesh[tri_id]['neighbours'][j] = neighb
                
    
    #%%
    
    def get_AABB_centers(self):
        
        """Calculates the center of each cell of the system grid.
        
        """
        
        AABB_centers = np.zeros((self.Ncell,3))
        
        for cx in range(self.cells_per_dim[0]):
            for cy in range(self.cells_per_dim[1]):
                for cz in range(self.cells_per_dim[2]):
                    AABB_center = np.array([(cx+0.5)*self.cell_length_per_dim[0], (cy+0.5)*self.cell_length_per_dim[1], (cz+0.5)*self.cell_length_per_dim[2]])
                    AABB_center += self.origin
                    
                    c = cx + cy * self.cells_per_dim[0] + cz * self.cells_per_dim[0] * self.cells_per_dim[1]
                    AABB_centers[c] = AABB_center
                            
        return AABB_centers
    
    
    def create_cell_list(self):
        
        """Creates the linked cell list for the triangulated meshes by calling system_util.CellList_util.create_cell_list_mesh() (Used during collision detection).
        
        """
        
        if self.boundary_2d:
            centroid_in_box = True
        else:
            centroid_in_box = False
        
        self.CellList, self.AABB_centers = create_cell_list_mesh(self, self, self.cells_per_dim, self.cell_length_per_dim, self.triangle_ids, centroid_in_box, False)


#%%

    def add_up_reaction(self, reaction_type, educt_type, rate, product_types, product_loc, product_direction, radius):
        
        """Registeres a Uni-Particle (UP) reaction for an educt particle.
        
        Parameters
        ----------
        reaction_type : `str` {'relase', 'conversion'}
            Name of the reaction type
        educt_type : `str`
            Name of the educt particle type
        rate : `float64`
            Reaction rate
        product_types : `array_like`
            List of products
        product_loc : `array_like`
            List specifying whether the individual products are volume (0) or surface (1) molecules 
        product_direction : `array_like`
            Specifies whether, in case of a surface particle releasing products into the volume, the realease occurs inside (-1) or outside (1) the mesh compartment.
        radius : `float64`
            The radius within which molecules are distributed upon a successfull release reaction.
            
        Notes
        -----
        Uni-particle reactions are evaluated using a variant of the Gillespie stochastic simulation algorithm. Thereby, for a particle type, the time point of the next uni-particle reaction occuring is calculated in advance and executed when the simualtion reaches the respective time point. 
        For unimolecular reactions we can draw the time point of the next reaction from a distribution using a variant of the Gillespie Stochastic Simulation Algorithm (SSA) :cite:t:`Stiles2001`, :cite:t:`Erban2007`. For a single molecule having :math:`n` possible transition reactions/reaction paths each having a reaction rate :math:`k_i`, let :math:`k_t = \sum_i^n k_i` be the total reaction rate. 
        
        Now, let :math:`\\rho(\\tau) d\\tau` be the probability that the next reaction occurs within :math:`[t+\\tau,t+\\tau+d\\tau)` and let :math:`g(\\tau)` be the probability that no reaction occurs within :math:`[t,t+\\tau)`. The probability that a reaction occurs within the time interval :math:`d\\tau` is simply :math:`k_t d\\tau`. Thereby
        
        .. math::
            
            \\rho(\\tau) d\\tau = g(\\tau) k_t d\\tau
        
        where :math:`g(\\tau) = e^{-k_t \\tau}` (see also Poisson process).
        From the above equation we easily find :math:`P(\\tau) = 1-e^{-k_t \\tau}` by integration.
        To sample from this distribution, we can use the inverse distribution function.
        
        .. math::
            
            \\tau = P^{-1}(U)
            
        where :math:`U` i uniformly distributed in :math:`(0,1)`. 
        Given that :math:`U = P(\\tau) = 1-e^{-k_t \\tau}`, we find :math:`P^{-1}(U) = \\frac{-log(1-U)}{k_t}`. Since 
        :math:`U` is uniformly distributed in 0,1, so is :math:`1-U`. Thereby, we can draw the time point of the next reaction from:
            
        .. math::
            \\tau = \\frac{1}{k_t} \ln\Big[\\frac{1}{U}\Big],
        
        With the above method, we accurately sample from the distribution of expected molecule lifetimes :math:`\\rho(\\tau) = k_t e^{-k_t \\tau}`.
        
        A uni-particle reaction can consist of several reaction paths. Each particle type has exactly one uni-particle reaction id registered for which the time point is drawn randomly based on the total reaction rate which is the sum of all path reaction rates. At the time point of the reaction occuring, a path is randomly drawn from the set of possible paths weighted by the individual path reaction rates. 
        To get the reaction path we compare a second random number, uninformly distributed in (0,k_t), with the cumulative set of reaction rates :math:`(k_1, k_1+k_2, ..., k_t)`. The comparison is done via a bisection algorithm (see rand_util.random_util.bisect_right()).
        
        To illustrate the above by an example: For particle 'A' a we may define two convcersion reactions where A -k1-> B, A -k2-> C. In addition, we define a release reaction A -k3-> 2*D+5*E. The next reaction will occur at a time point T that is calculated based on the total rate k1+k2+k3. At time T, one of the three reactions is executed. The probability for each reaction to be selected being k1/(k1+k2+k3), k2/(k1+k2+k3), k3/(k1+k2+k3).
        
        """
        
        # Get the educt id:
        educt_id = self.particle_types[educt_type][0]['id']
        
        if self.up_reaction_id[educt_id] == -1:
            
            self.up_reaction_id[educt_id] = self.reaction_id
            self.particle_types[educt_type][0]['UP_reaction'] = True
            
            self.Reactions_Dict[self.reaction_id] = rru.Reaction(np.array([educt_id]), 'unimolecular', 'Particle', None, None)
        
            self.reaction_id += 1

        if  reaction_type == 'release':
            # In case of a release reaction, we have a special case, since the educt is a particle and will therefore change to a different particle type whereas the other products are rigid bead molecules:
                
            if product_types is not None and product_loc is not None and product_direction is not None and  radius is not None:
                
                products_ids = np.empty(2, dtype = np.int64)
                
                educt_product_id = self.particle_types[str(product_types[0])][0]['id']
                products_ids[0] = educt_product_id
                product_id = self.molecule_types[str(product_types[1])].type_id
                products_ids[1] = product_id
                
                self.Reactions_Dict[self.up_reaction_id[educt_id]].add_path(self, reaction_type, rate, products_ids, product_loc, product_direction, radius)
        
        elif reaction_type == 'conversion':
            
            if product_types is not None:
                products_ids = np.empty(1, dtype = np.int64)
                products_ids[0] = self.particle_types[str(product_types[0])][0]['id']
                
                self.Reactions_Dict[self.up_reaction_id[educt_id]].add_path(self, reaction_type, rate, products_ids)
            
        elif reaction_type == 'decay':
            
            self.Reactions_Dict[self.up_reaction_id[educt_id]].add_path(self, reaction_type, rate)
        
        
        self.particle_types[educt_type][0]['transition_rate_total'] += rate
        

#%%


    def add_um_reaction(self, reaction_type, educt_type, rate, product_types, product_loc, product_direction, radius):
        
        """Registeres a Uni-Molecular (UM) reaction for an educt Molecule.
        
        Parameters
        ----------
        reaction_type : `str` {'production', 'fission', 'conversion_mol', 'decay'}
            Name of the reaction type
        educt_type : `str`
            Name of the educt molecule type
        rate : `float64`
            Reaction rate
        product_types : `array_like`
            List of products
        product_loc : `array_like`
            List specifying whether the individual products are volume (0) or surface (1) molecules 
        product_direction : `array_like`
            Specifies whether, in case of a surface molecule releasing products into the volume, the realease occurs inside (-1) or outside (1) the mesh compartment.
        radius : `float64`
            The radius within which molecules are distributed upon a successfull release reaction.
            
        Notes
        -----
        Uni-molecular reactions are evaluated using the Gillespie stochastic simulation algorithm. Thereby, for a molecule type, the time point of the next uni-molecular reaction occuring is calculated in advance and executed when the simualtion reaches the respective time point. A uni-molecular reaction can consist of several reaction paths. Each molecule type has exactly one uni-molecular reaction id registered for which the time point is drawn randomly based on the total reaction rate which is the sum of all path reaction rates. At the time point of the reaction occuring, a path is randomly drawn from the set of possible paths weighted by the individual path reaction rates. To illustrate the above by an example: For Molecule 'A' a we may define two convcersion reactions where A -k1-> B, A -k2-> C. In addition, we define a fission reaction A -k3-> D+E. The next reaction will occur at a time point T that is calculated based on the total rate k1+k2+k3. At time T, one of the three reactions is executed. The probability for each reaction to be selected being k1/(k1+k2+k3), k2/(k1+k2+k3), k3/(k1+k2+k3).
        
        """
        
        # Get the educt id:
        educt_id = self.molecule_types[str(educt_type)].type_id
        
        if self.um_reaction_id[educt_id] == -1:
            
            self.um_reaction_id[educt_id] = self.reaction_id
            
            self.Reactions_Dict[self.reaction_id] = rru.Reaction(np.array([educt_id]), 'unimolecular', 'Molecule', None, None)
        
            self.reaction_id += 1
            

        if  reaction_type == 'production':
            
            if product_types is not None and product_loc is not None and product_direction is not None and  radius is not None:
                
                products_ids = np.empty(len(product_types), dtype = np.int64)
                
                for i,prod in enumerate(product_types):
                    product_id = self.molecule_types[str(prod)].type_id
                    products_ids[i] = product_id
                
                self.Reactions_Dict[self.um_reaction_id[educt_id]].add_path(self, reaction_type, rate, products_ids, product_loc, product_direction, radius)
        
        elif  reaction_type == 'fission':
                
            if product_types is not None and product_loc is not None and product_direction is not None and  radius is not None:
                
                products_ids = np.empty(2, dtype = np.int64)
                
                for i,prod in enumerate(product_types):
                    product_id = self.molecule_types[str(prod)].type_id
                    products_ids[i] = product_id
                
                self.Reactions_Dict[self.um_reaction_id[educt_id]].add_path(self, reaction_type, rate, products_ids, product_loc, product_direction, radius)
            
        elif reaction_type == 'conversion_mol':
            
            if product_types is not None:
                
                products_ids = np.empty(1, dtype = np.int64)
                products_ids[0] = self.particle_types[str(product_types[0])][0]['id']
                
                self.Reactions_Dict[self.um_reaction_id[educt_id]].add_path(self, reaction_type, rate, products_ids)
            
        elif reaction_type == 'decay':
            
            self.Reactions_Dict[self.um_reaction_id[educt_id]].add_path(self, reaction_type, rate)
        
        
        self.molecule_types[str(educt_type)].update_um_reaction(rate)


#%%


    def add_bp_reaction(self, reaction_type, educt_types, product_types, rate, radius, interaction_type = None, interaction_parameters = None):

        """Registeres a Bi-Particle (BP) reaction for an educt particle.
        
        Parameters
        ----------
        reaction_type : `str` {'bind', 'absorption', 'enzymatic'}
            Name of the reaction type
        educt_type : `array_like`
            Names of the educt particle types
        product_types : `array_like`
            List of products (only used for binding reactions)
        rate : `float64`
            Reaction rate
        radius : `float64`
            Reaction radius.
        interaction_type : `str`
            Name of the energy potential function in case 'reaction_type' is a binding reaction.
        interaction_parameters : `dict`
            Parameters for the interaction function in case 'reaction_type' is a binding reaction.
        
        Notes
        -----
        Bi-particle reactions are evaluated using the Doi scheme. Thereby, two educt particles react with a reaction rate k if the inter-particle distance is below the reaction radius.
        
        `Binding Reactions`
        
        A particle pair will only enter a bound state if the binding reaction was succesfull. Otherwise two particles won't experience any pairwise interaction force! A reaction occurs if two particles for whose type a binding reaction has been defined are within the reaction radius. The reaction probability is evaluated as:
            
        .. math::
            
            1-exp(\lambda \cdot \Delta t),
        
        where :math:`\lambda` is the reaction rate and :math:`\Delta t` is the integration time step.
        
        
        """
        
        educts_ids = np.empty(2, dtype = np.int64)
        for i,educt in enumerate(educt_types):
            educt_id = self.particle_types[str(educt)][0]['id']
            educts_ids[i] = educt_id


        if self.reaction_args[educts_ids[0],educts_ids[1]]['id'] == -1:
            
            if educts_ids[0]!=educts_ids[1]:
                for i in range(2):
                    n_reactions = self.bp_reaction_ids[educts_ids[i]]['n_reactions']
                    self.bp_reaction_ids[educts_ids[i]]['ids'][n_reactions] = self.reaction_id
                    self.bp_reaction_ids[educts_ids[i]]['n_reactions'] += 1
            else:
                n_reactions = self.bp_reaction_ids[educts_ids[0]]['n_reactions']
                self.bp_reaction_ids[educts_ids[0]]['ids'][n_reactions] = self.reaction_id
                self.bp_reaction_ids[educts_ids[0]]['n_reactions'] += 1                
            
            # update the reaction_args matrix, which is used to lookup all the necessary properties for each reaction
            # during reaction handling (see update_util):
            self.reaction_args[educts_ids[0],educts_ids[1]]['id'] = self.reaction_id
            self.reaction_args[educts_ids[1],educts_ids[0]]['id'] = self.reaction_id
            self.reaction_args[educts_ids[0],educts_ids[1]]['type_BP_BM'] = 'BP'
            self.reaction_args[educts_ids[1],educts_ids[0]]['type_BP_BM'] = 'BP'
            self.reaction_args[educts_ids[0],educts_ids[1]]['defined'] = True
            self.reaction_args[educts_ids[1],educts_ids[0]]['defined'] = True
            self.reaction_args[educts_ids[0],educts_ids[1]]['radius'] = radius
            self.reaction_args[educts_ids[1],educts_ids[0]]['radius'] = radius       
            # Which of the two educts is the enzyme?:
            self.reaction_args[educts_ids[0],educts_ids[1]]['enzyme'] = educts_ids[1] 
            self.reaction_args[educts_ids[1],educts_ids[0]]['enzyme'] = educts_ids[1]
    
            self.interaction_defined[educts_ids[0],educts_ids[1]] = True
            self.interaction_defined[educts_ids[1],educts_ids[0]] = True
            self.pair_interaction[educts_ids[0]] = True
            self.pair_interaction[educts_ids[1]] = True
            
            self.interactions = True
    
            if radius/2 > self.particle_types[str(educt_types[0])][0]['cutoff']:
                self.particle_types[str(educt_types[0])][0]['cutoff'] = radius/2
    
            if radius/2 > self.particle_types[str(educt_types[1])][0]['cutoff']:
                self.particle_types[str(educt_types[1])][0]['cutoff'] = radius/2
            
            #-------------
            
            self.Reactions_Dict[self.reaction_id] = rru.Reaction(educts_ids, 'bimolecular', 'Particle', None, radius)
            current_reaction_id = self.reaction_args[educts_ids[0],educts_ids[1]]['id']
            
            self.reaction_id += 1
            
            
        else:
            
            
            if self.reaction_args[educts_ids[0],educts_ids[1]]['type_BP_BM'] != 'BP':
                raise ValueError('Unable to add the Biparticle reaction! A Bimolecular reaction has already been defined betwen these two particle types!')
                
            elif reaction_type == 'bind':
                
                raise ValueError('Unable to add the binding reaction! A Biparticle reaction has already been defined betwen these two particle types! Binding reactions cannot be defined in parallel with other biparticle reactions!')
                
            elif self.reaction_args[educts_ids[0],educts_ids[1]]['bond'] == True:
                
                raise ValueError('Unable to add the Biparticle reaction! A binding reaction has already been defined betwen these two particle types! Binding reactions cannot be defined in parallel with other biparticle reactions!')
                
            print('A new reaction path is added. Note that any new reaction radius that has been passed will be ignored!')
                
                
            current_reaction_id = self.reaction_args[educts_ids[0],educts_ids[1]]['id']
            


        if reaction_type == 'bind':
            
            products_ids = np.empty(2, dtype = np.int64)
            products_ids[0] = self.particle_types[str(product_types[0])][0]['id']
            products_ids[1] = self.particle_types[str(product_types[1])][0]['id']
            
            self.reaction_args[educts_ids[0],educts_ids[1]]['bond'] = True
            self.reaction_args[educts_ids[1],educts_ids[0]]['bond'] = True
            
            # If the particles change their type by binding, we also want to save the 
            # original type (bond_educt_id).
            # If a bond breaks, the otiginal pyrticle type will be restored.
            self.particle_types[str(product_types[0])][0]['bond_educt_id'] = educts_ids[0]
            self.particle_types[str(product_types[1])][0]['bond_educt_id'] = educts_ids[1]
            
            self.Reactions_Dict[current_reaction_id].add_path(self, reaction_type, rate, products_ids)            
            
            if interaction_type is not None:
                self.add_interaction(interaction_type, str(educt_types[0]), str(educt_types[1]), interaction_parameters, True, True)
            
        if  reaction_type == 'absorption':
            
            products_ids = np.empty(1, dtype = np.int64)
            products_ids[0] = self.particle_types[str(product_types[0])][0]['id']
            
            self.Reactions_Dict[current_reaction_id].add_path(self, reaction_type, rate, products_ids)
            
        if  reaction_type == 'enzymatic':
        
            products_ids = np.empty(2, dtype = np.int64)
            products_ids[0] = self.particle_types[str(product_types[0])][0]['id']
            products_ids[1] = self.particle_types[str(product_types[1])][0]['id']
            
            self.Reactions_Dict[current_reaction_id].add_path(self, reaction_type, rate, products_ids)            
            
            
        # print('Particle enzymatic reaction added ('+type1+'+'+type2+'->'+type3+'+'+type2+')')        
        
        
        
        
#%%

    def add_interaction(self, interaction_type, type1, type2, parameters, bond = False, breakable = False):
         
        """Assigns a particle-particle interaction potential to a particle type pair.
        
        Parameters
        ----------
        interaction_type : `str` {'harmonic_repulsion', 'PHS', 'harmonic_attraction', 'CSW'}
            Type of interaction potential
        type1 : `str`
            Names of the educt particle type 1
        type2 : `str`
            Names of the educt particle type 2
        parameters : `dict`
            Parameters for the interaction potential function.
        bond : `bool`
            True if interaction is initialized as part of a bindign reaction. (Default: False)
        breakable : `bool`
            If set to True, bonds that have been formed by a binding reaction are deleted if the interparticle distance is above the cutoff radius of the interaction potenial.
        
        Notes
        -----
        Bi-particle reactions are evaluated using the Doi scheme. Thereby, two educt particles react with a reaction rate k if the inter-particle distance is below the reaction radius.
        
        """
        
        type1_id = self.particle_types[type1][0]['id']
        type2_id = self.particle_types[type2][0]['id']
        
        stokes_radius_1 = self.particle_types[type1][0]['radius']
        stokes_radius_2 = self.particle_types[type2][0]['radius']
        
        interaction_id = self.interaction_IDs[str(interaction_type)]
        self.interaction_args[type1_id][type2_id]['global'] = not bond
        self.interaction_args[type2_id][type1_id]['global'] = not bond 
        self.interaction_args[type1_id][type2_id]['id'] = interaction_id
        self.interaction_args[type2_id][type1_id]['id'] = interaction_id    
        self.interaction_args[type1_id][type2_id]['type'] = interaction_type
        self.interaction_args[type2_id][type1_id]['type'] = interaction_type
        self.interaction_args[type1_id][type2_id]['breakable'] = breakable
        self.interaction_args[type2_id][type1_id]['breakable'] = breakable
        
        if interaction_type == 'harmonic_repulsion':
            
            if 'radius_1' in parameters and 'radius_2' in parameters:
                radius_1 = parameters['radius_1']
                radius_2 = parameters['radius_2']
            else:
                radius_1 = stokes_radius_1 
                radius_2 = stokes_radius_2
                
            self.interaction_args[type1_id][type2_id]['parameters'][0] = radius_1 + radius_2
            self.interaction_args[type2_id][type1_id]['parameters'][0] = radius_1 + radius_2
            self.interaction_args[type1_id][type2_id]['parameters'][1] = parameters['k']
            self.interaction_args[type2_id][type1_id]['parameters'][1] = parameters['k']
            
            self.interaction_args[type1_id][type2_id]['cutoff'] = radius_1 + radius_2
            self.interaction_args[type2_id][type1_id]['cutoff'] = radius_1 + radius_2
          
            if radius_1 > self.particle_types[type1][0]['cutoff']:
                self.particle_types[type1][0]['cutoff'] = radius_1
            
            if radius_2 > self.particle_types[type2][0]['cutoff']:
                self.particle_types[type2][0]['cutoff'] = radius_2
            
        
        elif interaction_type == 'PHS':

            if 'radius_1' in parameters and 'radius_2' in parameters:
                radius_1 = parameters['radius_1']
                radius_2 = parameters['radius_2']
            else:
                radius_1 = stokes_radius_1 
                radius_2 = stokes_radius_2
                
            self.interaction_args[type1_id][type2_id]['parameters'][0] = radius_1 + radius_2
            self.interaction_args[type2_id][type1_id]['parameters'][0] = radius_1 + radius_2
            self.interaction_args[type1_id][type2_id]['parameters'][1] = parameters['EpsR']
            self.interaction_args[type2_id][type1_id]['parameters'][1] = parameters['EpsR']
            self.interaction_args[type1_id][type2_id]['parameters'][2] = parameters['lr']
            self.interaction_args[type2_id][type1_id]['parameters'][2] = parameters['lr']
            self.interaction_args[type1_id][type2_id]['parameters'][3] = parameters['la']
            self.interaction_args[type2_id][type1_id]['parameters'][3] = parameters['la']
            
            self.interaction_args[type1_id][type2_id]['cutoff'] = radius_1 + radius_2
            self.interaction_args[type2_id][type1_id]['cutoff'] = radius_1 + radius_2

            if radius_1 > self.particle_types[type1][0]['cutoff']:
                self.particle_types[type1][0]['cutoff'] = radius_1
            
            if radius_2 > self.particle_types[type2][0]['cutoff']:
                self.particle_types[type2][0]['cutoff'] = radius_2
                
        elif interaction_type == 'harmonic_attraction':
            
            if 'radius_1' in parameters and 'radius_2' in parameters:
                radius_1 = parameters['radius_1']
                radius_2 = parameters['radius_2']
            else:
                radius_1 = stokes_radius_1 
                radius_2 = stokes_radius_2   

            self.interaction_args[type1_id][type2_id]['parameters'][0] = radius_1 + radius_2 + parameters['rc']
            self.interaction_args[type2_id][type1_id]['parameters'][0] = radius_1 + radius_2 + parameters['rc']
            self.interaction_args[type1_id][type2_id]['parameters'][1] = parameters['k']
            self.interaction_args[type2_id][type1_id]['parameters'][1] = parameters['k']
            self.interaction_args[type1_id][type2_id]['parameters'][2] = parameters['h']
            self.interaction_args[type2_id][type1_id]['parameters'][2] = parameters['h']
            self.interaction_args[type1_id][type2_id]['parameters'][3] = radius_1 + radius_2
            self.interaction_args[type2_id][type1_id]['parameters'][3] = radius_1 + radius_2
            
            self.interaction_args[type1_id][type2_id]['cutoff'] = radius_1 + radius_2 + parameters['rc']
            self.interaction_args[type2_id][type1_id]['cutoff'] = radius_1 + radius_2 + parameters['rc']
            
            if radius_1+parameters['rc']/2 > self.particle_types[type1][0]['cutoff']:
                self.particle_types[type1][0]['cutoff'] = radius_1+parameters['rc']/2 
            if radius_2+parameters['rc']/2 > self.particle_types[type2][0]['cutoff']:
                self.particle_types[type2][0]['cutoff'] = radius_2+parameters['rc']/2 
                
                
        elif interaction_type == 'CSW':
            
            self.interaction_args[type1_id][type2_id]['parameters'][0] = parameters['rw']
            self.interaction_args[type2_id][type1_id]['parameters'][0] = parameters['rw']
            self.interaction_args[type1_id][type2_id]['parameters'][1] = parameters['eps_csw']
            self.interaction_args[type2_id][type1_id]['parameters'][1] = parameters['eps_csw']
            self.interaction_args[type1_id][type2_id]['parameters'][2] = parameters['alpha']
            self.interaction_args[type2_id][type1_id]['parameters'][2] = parameters['alpha']
            
            self.interaction_args[type1_id][type2_id]['cutoff'] = 1.75*parameters['rw']
            self.interaction_args[type2_id][type1_id]['cutoff'] = 1.75*parameters['rw']
            
            if 1.75*parameters['rw']/2 > self.particle_types[type1][0]['cutoff']:
                self.particle_types[type1][0]['cutoff'] = 1.75*parameters['rw']/2 
            if 1.75*parameters['rw']/2 > self.particle_types[type2][0]['cutoff']:
                self.particle_types[type2][0]['cutoff'] = 1.75*parameters['rw']/2 
                
            
            
        self.interaction_defined[type2_id][type1_id] = True
        self.interaction_defined[type1_id][type2_id] = True
        self.pair_interaction[type1_id] = True
        self.pair_interaction[type2_id] = True
        
        self.interactions = True
        
        if bond == False:
            print('Global '+interaction_type+' added ('+type1+'-'+type2+')')  
        else:
            print(interaction_type+' bond added ('+type1+'-'+type2+')')


#%%


    def add_bm_reaction(self, reaction_type, educt_types, product_types, particle_pairs, pair_rates, pair_Radii):

        """Registeres a Bi-Molecular (BM) reaction for an educt molecule.
        
        Parameters
        ----------
        reaction_type : `str` {'fusion', 'enzymatic_mol'}
            Name of the reaction type
        educt_type : `array_like`
            Names of the educt molecule types
        particle_pairs : `array_like`
            List of educt particle pairs belonging to each of the two educt molecules
        pair_rates : `array_like`
            List of reaction rates for each particle pair
        pair_Radii : `array_like`
            List of reaction radii for each particle pair
        
        Notes
        -----
        Bi-molecular reactions are evaluated using the Doi scheme. Thereby, two educt particles react with a reaction rate k if the inter-particle distance is below the reaction radius. PyRID only evaluates the eucludean distances between particles but not molecule centers. Thereby, also for bi-molecular reactions educt particles need to be set. Since molecules can consist of several particles, a list of particle pairs can be passed and for each particle pair a reaction rate and radius is defined. If one of the pair particle reactions is successfull, the corresponding molecule that the particle belongs to is converted as defined by the reaction type. If the moelcule cosnsists of a single particle, the reaction will be handled just as the bi-particle reactions. The user needs to take care of calculating accurate reaction radii and rates such that the desired reaction dynamics are simulated, which can become difficult for large molecules. Therefore it is sometimes more convenient to place an interaction free particle at the origin of the molecule and execute the bimolecular reaction via this particle. This method is equivalent to the case where one would evaluate reactions based on the distance between molecule centers.
        As for the uni-moelcular reactions, several reaction paths can be defined. Each reaction path may involve different particle pairs of the educt molecules. On a side note: Thereby, orientation dependent bimolecular reactions are in principle possibile. However, since the reaction probability does not scale with the distance, directionality is only really accounted for if the reaction radius is small compared to the molecule size.
        
        """
        
        for educt_particle_types in particle_pairs:
            if not string_in_array(educt_particle_types[0], self.molecule_types[str(educt_types[0])].types):
                print('Particle type '+educt_particle_types[0]+' ist not part of molecule '+str(educt_types[0])+' topology!')
                raise ValueError('Particle type ist not part of molecule topology!')
                
            if not string_in_array(educt_particle_types[1], self.molecule_types[str(educt_types[1])].types):
                print('Particle type '+educt_particle_types[1]+' ist not part of molecule '+str(educt_types[1])+' topology!')
                raise ValueError('Particle type ist not part of molecule topology!')
                
            for mol_type in self.molecule_types:
                if not string_in_array(mol_type, educt_types):
                    if string_in_array(educt_particle_types[0], self.molecule_types[mol_type].types) or string_in_array(educt_particle_types[1], self.molecule_types[mol_type].types):
                        raise ValueError('Educt particle type found in multiple molecule topologies! Cannot unambiguously assign reaction educts for bimolecular reaction!')

        educt_molecule_ids = np.empty(2, dtype = np.int64)
        educt_molecule_ids[0] = self.molecule_types[str(educt_types[0])].type_id
        educt_molecule_ids[1] = self.molecule_types[str(educt_types[1])].type_id
        
        for educt_particle_types, rate, radius in zip(particle_pairs, pair_rates, pair_Radii):
            
            educts_ids = np.empty(2, dtype = np.int64)
            for i,educt in enumerate(educt_particle_types):
                educt_id = self.particle_types[str(educt)][0]['id']
                educts_ids[i] = educt_id
    
    
            if self.reaction_args[educts_ids[0],educts_ids[1]]['id'] == -1:
                
                if educts_ids[0]!=educts_ids[1]:
                    for i in range(2):
                        n_reactions = self.bp_reaction_ids[educts_ids[i]]['n_reactions']
                        self.bp_reaction_ids[educts_ids[i]]['ids'][n_reactions] = self.reaction_id
                        self.bp_reaction_ids[educts_ids[i]]['n_reactions'] += 1
                else:
                    n_reactions = self.bp_reaction_ids[educts_ids[0]]['n_reactions']
                    self.bp_reaction_ids[educts_ids[0]]['ids'][n_reactions] = self.reaction_id
                    self.bp_reaction_ids[educts_ids[0]]['n_reactions'] += 1                
                
                self.reaction_args[educts_ids[0],educts_ids[1]]['id'] = self.reaction_id
                self.reaction_args[educts_ids[1],educts_ids[0]]['id'] = self.reaction_id
                self.reaction_args[educts_ids[0],educts_ids[1]]['type_BP_BM'] = 'BM'
                self.reaction_args[educts_ids[1],educts_ids[0]]['type_BP_BM'] = 'BM'
                # update the reaction_args matrix, which is used to lookup all the necessary properties for each reaction
                # during reaction handling (see update_util):
                self.reaction_args[educts_ids[0],educts_ids[1]]['defined'] = True
                self.reaction_args[educts_ids[1],educts_ids[0]]['defined'] = True
                self.reaction_args[educts_ids[0],educts_ids[1]]['radius'] = radius
                self.reaction_args[educts_ids[1],educts_ids[0]]['radius'] = radius
                # Which of the two educts is the enzyme?:
                self.reaction_args[educts_ids[0],educts_ids[1]]['enzyme'] = educts_ids[1] 
                self.reaction_args[educts_ids[1],educts_ids[0]]['enzyme'] = educts_ids[1]
                
                self.interaction_defined[educts_ids[0],educts_ids[1]] = True
                self.interaction_defined[educts_ids[1],educts_ids[0]] = True
                self.pair_interaction[educts_ids[0]] = True
                self.pair_interaction[educts_ids[1]] = True
                
                self.Reactions_Dict[self.reaction_id] = rru.Reaction(educt_molecule_ids, 'bimolecular', 'Molecule', educts_ids, radius)
                current_reaction_id = self.reaction_args[educts_ids[0],educts_ids[1]]['id']
    
                self.reaction_id += 1
                
            else:
                if self.reaction_args[educts_ids[0],educts_ids[1]]['type_BP_BM'] != 'BM':
                    raise ValueError('Unable to add the Bimolecular reaction! A Biparticle reaction has already been defined betwen these two particle types!')
    
                current_reaction_id = self.reaction_args[educts_ids[0],educts_ids[1]]['id']
    
    
    
            if  reaction_type == 'fusion':
                
                products_ids = np.empty(1, dtype = np.int64)
                products_ids[0] = self.molecule_types[str(product_types[0])].type_id
                
                self.Reactions_Dict[current_reaction_id].add_path(self, reaction_type, rate, products_ids)
                
            if  reaction_type == 'enzymatic_mol':
            
                products_ids = np.empty(2, dtype = np.int64)
                products_ids[0] = self.molecule_types[str(product_types[0])].type_id
                products_ids[1] = self.molecule_types[str(product_types[1])].type_id
                
                self.Reactions_Dict[current_reaction_id].add_path(self, reaction_type, rate, products_ids)            
                
            
            self.interactions = True
            
            
            if radius/2 > self.particle_types[str(educt_particle_types[0])][0]['cutoff']:
                self.particle_types[str(educt_particle_types[0])][0]['cutoff'] = radius/2
    
            if radius/2 > self.particle_types[str(educt_particle_types[1])][0]['cutoff']:
                self.particle_types[str(educt_particle_types[1])][0]['cutoff'] = radius/2
            
        # print('Particle enzymatic reaction added ('+type1+'+'+type2+'->'+type3+'+'+type2+')')           

        

