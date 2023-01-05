# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numba as nb
import numpy as np
from scipy.linalg import sqrtm
import os
import warnings
import h5py
from pathlib import Path

from .system import update_pos as upos
from .system import update_force as uf
from .reactions import update_reactions as uru
from .observables import write_util as wru
from .data_structures import h_grid_util as hu
from .observables import checkpoint_util as cp
from .system.system_util import System
from .molecules.particles_util import Particles
from .molecules.rigidbody_util import RBs
from .system import distribute_vol_util as distribute_vol
from .system import distribute_surface_util as distribute_surf
from .observables.checkpoint_util import load_checkpoint
from .observables.observables_util import Observables
from .math import transform_util as trf
from .evaluation.rdf_util import create_rb_hgrid, update_rb_hgrid
from .reactions import reactions_util as ru

import time


def convert_dict(untyped_dict):
    
    key_type = nb.typeof(list(untyped_dict.keys())[0])
    value_type = nb.typeof(list(untyped_dict.values())[0])
    
    typed_dict = nb.typed.Dict.empty(key_type, value_type)
    
    for key in untyped_dict:
        typed_dict[key] = untyped_dict[key]
    
    return typed_dict


@nb.njit#(cache=True)
def random_quaternion():
    
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
    
    # Based on K. Shoemake. Uniform random rotations. In D. Kirk, editor, Graphics Gems III, pages 124-132. Academic, New York, 1992. http://planning.cs.uiuc.edu/node198.html
    
    u1,u2,u3 = np.random.rand(3)
    
    return np.array([np.sqrt(1-u1)*np.sin(2*np.pi*u2), np.sqrt(1-u1)*np.cos(2*np.pi*u2), np.sqrt(u1)*np.sin(2*np.pi*u3), np.sqrt(u1)*np.cos(2*np.pi*u3)])


def update_pressure(Simulation):

    """Calculates the pressure from the virial tensor.
    
    Parameters
    ----------
    Simulation : `ob`
        Simulation instance
    
    Notes
    -----
    When doing rigid body molecular dynamics or Brownian dynamics simulations, we need to be careful how to calculate the virial tensor. Taking the interparticle distances will result in the wrong pressure. Instead, one needs to use the pairwise distance between the centers of mass of the rigid bodies :cite:p:`Glaser2020` :
        
    .. math::
        :label: Pressure
        
        P_{mol} = P_{mol}^{kin} + \\frac{1}{6 V} \\sum_{i=1}^{N} \\sum_{j \\neq}^{N} \\langle \\boldsymbol{F}_{ij} \cdot (\\boldsymbol{R}_i - \\boldsymbol{R}_j) \\rangle,
        
    where :math:`V` is the total volume of the simulation box, :math:`\\boldsymbol{F}_{ij}` is the force on particle i exerted by particle j and :math:`\\boldsymbol{R}_{i}, \\boldsymbol{R}_{j}` are the center of mass of the rigid body molecules, not the center of particles i and j! In Brownian dynamics simulations, :math:`P_{mol}^{kin} = N_{mol} k_B T`, where :math:`N_{mol}` is the number of molecules. Also, the origin of molecules is represented by the center of diffusion around which the molecule rotates about, which is not the center of mass :cite:p:`Harvey1980`! The net frictional force and troque act thorugh the center of diffusion. This is because when doing Brownian dynamics (and equaly for Langevin dynamics), we do account for the surrounding fluid. Different parts of the molecule will therefore interact with each other via hydrodynamic interactions/coupling. As a result, the center of the molecule (around which the molecule rotates in response to external forces) is not the same as the center of mass, which assumes no such interactions (the molecule sites in empry space). However, for symmetric molecules, the center of mass and the center of diffusion are the same!
    
    
    Returns
    -------
    tuple(float, float64[3,3])
        Total pressure and pressure tensor.
    
    """
    
    for mol_name in Simulation.System.molecule_types:
        mol_id = Simulation.System.molecule_types[mol_name].type_id
        Simulation.System.Pressure[mol_id] = (Simulation.System.Nmol[mol_id]*Simulation.System.kbt+(Simulation.System.virial_scalar[mol_id]+Simulation.System.virial_scalar_Wall[mol_id])/3)/Simulation.System.volume
        
    Pressure_total = np.sum(Simulation.System.Pressure)
    Pressure_Tensor = (Simulation.System.N*Simulation.System.kbt + np.sum(Simulation.System.virial_tensor, axis=0))/Simulation.System.volume
                
    return Pressure_total, Pressure_Tensor

#%%

class Simulation(object):
    
    """
    A brief summary of the classes purpose and behavior
    
    Attributes
    ----------
    attribute_1 : dtype
        Some Information
    attribute_2 : dtype
        Some Information
    
    Methods
    -------
    method_1(arguments)
        Some general information about this method
    
    """
    
    def __init__(self, box_lengths = np.array([50.0,50.0,50.0]), dt = 0.1, Temp = 293.15, eta = 1e-21, stride = 100, write_trajectory = True, file_path = 'Files/', file_name = 'PyRID', fig_path = 'Figures/', boundary_condition = 'periodic', nsteps = 10000, sim_time = None, wall_force = 100.0, seed = None, length_unit = 'nanometer', time_unit = 'ns', cells_per_dim = None, max_cells_per_dim = 50) :
                      
        # print('Initialized simulation.')
        
        np.set_printoptions(precision=np.finfo(np.float64).precision*2)
        
        box_lengths = np.array(box_lengths)
        
        self.System = System(box_lengths = box_lengths, dt = dt, Temp = Temp, eta = eta, boundary_condition = boundary_condition, seed = seed, wall_force = wall_force, length_unit = length_unit, time_unit = time_unit, cells_per_dim = cells_per_dim, max_cells_per_dim = max_cells_per_dim)#'repulsive',  wall_force = 1e7
        
        self.Particles = Particles() # particle_array()
        
        self.RBs = RBs()
        
        # self.Observables = {}
        
        self.file_path = Path(file_path)
        self.file_name = file_name
        self.fig_path = Path(fig_path)
        
        try:
            os.makedirs(self.file_path) 
        except FileExistsError:
            # directory already exists
            pass
        
        try:
            os.makedirs(self.fig_path) 
        except FileExistsError:
            # directory already exists
            pass
        
        self.write_trajectory = write_trajectory
        self.stride = stride
        self.current_step = 0
        
        self.release_events = []
        
        if nsteps is not None:
            self.nsteps = nsteps
        else:
            self.nsteps = 1e10
        if sim_time is not None:
            self.sim_time = sim_time
        else:
            self.sim_time = 1e10
            
        self.Observer = None
        
        self.Measures = ['Position', 'Energy', 'Pressure', 'Volume', 'Virial', 'Virial Tensor', 'Number', 'Orientation', 'Bonds', 'Reactions', 'Force', 'Torque', 'RDF']
        
        self.length_units = {'micrometer':'\mu m', 'nanometer':'nm'}
        
        self.units = dict()
        self.units['Time'] = self.System.time_unit
        self.units['Length'] = self.length_units[self.System.length_unit]
        self.units['Position'] = self.length_units[self.System.length_unit]
        self.units['Energy'] = r'$\frac{kJ}{mol}$'
        self.units['Volume'] = r'${}^3$'.format(self.length_units[self.System.length_unit])
        self.units['Force'] = r'$\frac{{kJ}}{{mol \cdot {}}}$'.format(self.length_units[self.System.length_unit])
        self.units['Torque'] = r'$\frac{kJ}{mol}$'
        self.units['Virial'] = r'$\frac{kJ}{mol}$'
        self.units['Virial Tensor'] = r'$\frac{kJ}{mol}$'
        self.units['Pressure'] = r'$\frac{{kJ}}{{mol \cdot {}^3}}$'.format(self.length_units[self.System.length_unit])
        self.units['Reactions'] = r'#'
        self.units['Bonds'] = r'#'
        self.units['Number'] = r'#'
        self.units['Orientation'] = r'°'

            
        self.time_unit = time_unit
        
        item_t_checkpoint = np.dtype([('directory', 'U40'),('max_saves', np.int32),('stride', np.int32)],  align=True)   
        self.checkpoint = np.zeros(1, dtype = item_t_checkpoint)
        
        self.checkpoint[0]['stride'] = 0
        self.checkpoint[0]['directory'] = ''
        self.checkpoint[0]['max_saves'] = 0
        
        
    def print_timer(self):
        
        t_total = 0.0
        for key in self.Timer:
            if key != 'pu/s':
                print(key+': ', (self.current_step-1)/self.Timer[key],' it/s | ', (self.Timer[key]/(self.current_step-1))*1e3, ' ms/it' )
                t_total += self.Timer[key]/(self.current_step-1)
            else:
                print('total pu/s: ', self.Timer[key]/(self.current_step-1),' pu/s')
            
    def k_micro(self, mol_type_1, mol_type_2, k_macro, R_react, loc_type = 'Volume'):
        
        k = ru.k_micro(self.System, mol_type_1, mol_type_2, k_macro, R_react, loc_type = loc_type)
            
        return k
    
    def k_macro(self, D1, D2, k_micro, R_react):
        
        k = ru.k_macro(D1, D2, k_micro, R_react)
            
        return k
    
    def add_checkpoints(self, stride, directory, max_saves):

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
    
        item_t_checkpoint = np.dtype([('directory', 'U40'),('max_saves', np.int32),('stride', np.int32)],  align=True)   
        self.checkpoint = np.zeros(1, dtype = item_t_checkpoint)
        
        self.checkpoint[0]['stride'] = stride
        self.checkpoint[0]['directory'] = directory
        self.checkpoint[0]['max_saves'] = max_saves
        
        
    def add_barostat_berendsen(self, Tau_P, P0, start):
        
        self.System.add_barostat_berendsen(Tau_P, P0, start)
        
    def set_compartments(self, Compartments, triangles, vertices, mesh_scale = 1.0, adapt_box = False):
        
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
        
        if self.System.cells_per_dim.shape[0]!=3:
            if len(self.System.particle_types) > 0:
                self.System.add_system_grid()
            else:
                raise ValueError('Please register all particle types used in the simulation or manualy define a grid size for the simulation box before adding a mesh compartment! ')
            
        # Check if there are any transparent face groups:
        triangle_ids_transparent = []
        for comp_name in Compartments:
            if len(Compartments[comp_name]['face_groups']) > 0:
                for group in Compartments[comp_name]['face_groups']:
                    if group == 'transparent':
                        triangle_ids_transparent.extend(Compartments[comp_name]['face_groups'][group])
        
        if len(triangle_ids_transparent)>0:
            triangle_ids_transparent = np.array(triangle_ids_transparent)
        else:
            triangle_ids_transparent = None
            
        # --------------------
        # Check if a Box compartment is defined:
        if 'Box' in Compartments: 
            self.System.add_box_compartment()
            box_triangle_ids = Compartments['Box']['triangle_ids']
        else:
            box_triangle_ids = None
            
        # --------------------
        #Add mesh:
        self.System.add_mesh(vertices, triangles, mesh_scale, box_triangle_ids, triangle_ids_transparent, adapt_box)
        
        # -------------------
        
        for comp_name in Compartments:
            
            if comp_name != 'Box':
                self.System.set_compartments(comp_name, Compartments[comp_name]['triangle_ids'])
                
                comp_id = self.System.n_comps
            
                if len(Compartments[comp_name]['face_groups']) > 0:
                    for group in Compartments[comp_name]['face_groups']:
                        if group == 'border':
                            triangle_ids_border = Compartments[comp_name]['face_groups'][group]
                            self.System.Compartments[comp_id].add_border_2d(triangle_ids_border, self.System)
                            
                            if 'Box' not in Compartments:
                                raise KeyError('Compartment border given, but the Box Compartment is missing!')
                            if comp_name not in Compartments['Box']['face_groups']:
                                raise KeyError('Compartment-Box intersection face group is missing!')
                            
                            triangle_ids_border_box = Compartments['Box']['face_groups'][comp_name]
                            self.System.Compartments[comp_id].add_border_3d(triangle_ids_border_box, self.System)
                                
                        else:
                            triangle_ids_group = Compartments[comp_name]['face_groups'][group]
                            self.System.Compartments[comp_id].add_group(group, triangle_ids_group, self.System)
                        
            elif comp_name == 'Box' and len(Compartments[comp_name]['face_groups']) > 0: 
                
                triangle_ids_box = Compartments['Box']['face_groups']['Box']
                self.System.add_border_3d(triangle_ids_box)
                
                
        self.System.create_cell_list()
                
        if adapt_box:
            print('Simulation box has been adapted. box_lengths = ', self.System.box_lengths)
        
    def register_particle_type(self, Type, radius):
        self.System.register_particle_type(Type, radius)
        
    #%%
    
    def add_interaction(self, interaction_type, type1, type2, parameters, bond = False):
        
        parameters = convert_dict(parameters)
        
        self.System.add_interaction(interaction_type, type1, type2, parameters, bond)
        

#%%
    
    def add_up_reaction(self, reaction_type, educt_type, rate, product_types = None, product_loc = None, product_direction = None, radius = None):
            
        if reaction_type == 'release':
            product_loc = np.array([0,product_loc])
            product_direction = np.array([0,product_direction])
            
        if product_types is not None:
            product_types = np.array(product_types)
        if product_loc is not None:
            product_loc = np.array(product_loc)
        if product_direction is not None:
            product_direction = np.array(product_direction)
            
        self.System.add_up_reaction(reaction_type, educt_type, rate, product_types, product_loc, product_direction, radius)
            
    #%%
    
    def add_um_reaction(self, reaction_type, educt_type, rate, product_types = None, product_loc = None, product_direction = None, radius = None):
        
        if product_types is not None:
            product_types = np.array(product_types)
        if product_loc is not None:
            product_loc = np.array(product_loc)
        if product_direction is not None:
            product_direction = np.array(product_direction)
            
        self.System.add_um_reaction(reaction_type, educt_type, rate, product_types, product_loc, product_direction, radius)
            
    #%%
    
    def add_bp_reaction(self, reaction_type, educt_types, product_types, rate, radius, interaction_type = None, interaction_parameters = None):
        
        educt_1_radius = self.System.particle_types[educt_types[0]][0]['radius']
        educt_2_radius = self.System.particle_types[educt_types[1]][0]['radius']
        
        if radius < educt_1_radius+educt_2_radius:
                warnings.warn('Warning: The reaction radius for the particle pair {0}, {1} is smaller than the sum of their radii. The reaction radius should not be smaller than {2:.3g}'.format(educt_types[0], educt_types[1], educt_1_radius+educt_2_radius))
        
        if interaction_parameters is not None:
            interaction_parameters = convert_dict(interaction_parameters)
        
        self.System.add_bp_reaction(reaction_type, np.array(educt_types), np.array(product_types), rate, radius, interaction_type, interaction_parameters)
        
    #%%
    
    def add_bm_reaction(self, reaction_type, educt_types, product_types, particle_pairs, pair_rates, pair_radii, placement_factor = 0.5):
        
        for i,particle_types in enumerate(particle_pairs):
            educt_1_radius = self.System.particle_types[particle_types[0]][0]['radius']
            educt_2_radius = self.System.particle_types[particle_types[1]][0]['radius']
            
            if pair_radii[i] < educt_1_radius+educt_2_radius:
                warnings.warn('Warning: The reaction radius for the particle pair {0}, {1} is smaller than the sum of their radii. The reaction radius should not be smaller than {2:.3g}'.format(particle_types[0], particle_types[1], educt_1_radius+educt_2_radius))
            
        self.System.add_bm_reaction(reaction_type, np.array(educt_types), np.array(product_types), np.array(particle_pairs), np.array(pair_rates), np.array(pair_radii), placement_factor)
            
    #%%
    
    def register_molecule_type(self, molecule_name, particle_pos, particle_types, collision_type = 0, h_membrane = None):
        
        particle_pos = np.array(particle_pos)
        particle_types = np.array(particle_types, dtype = np.dtype('U20'))
        
        self.System.register_molecule_type(molecule_name, particle_pos, particle_types, collision_type, h_membrane)
        
    def set_diffusion_tensor(self, molecule_name, D_tt, D_rr):
        
        D_tt = np.array(D_tt)
        D_rr = np.array(D_rr)
        
        mu_tb = np.ascontiguousarray(D_tt/(self.System.kbt))
        mu_rb = np.ascontiguousarray(D_rr/(self.System.kbt))
        
        mu_tb_sqrt = sqrtm(mu_tb)
        trf.valid_mobility_tensor_test(mu_tb, mu_tb_sqrt)
        mu_rb_sqrt = sqrtm(mu_rb)
        trf.valid_mobility_tensor_test(mu_rb, mu_rb_sqrt)
        
        # If mu passed the valid mobility tensor test there might still exist a small imaginary part due to numerical errors occuring in sqrtm():
        mu_rb_sqrt = np.ascontiguousarray(mu_rb_sqrt.real)
        mu_tb_sqrt = np.ascontiguousarray(mu_tb_sqrt.real)
        
        self.System.set_diffusion_tensor(molecule_name, D_tt, D_rr, mu_tb, mu_rb, mu_tb_sqrt, mu_rb_sqrt)
        
    def fixed_concentration_at_boundary(self, molecule_name, C, comp_name, location):
        
        if location == 'Volume':
            self.System.fixed_concentration_at_boundary(molecule_name, C, comp_name, 0)
        elif location == 'Surface':
            self.System.fixed_concentration_at_boundary(molecule_name, C, comp_name, 1)
        else:
            raise NameError('Location must be either Volume or Surface!')            
        
    def add_molecules(self, Location, Compartment_Number, points, quaternion, points_type, face_ids = None):
        
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
        
        if Compartment_Number>0 and Compartment_Number not in self.System.Compartments:
            raise KeyError('No compartment with id {} found!'.format(Compartment_Number))
        
        Types = np.array(list(self.System.molecule_types.keys()))
        
        if Location == 'Volume':
            
            if Compartment_Number>0:
                comp_id = self.System.Compartments[Compartment_Number].id
            elif Compartment_Number==0:
                comp_id = self.System.id
            else:
                raise IndexError('Compartment number must be a positive interger!')
                
            for i in range(len(points)):
                
                self.RBs.add_RB(str(Types[points_type[i]]), comp_id, self.System, self.Particles, 0)
                
                self.RBs.set_pos(self.Particles, self.System, points[i], self.RBs.slot-1) 
                
                self.RBs.set_orientation_quat(self.Particles, quaternion[i], self.System, self.RBs.slot-1)
                
        elif Location == 'Surface':
            
            if Compartment_Number>0:
                comp_id = self.System.Compartments[Compartment_Number].id
            else:
                raise IndexError('Compartment number must be greater zero!')
                
            if face_ids is None:
                raise TypeError('Missing required argument face_ids')
                
            # a_n = np.zeros(3)
            for i in range(len(points)):
                
                self.RBs.add_RB(str(Types[points_type[i]]), comp_id, self.System, self.Particles, 1)
        
                self.RBs.set_triangle(face_ids[i], self.RBs.slot-1)
        
                self.RBs.set_pos(self.Particles, self.System, points[i], self.RBs.slot-1) 
                
                self.RBs.set_orientation_quat(self.Particles, quaternion[i], self.System, self.RBs.slot-1)
        
        else:
            raise NameError('Location must be either Volume or Surface!')            

            
    def distribute(self, Method, Location, Compartment_Number, Types, Number, clustering_factor=1, max_trials=100, jitter = None, triangle_id = None, multiplier = 100, facegroup = None):

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
        
        if Compartment_Number>0 and Compartment_Number not in self.System.Compartments:
            raise KeyError('No compartment with id {} found!'.format(Compartment_Number))
    
        Number = np.array(Number)
        Types = np.array(Types)
        
        if Method not in ['PDS', 'PDS uniform', 'Gauss', 'MC']:
            raise NameError('Method {} does not exist!'.format(Method))
        
        if Method == 'MC':
            if Location == 'Volume':
                if Compartment_Number>0:
                    points, points_types, quaternion = distribute_vol.mc(self.System, self.System.Compartments[Compartment_Number], Types, Number)
                    
                    return points, points_types, quaternion
                
                elif Compartment_Number == 0:
                    points, points_types, quaternion = distribute_vol.mc(self.System, self.System, Types, Number)
                    
                    return points, points_types, quaternion
                
                else:
                    raise IndexError('Compartment number must be a positive interger!')
                    
            elif Location == 'Surface':
                
                points, points_types, quaternion, face_ids = distribute_surf.mc(self.System, self.System.Compartments[Compartment_Number], Types, Number, facegroup)
                
                return points, points_types, quaternion, face_ids
                
            else:
                raise NameError('Location must be either Volume or Surface!')
                
        if Method == 'PDS uniform':
            if Location == 'Volume':
                if Compartment_Number>0:
                    points, points_types, quaternion = distribute_vol.pds_uniform(self.System.Compartments[Compartment_Number], self.System, Types, Number, multiplier = multiplier)
                    return points, points_types, quaternion
                
                elif Compartment_Number == 0:
                    points, points_types, quaternion = distribute_vol.pds_uniform(self.System, self.System, Types, Number, multiplier = multiplier)
                    
                    return points, points_types, quaternion
                
                else:
                    raise IndexError('Compartment number must be a positive interger!')
                    
            elif Location == 'Surface':
                print('Method "PDS uniform" is only available for distribution of points in a Compartment Volume. For Surface distribution use Method "PDS" instead!')
            else:
                raise NameError('Location must be either Volume or Surface!')
                    
                    
        if Method == 'PDS':
            if Location == 'Volume':
                if Compartment_Number>0:
                    points, points_types, quaternion = distribute_vol.pds(self.System.Compartments[Compartment_Number], self.System, Types, Number, clustering_factor, max_trials)
                    
                    return points, points_types, quaternion
                
                elif Compartment_Number == 0:
                    points, points_types, quaternion = distribute_vol.pds(self.System, self.System, Types, Number, clustering_factor, max_trials)
                    
                    return points, points_types, quaternion
                
                else:
                    raise IndexError('Compartment number must be a positive interger!')
            
            elif Location == 'Surface':
                if Compartment_Number>0:
                    points, points_types, quaternion, face_ids = distribute_surf.pds(self.System.Compartments[Compartment_Number], self.System, Types, Number, facegroup)
                    
                    return points, points_types, quaternion, face_ids
                
                else:
                    raise IndexError('Compartment number must be greater than zero!')
                    
            else:
                raise NameError('Location must be either Volume or Surface!')
                
        if Method == 'Gauss':
            points, quaternion, points_types, face_ids = distribute_surf.normal(triangle_id, jitter, Number, Types, self.System)
            
            return points, points_types, quaternion, face_ids
        
    #%%
    
    def observe(self, Measure, molecules = 'All', reactions = 'All', obs_stride = 1, bin_width = None, n_bins = None, stepwise = True, binned= False):
        
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
        
        
        if self.Observer is None:
            self.Observer = Observables(self)
            
        if Measure not in self.Measures:
            raise NameError('Measure {} not known!'.format(Measure))
          
            
        if Measure in ['Force', 'Troque', 'Orientation', 'Position'] and binned == True:
            binned = False
            stepwise = True
            warnings.warn('Warning: Binning is not supported for property {0}. PyRID will instead sample {0} stepwise!'.format(Measure))
            
        if molecules == 'All':
            molecules = []
            for mol in self.System.molecule_types:
                molecules.append(mol)
                
        if reactions == 'All':
            reactions = []
            for reactions_id in self.System.Reactions_Dict:
                reactions.append(reactions_id)
                
        molecules = np.array(molecules)
        
        self.Observer.observe(Measure, obs_stride, self, types = molecules, reactions = reactions, stepwise = stepwise, binned= binned)
            
        
        
           
    def observe_rdf(self, rdf_pairs = None, rdf_bins = 100, rdf_cutoff = None, stride = None):
        
        if self.Observer is None:
            self.Observer = Observables(self)
            
        self.Observer.observe_rdf(rdf_pairs, rdf_bins, rdf_cutoff, stride, self)
        
        print('Observing RDF.')
        
        
    #%%
                
    def add_release_site(self, Location, timepoint, compartment_id, Number = None, Types = None, origin = None, jitter = None, Molecules_id = None, Positions = None, Quaternion = None, triangle_id = None):
        
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
        
        release_dict = {}
        release_dict['compartment_id'] = compartment_id
        release_dict['timepoint'] = timepoint
        release_dict['location'] = Location
        release_dict['triangle_id'] = triangle_id
        
        # Use position data passed by user:
        release_dict['Positions'] = np.array(Positions)
        release_dict['Quaternion'] = np.array(Quaternion)
        release_dict['Molecules_id'] = Molecules_id
        
        # Release Molecules based on some Method:
        release_dict['Number'] = np.array(Number)
        release_dict['Types'] = np.array(Types)
        release_dict['jitter'] = jitter
        release_dict['origin'] = np.array(origin)
        
        self.release_events.append(release_dict)
            
            
    def release_molecules(self, release_event):
        
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
        
        Types = np.array(list(self.System.molecule_types.keys()))
        
        if release_event['location'] == 'Volume':
            if release_event['jitter'] is not None:
                
                Positions, Quaternion, Molecules_id = distribute_vol.normal(release_event['origin'], release_event['jitter'], release_event['Number'], release_event['Types'], self.System)
            
            else:
                Positions = release_event['Positions']
                Quaternion = release_event['Quaternion']
                Molecules_id = release_event['Molecules_id']
                
            if len(Positions)>0:
                distribute_vol.add_molecules(release_event['compartment_id'], self.System, self.RBs, self.Particles, Positions, Quaternion, Molecules_id, Types)
        
        if release_event['location'] == 'Surface':
        
            Positions, Quaternion, Molecules_id, Triangle_ids = distribute_surf.normal(release_event['triangle_id'], release_event['jitter'], release_event['Number'], release_event['Types'], self.System)
            
            if len(Positions)>0:
                distribute_surf.add_molecules(release_event['compartment_id'], self.System, self.RBs, self.Particles, Positions, Quaternion, Triangle_ids, Molecules_id, Types)
                
    
#%%

    # ------------------
    
    def progress(self, sim_time, nsteps, j, time_passed, it_per_s, Np, Pressure_total, N_bonds, Volume, out_linebreak, N, width=30):
        
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
        
        # sys.stdout.write("\033[F") #back to previous line 
        # sys.stdout.write("\033[K") #clear line 
        
        if out_linebreak==False:
            end_type = ''
        else:
            end_type = '\n'
            
            
        if sim_time<1e10:
            
            left = int(width * time_passed // sim_time)
            right = width - left
            percent = time_passed/ sim_time *100
            
            print('\r[', '#' * left, ' ' * right, ']',
                  ' {0:1.0f}% |'.format(percent)+'Time left: {0:1.1f} min.'.format((sim_time-time_passed)/60), sep='', end=end_type, flush=True)
        else:
            
            left = int(width * j // nsteps)
            right = width - left
            percent = j/nsteps *100
            
            print('\r[', '#' * left, ' ' * right, ']', ' {0:1.0f}% |'.format(percent)+'Steps left: {0:1.0f}'.format(nsteps-j), sep='', end=end_type, flush=True)
            
            
                
        # print('\r[', '#' * left, ' ' * right, ']',
        #       ' {0:.0f}% |'.format(percent)+progress_type+': {0:1.1f} min.'.format((sim_time-time_passed)/60),
        #       '| P: {0:.2E}'.format(Pressure_total), '| Bonds: {0:1.0f}'.format(np.sum(N_bonds)),
        #       '| V: {0:.2E}'.format(Volume), '| Time: {0:1.1f} min.'.format(time_passed/60), '| step: {0:1.0f}'.format(j), '| it/s: {0:1.1f}'.format(it_per_s), '| pu/s: {0:1.1f}'.format(Np*it_per_s), '| N: {0:d} |'.format(N),
        #       sep='', end=end_type, flush=True)
        
        
        for prop in self.progress_properties:
            
            if prop == 'it/s':
                
                print('| it/s: {0:1.1f}'.format(it_per_s), sep='', end=end_type, flush=True)
                
            if prop == 'pu/s':
                
                print('| pu/s: {0:1.1f}'.format(Np*it_per_s),sep='', end=end_type, flush=True)
            
            if prop == 'Time passed':
                
                print('| Time: {0:1.1f} min.'.format(time_passed/60), sep='', end=end_type, flush=True)
                
            if prop == 'step':
                
                print('| step: {0:1.0f}'.format(j), sep='', end=end_type, flush=True)
                
            if prop == 'N':
                
                print('| N: {0:d}'.format(N), sep='', end=end_type, flush=True)                
                
            if prop == 'Pressure':
                
                print('| P: {0:1.2E}'.format(Pressure_total), sep='', end=end_type, flush=True)
            
            if prop == 'Volume':
                
                print('| V: {0:1.2E}'.format(Volume), sep='', end=end_type, flush=True)
                
            if prop == 'Vol_Frac':
                
                Vfrac = 0
                for mol_name in self.System.molecule_types:
                    mol_id = self.System.molecule_types[mol_name].type_id
                    N = self.System.Nmol[mol_id]
                    Vfrac += N*self.System.molecule_types[mol_name].volume/self.System.volume
                    
                 
                print('| Vfrac: {0:1.2E}'.format(Vfrac), sep='', end=end_type, flush=True)
                
            if prop == 'Bonds':
                
                print('| Bonds: {0:1.0f}'.format(np.sum(N_bonds)), sep='', end=end_type, flush=True)
                

            
#%%

    def load_checkpoint(self, file_name, index, directory = 'checkpoints/'):
        
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
        
        file_extension = '.npz'
        if file_name[-4:] == '.npz':
            file_name = file_name[0:-4]
            
        self.RBs, self.Particles, self.HGrid = load_checkpoint(self.System, directory = directory, file = file_name + f'_{index}' + file_extension)

        self.System.box_lengths[0] *= 1
        self.System.volume = self.System.box_lengths.prod()
        
        
    #%%
    
    
    def run(self, progress_stride = 100, keep_time = False, out_linebreak = False, progress_bar_properties = None, start_reactions = 0):
        
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
        
        if progress_bar_properties is None:
            self.progress_properties = ['it/s', 'pu/s']
        elif progress_bar_properties == 'All':
            self.progress_properties = ['it/s', 'pu/s', 'Time passed', 'step', 'N', 'Pressure', 'Volume', 'Vol_Frac', 'Bonds']
        else:
            if not isinstance(progress_bar_properties, list):
                raise ValueError('progress_bar_properties must be a list of strings!')
            for prop in progress_bar_properties:
                if prop not in ['it/s', 'pu/s', 'Time passed', 'step', 'N', 'Pressure', 'Volume', 'Vol_Frac', 'Bonds']:
                    raise ValueError(str(prop)+'is not a supported property to be displayed in the progress bar!')
                    
            self.progress_properties = ['it/s', 'pu/s']+progress_bar_properties
        
        if self.Observer is not None:
            self.hdf = h5py.File(self.file_path / 'hdf5' / (self.file_name+'.h5'), 'a')
        
        print_system_count = False
        
        self.HGrid = hu.create_hgrid(self.Particles, self.System.particle_types, self.System.box_lengths, self.System.Np, self.System)
        
        # Check if the reaction rates are not too high for the chosen time step, otherwise print a warning:
        for reaction_type_index in self.System.Reactions_Dict:
            Reaction = self.System.Reactions_Dict[reaction_type_index]
            if 10*self.System.dt > 1/Reaction.rate:
                if Reaction.reaction_educt_type == 'Molecule':
                    educts = [self.System.molecule_id_to_name[educt_id] for educt_id in Reaction.educts]
                else:
                    educts = [self.System.particle_id_to_name[educt_id] for educt_id in Reaction.educts]
                warnings.warn('Warning: Reaction rate for the {0} reaction of educts {1} > 1/(10*dt). k*dt = {2:.3g}. Discretization errors may become too large!'.format(Reaction.reaction_educt_type ,educts, Reaction.rate*self.System.dt))
        
        #-----------------------------------------------------
        # Initialize Write Molecule Traces
        #-----------------------------------------------------

        wru.write_init(self, self.System)
        
        #---------------------
        
        
        if self.checkpoint[0]['max_saves']>0:
            try:
                os.makedirs(self.checkpoint[0]['directory']) 
            except FileExistsError:
                # directory already exists
                pass
                    
    
        lu_prefix = self.System.length_units_prefix[self.System.length_unit]
        Pressure_total = 0.0
        Pressure_Tensor = np.zeros((3,3), dtype=np.float64)
        checkpoint_counter = 0
        
        self.Timer = {}
        self.Timer['total'] = 0.0
        self.Timer['pu/s'] = 0.0
        self.Timer['force'] = 0.0
        self.Timer['integrate'] = 0.0
        self.Timer['reactions'] = 0.0
        self.Timer['observables'] = 0.0
        self.Timer['barostat'] = 0.0
        self.Timer['write'] = 0.0
        
        #%%
        # ------------------------------------------
        # Write to file
        # ------------------------------------------
        if self.checkpoint[0]['max_saves']>0:
            
            cp.save(self, self.System, self.RBs, self.Particles, self.HGrid, checkpoint_counter)
            checkpoint_counter += 1
            
            if checkpoint_counter>=self.checkpoint[0]['max_saves']:
                checkpoint_counter = 0
    
        wru.write(0, self, self.System, self.Particles)
        
        #%%
        
        # # ------------------------------------------
        # # Update Force
        # # ------------------------------------------
        
        # if self.System.interactions == True:
            
        #     uf.update_force_append_reactions(self.Particles, self.System, self.RBs, self.HGrid)
            
        #     Pressure_total, Pressure_Tensor = update_pressure(self)
            
        #%%
                
        time_stamp = time.perf_counter()
        start = time.perf_counter()
        
        time_jit = time.perf_counter()
        
        done = False
        j = 0
        self.current_step = j
        self.System.current_step = j
        while not done:
                
            
            start_loop = time.perf_counter()
            
            #%%
            
            # ------------------------------------------
            # Progress
            # ------------------------------------------

            if j%progress_stride == 0 and j>1:
                # if (time.perf_counter()-time_stamp)>0.0:
                itps = progress_stride/(time.perf_counter()-time_stamp)
                # else:
                #     itps = 10000
                    
                self.progress(self.sim_time, self.nsteps,j, time.perf_counter()-start, itps, self.System.Np, Pressure_total*1e3*lu_prefix**3/self.System.avogadro*1e-5, self.System.N_bonds, self.System.volume, out_linebreak, self.System.N)
                
                time_stamp = time.perf_counter()
            
            if j == 1: # We start measuring time after the first iteration so the result does not get biased by the initial jit compilation!
                start = time.perf_counter()
                time_stamp = time.perf_counter()
                
                if keep_time and self.Observer is not None:
                    self.Observer.Time_passed.append(time.perf_counter()-start)
                
            #%%
            
            if print_system_count and j%progress_stride == 0:
                print('count:', self.System.count)
            
            self.System.count = 0
            
            
            #%%
            
            for key in self.System.Reactions_Dict:
                self.System.Reactions_Dict[key].re_initialize()
                
            #%%
            # ------------------------------------------
            # Update Positions
            # ------------------------------------------
            
            start_pos = time.perf_counter()
            
            if self.System.mesh == True:
                upos.update_rb_compartments(self.RBs, self.Particles, self.System)
            else:
                upos.update_rb(self.RBs, self.Particles, self.System)
                
            if j>0:
                end_pos = time.perf_counter() - start_pos
                self.Timer['integrate']+=end_pos

            
            #%%
            
            # ------------------------------------------
            # Update Force
            # ------------------------------------------
            
            start_force = time.perf_counter()
            
            if self.System.interactions == True:
                
                uf.update_force_append_reactions(self.Particles, self.System, self.RBs, self.HGrid)
                
                Pressure_total, Pressure_Tensor = update_pressure(self)
                
                #Ideales Gas: P =2/3*N/V*Ekin, Ekin = 1/2*m*v^2 = f/2*kB*T (3D: f=3)
                
                # Pressure_Tensor = (self.System.N*self.System.kbt + Vir_Tensor)/self.System.volume
                
            if j>0:
                end_force = time.perf_counter() - start_force
                self.Timer['force']+=end_force
            
            
            #%%
            # ------------------------------------------
            # Evaluate Reactions
            # ------------------------------------------
            
            if j>=start_reactions:
                
                start_react = time.perf_counter()
                
                if len(self.System.Reactions_Dict)>0:
                    uru.update_reactions(self.System, self.Particles, self.RBs)
                
                for event in self.release_events:
                    
                    if j == event['timepoint']:
                        self.release_molecules(event)
                        
                if j>0:
                    end_react = time.perf_counter() - start_react
                    self.Timer['reactions']+=end_react
                
            else:
                self.System.reactions_left = 0
                
            #%%
            
            # ------------------------------------------
            # Handle fixed concentration boundary
            # ------------------------------------------
            
            if self.System.boundary_condition_id == 2:
                distribute_vol.release_molecules_boundary(self.System, self.RBs, self.Particles)
                distribute_surf.release_molecules_boundary_2d(self.System, self.RBs, self.Particles)
            
            #%%
            
            # UPDATE FORCES FOR REACTION PRODUCTS
            
            #%%
            # ------------------------------------------
            # Update Barostat
            # ------------------------------------------
            
            start_barostat = time.perf_counter()
            
            if self.System.barostat[0]['active'] == True and j>=self.System.barostat[0]['start']:
    
                self.System.barostat[0]['mu']=(1+self.System.dt/self.System.barostat[0]['Tau_P']*(Pressure_total-self.System.barostat[0]['P0']))**(1/3)
                
                self.System.box_lengths*=self.System.barostat[0]['mu']
                self.System.volume = self.System.box_lengths.prod()
                
                #----------------------------------
                #=====================
                # Anisotropic barostat:
                #=====================    
                
                # self.System.barostat[0]['mu_tensor'][:,:]=(np.eye(3)+self.System.dt/self.System.barostat[0]['Tau_P']*(Pressure_Tensor-self.System.barostat[0]['P0']))**(1/3)
                
                # #We assume that our simulation box is rectangular!
                # self.System.box_lengths[0]=self.System.barostat[0]['mu_tensor'][0][0]*self.System.box_lengths[0]
                # self.System.box_lengths[1]=self.System.barostat[0]['mu_tensor'][1][1]*self.System.box_lengths[1]
                # self.System.box_lengths[2]=self.System.barostat[0]['mu_tensor'][2][2]*self.System.box_lengths[2]
                
                # self.System.volume = self.System.box_lengths.prod()
                
                #----------------------------------
                
                
                hu.update_hgrid(self.HGrid, self.System.box_lengths, self.System.Np, self.System)
                
                if self.Observer is not None:
                    if self.Observer.observing_rdf == True:
                        update_rb_hgrid(self.Observer.rdf_hgrid, self.System.box_lengths, self.System.N, self.System)
                
                self.System.update_rb_barostat(self.RBs, self.Particles)
                
            if j>0:
                end_barostat = time.perf_counter() - start_barostat
                self.Timer['barostat']+=end_barostat
            
            
            #%%
            
            # ------------------------------------------
            # Update HGrid
            # ------------------------------------------
            
            if len(self.HGrid[0]['ls'])<=self.System.Np or len(self.HGrid[0]['head'])<np.sum(self.HGrid[0]['NCells']):
            
                self.HGrid = hu.create_hgrid(self.Particles, self.System.particle_types, self.System.box_lengths, self.System.Np, self.System)
                
            
            if self.Observer is not None:
                if self.Observer.observing_rdf == True:
                    if len(self.Observer.rdf_hgrid[0]['ls'])<=self.System.N:
                    
                        self.Observer.rdf_hgrid = create_rb_hgrid(self, self.RBs, self.Observer.rdf_molecules, self.System.box_lengths, self.System.N, self.System)
                
            
            #%%
            
            # ------------------------------------------
            # Update Observables
            # ------------------------------------------
            
            start_obs = time.perf_counter()
            
            if self.Observer is not None:
                
                self.Observer.update_bins(self)
                
                self.Observer.update(self, self.hdf, self.RBs)
                
                if self.Observer.observing_rdf == True:
                    self.Observer.update_rdf(self, self.hdf)
            
            if j>0:
                end_obs = time.perf_counter() - start_obs
                self.Timer['observables']+=end_obs
            
            
            #%%
            # ------------------------------------------
            # Write to File
            # ------------------------------------------
            
            start_write = time.perf_counter()
            
            if j!=0:
                if self.checkpoint[0]['max_saves']>0 and j%self.checkpoint[0]['stride'] == 0:
                    
                    cp.save(self, self.System, self.RBs, self.Particles, self.HGrid, checkpoint_counter)
                    checkpoint_counter += 1
                    
                    if checkpoint_counter>=self.checkpoint[0]['max_saves']:
                        checkpoint_counter = 0
            
                wru.write(j, self, self.System, self.Particles)
                
            if j>0:
                end_write = time.perf_counter() - start_write
                self.Timer['write']+=end_write


            #%%
            
            if j>0:
                end_loop = time.perf_counter() - start_loop
                self.Timer['total'] += end_loop
                self.Timer['pu/s'] += self.System.Np/end_loop
                
            #%%
            
            if j==self.nsteps or time.perf_counter()-start>self.sim_time:
                done=True
                
            j+=1
            self.current_step = j
            self.System.current_step = j
            
            #%%
            
            if j==1:
                print('Simulation started!'+'(JIT Compilation Time:{:1.1f} s)'.format(time.perf_counter()-time_jit))
                

             
        #%%
        print('\n')
        
        total_time=time.perf_counter()-start
        print('Time per steps: ', total_time/self.current_step*1e3, 'ms')
        print('it/s: ', self.current_step/total_time)
        
        if self.Observer is not None:
            self.hdf.close() 
           
        #%%
        
        box_lengths = self.System.box_lengths
            
        self.particles_left_box = []
        for i0 in range(self.Particles.occupied.n):
            i = self.Particles.occupied[i0]
            
            pos_i = self.Particles[i]['pos']

            within_box = -box_lengths[0]/2<=pos_i[0]<box_lengths[0]/2 and -box_lengths[1]/2<=pos_i[1]<box_lengths[1]/2 and -box_lengths[2]/2<=pos_i[2]<box_lengths[2]/2
        
            if within_box == False:
                self.particles_left_box.append(i)
                
        if len(self.particles_left_box)>0:
            warnings.warn('Warning: Molecules have left the simulation box! You may want to choose a smaller time step and/or check the initial distribution of molecules (avoid molecules that overlap while also repelling each other). To check which particles have left the simualtion box call Simulation.particles_left_box')
        
    
