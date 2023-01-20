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
    
    """Converts a python dictionary to a Numba typed dictionary.
    
    Parameters
    ----------
    untyped_dict : `dict`
        Untyped python dictionary
    
    Raises
    ------
    NotImplementedError (just an example)
        Brief explanation of why/when this exception is raised
    
    Returns
    -------
    nb.types.DictType
        Numba typed dictionary
    
    """

    key_type = nb.typeof(list(untyped_dict.keys())[0])
    value_type = nb.typeof(list(untyped_dict.values())[0])
    
    typed_dict = nb.typed.Dict.empty(key_type, value_type)
    
    for key in untyped_dict:
        typed_dict[key] = untyped_dict[key]
    
    return typed_dict


@nb.njit#(cache=True)
def random_quaternion():
    
    """Returns a random rotation quaternion.
    
    Notes
    -----
    The method can be found in K. Shoemake, Graphics Gems III by D. Kirk, pages 124-132 :cite:p:`Kirk2012` and on http://planning.cs.uiuc.edu/node198.html:
        
        .. math::
            
            \\boldsymbol{q} = [\\sqrt{1-u_1} \\, \\sin(2 \\pi u_2), \\sqrt{1-u_1} \\, \\cos(2 \\pi u_2), \\sqrt{u_1} \\, \\sin(2 \\pi u_3), \\sqrt{u_1} \\, \\cos(2 \\pi u_3)] ,
    
            
    where :math:`u_1,u_2,u_3 \\in [0,1]` are uniformly distributed random numbers. :math:`\\boldsymbol{q}` is a uniformly distributed, random rotation quaternion.
                                                   
    
    Returns
    -------
    float64[4]
        Unit quaternion representing a uniformly distributed random rotation / orientation.
    
    """
    
    
    u1,u2,u3 = np.random.rand(3)
    
    return np.array([np.sqrt(1-u1)*np.sin(2*np.pi*u2), np.sqrt(1-u1)*np.cos(2*np.pi*u2), np.sqrt(u1)*np.sin(2*np.pi*u3), np.sqrt(u1)*np.cos(2*np.pi*u3)])


def update_pressure(Simulation):

    """Calculates the pressure from the virial tensor.
    
    Parameters
    ----------
    Simulation : `object`
        Instance of the Simulation class.
    
    Notes
    -----
    When doing rigid body molecular dynamics or Brownian dynamics simulations, we need to be careful how to calculate the virial tensor. 
    Taking the interparticle distances will result in the wrong pressure. 
    Instead, one needs to use the pairwise distance between the centers of mass of the rigid bodies :cite:p:`Glaser2020` :
        
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
    The Simulation class takes care of most of the user communication with PyRIDs numba compiled functions that are the core of the PyRID simulator. 
    The Simulation class thereby represents the biggest part of the user API. 
    By creating an instance of the Simulation class and calling its methods, the user can define molecule types, setup the simulation volume, 
    define reactions etc. and, last but not least, start the simulation.
    
    Attributes
    ----------
    System : `object`
        Instance of the System class
    Particles : `object`
        Instance of the Particles class
    RBs : `object`
        Instance of the RBs class
    file_path : `object`
        Directory for simulation results. Uses the pathlib library.
    file_name : `string`
        Project name for the simulation files.
    fig_path : `object`
        Directory for any figures created with PyRID. Uses the pathlib library.
    write_trajectory : `boolean`
        If false, PyRID will not save any molecule trajectory data. Default = True
    stride : `int64`
        Stride used for saving molecule trajectories to the hard drive.
    current_step : `int64`
        Current simulation time step.
    release_events = `list`
        List of molecule release events
    nsteps : `int64`
        Number of simulation steps.
    sim_time : `float64`
        Instead of a number of simulation steps, the user can also define the simulation time/duration. 
        This option is useful, e.g., when running PyRID on a cluster with maximum runtime for users.
    Observer : `object`
        Instance of the Observables class.
    Measures : `list`
        List of all measures / observables supported by PyRID.
    length_units : `dict`
        Dictionary assigning written out unit name and unit short form.
    units : `dict`
        Dictionary containing for each system property / observable the corresponding unit.
    time_unit : 'ns', 'mus', 'ms', 's'
        Unit of time
    checkpoint : `array like`
        Structured array that contains fields for different checkpoint parameters (stride, directory, maximum number of saves).
    Timer : `dict`
        Dictionary that saves the runtime for different parts of the simulation loop such as integration, reaction handling and writing data to files.
    progress_properties : `list`
        List of simulation properties to print/log while the simulation is running. 
        Supported are 'it/s' (iterations per second), 'pu/s' (particle updates per second), 'Time passed', 'step', 'N' (Number of molecules), 'Pressure', 
        'Volume' (Simulation box volume), 'Vol_Frac' (Volume fraction per molecule type), 'Bonds' (Number of particle-particle bonds).
    hdf : `object`
        hdf5 file that PyRID writes all observable data to.
    HGrid : `array like`
        Hierarchical grid represented by a numpy structures array.
    particles_left_box : `list`
        List of particles that have left the simulation volume. 
        If this list is not empty this is always a sign that some system parameters need to changed because the simulation is unstable.




    Methods
    -------
    print_timer(self)
        Prints the runtime of different parts of the simulation loop such as integration, reaction handling and writing data to files.
    k_micro(self, mol_type_1, mol_type_2, k_macro, R_react, loc_type = 'Volume')
        Calculates the microscopic reaction rate from a given macroscopic rate, reaction radius and the educts´ diffusion coefficients. 
        Calls :func:`pyrid.reactions.reactions_util.k_micro`.
    k_macro(self, D1, D2, k_micro, R_react)
        Calculates the macroscopic reaction rate from a given microscopic rate, reaction radius and the educts´ diffusion coefficients. 
        Calls :func:`pyrid.reactions.reactions_util.k_macro`.
    add_checkpoints(self, stride, directory, max_saves)
        Registers the parameters for saving checkpoints.
    add_barostat_berendsen(self, Tau_P, P0, start)
        Adds a Berendsen barostat. Calls :func:`pyrid.system.system_util.System.add_barostat_berendsen`.
    set_compartments(self, Compartments, triangles, vertices, mesh_scale = 1.0, adapt_box = False)
        Sets up 3d mesh compartments.
    register_particle_type(self, Type, radius)
        Registers a particle type. Calls :func:`pyrid.system.system_util.System.register_particle_type`.
    add_interaction(self, interaction_type, type1, type2, parameters, bond = False)
        Assigns a particle-particle interaction potential to a particle type pair. Calls :func:`pyrid.system.system_util.System.add_interaction`.
    add_up_reaction(self, reaction_type, educt_type, rate, product_types = None, product_loc = None, product_direction = None, radius = None)
        Registeres a Uni-Particle (UP) reaction for an educt particle. Calls :func:`pyrid.system.system_util.System.add_up_reaction`.
    add_um_reaction(self, reaction_type, educt_type, rate, product_types = None, product_loc = None, product_direction = None, radius = None)
        Registeres a Uni-Molecular (UM) reaction for an educt Molecule. Calls :func:`pyrid.system.system_util.System.add_um_reaction`.
    add_bp_reaction(self, reaction_type, educt_types, product_types, rate, radius, interaction_type = None, interaction_parameters = None)
        Registeres a Bi-Particle (BP) reaction for an educt particle. Calls :func:`pyrid.system.system_util.System.add_bp_reaction`.
    add_bm_reaction(self, reaction_type, educt_types, product_types, particle_pairs, pair_rates, pair_radii, placement_factor = 0.5)
        Registeres a Bi-Molecular (BM) reaction for an educt molecule. Calls :func:`pyrid.system.system_util.System.add_bm_reaction`.
    register_molecule_type(self, molecule_name, particle_pos, particle_types, collision_type = 0, h_membrane = None)
        Regsiters a molecule type. Calls :func:`pyrid.system.system_util.System.register_molecule_type`.
    set_diffusion_tensor(self, molecule_name, D_tt, D_rr)
        Checks whether a given diffusion tensor is valid (physically meaningful) and sets the diffusion tensor for a given molecule type. 
        Calls :func:`pyrid.system.system_util.System.set_diffusion_tensor`.
    fixed_concentration_at_boundary(self, molecule_name, C, comp_name, location)
        Sets the fixed concentration boundary given a concentartion for each defined molecule type and for each compartents volume and surface.
        Calls :func:`pyrid.system.system_util.System.fixed_concentration_at_boundary`.
    add_molecules(self, Location, Compartment_Number, points, quaternion, points_type, face_ids = None)
        Places molecules in the simulation box given their location, orientation, type, and compartment number.
    distribute(self, Method, Location, Compartment_Number, Types, Number, clustering_factor=1, max_trials=100, jitter = None, triangle_id = None, multiplier = 100, facegroup = None)
        Calculates the distribution of molecules in the simulation box / the surface or volume of a compartment using one of several different methods.
    observe(self, Measure, molecules = 'All', reactions = 'All', obs_stride = 1, bin_width = None, n_bins = None, stepwise = True, binned= False)
        Sets up an observer for a certain system property / measure.
    observe_rdf(self, rdf_pairs = None, rdf_bins = 100, rdf_cutoff = None, stride = None)
        Sets up an observer for the radial distribution function.
    add_release_site(self, Location, timepoint, compartment_id, Number = None, Types = None, origin = None, jitter = None, Molecules_id = None, Positions = None, Quaternion = None, triangle_id = None)
        Adds a molecule release site that releases a number of molecules at a defined time point into the system.
    progress(self, sim_time, nsteps, j, time_passed, it_per_s, Np, Pressure_total, N_bonds, Volume, out_linebreak, N, width=30)
        prints / logs several different simulation parameters such as iterations per second, molecule number and pressure.
    load_checkpoint(self, file_name, index, directory = 'checkpoints/')
        Loads a system state from a checkpoint file.
    run(self, progress_stride = 100, keep_time = False, out_linebreak = False, progress_bar_properties = None, start_reactions = 0)
        Simulation loop. Runs the simulation.


    
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

        """Prints the runtime of different parts of the simulation loop such as integration, reaction handling and writing data to files.
        
        
        """
        
        t_total = 0.0
        for key in self.Timer:
            if key != 'pu/s':
                print(key+': ', (self.current_step-1)/self.Timer[key],' it/s | ', (self.Timer[key]/(self.current_step-1))*1e3, ' ms/it' )
                t_total += self.Timer[key]/(self.current_step-1)
            else:
                print('total pu/s: ', self.Timer[key]/(self.current_step-1),' pu/s')
            
    def k_micro(self, mol_type_1, mol_type_2, k_macro, R_react, loc_type = 'Volume'):

        """Calculates the microscopic reaction rate from a given macroscopic rate, reaction radius and the educts´ diffusion coefficients. 
        Calls :func:`pyrid.reactions.reactions_util.k_micro`.
        
        Parameters
        ----------
        mol_type_1 : `string`
            Name of educt moelcule 1.
        mol_type_2 : `string`
            Name of educt moelcule 2.
        k_macro : `float`
            Macroscopic reaction rate.
        R_react : `float`
            Reaction radius.
        loc_type : `string`
            Location type of the educts ('Volume' or 'Surface').
        
        Notes
        -----
        The method used here is only valid for volume molecules, i.e. diffusion in 3D. For surface molecules a value is also returned, however with a warning.
        
        Returns
        -------
        `float64`
            Microscopic reaction rate
        
        """
        
        k = ru.k_micro(self.System, mol_type_1, mol_type_2, k_macro, R_react, loc_type = loc_type)
            
        return k
    
    def k_macro(self, D1, D2, k_micro, R_react):

        """Calculates the macroscopic reaction rate from a given microscopic rate, reaction radius and the educts´ diffusion coefficients. 
        Calls :func:`pyrid.reactions.reactions_util.k_macro`.
        
        Parameters
        ----------
        D1 : `float`
            Translational diffusion coefficient of educt 1.
        D2 : `float`
            Translational diffusion coefficient of educt 2.
        k_micro : `float`
            Microscopic reaction rate.
        R_react : `float`
            Reaction radius.
        
        Notes
        -----
        
        
        Returns
        -------
        `float64`
            Macroscopic reaction rate
        
        """
        
        k = ru.k_macro(D1, D2, k_micro, R_react)
            
        return k
    
    def add_checkpoints(self, stride, directory, max_saves):

        """Registers the parameters for saving checkpoints such as stride, maximum number of saves and target directory.
        
        Parameters
        ----------
        stride : `int64`
            Stride for making checkpoints.
        directory : `string`
            Target directory for the checkpoint files.
        max_saves : `int64`
            Maximum number of files that are saved. Old files will be overwritten to keep the number of checkpoint files at the value of max_saves.
        
        
        """
    
        item_t_checkpoint = np.dtype([('directory', 'U40'),('max_saves', np.int32),('stride', np.int32)],  align=True)   
        self.checkpoint = np.zeros(1, dtype = item_t_checkpoint)
        
        self.checkpoint[0]['stride'] = stride
        self.checkpoint[0]['directory'] = directory
        self.checkpoint[0]['max_saves'] = max_saves
        
        
    def add_barostat_berendsen(self, Tau_P, P0, start):

        """Adds a Berendsen barostat. Calls :func:`pyrid.system.system_util.System.add_barostat_berendsen`.
        
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
        
        self.System.add_barostat_berendsen(Tau_P, P0, start)
        
    def set_compartments(self, Compartments, triangles, vertices, mesh_scale = 1.0, adapt_box = False):
        
        """Sets up 3d mesh compartments.
        
        Parameters
        ----------
        Compartments : `dict`
            Dictionary containing for each compartment number the triangle indices and face groups. Also see :func:`pyrid.geometry.load_wavefront.load_compartments`.
        triangles : `int64[:,3]`
            List of vertex indices (3) per triangle.
        vertices : `float64[:,3]`
            List of vertex coordinates.
        meshscale : `float64`
            Factor by which to scale the 3d mesh. Default = 1.0
        adapt_box : `boolean`
            If true, the simulation box size is adapted such that the mesh fits exactly in the simulation box. Default = False
        
        
        """
        
        if self.System.cells_per_dim.shape[0]!=3:
            if len(self.System.particle_types) > 0:
                self.System.add_system_grid()
            else:
                raise ValueError('Please register all particle types used in the simulation or manually define a grid size for the simulation box before adding a mesh compartment! ')
            
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

        """Registers a particle type. Calls :func:`pyrid.system.system_util.System.register_particle_type`.
        
        Parameters
        ----------
        Type : `str`
            Name of the particle type.
        radius : `float`
            radius of the particle.
        
        """

        self.System.register_particle_type(Type, radius)
        
    #%%
    
    def add_interaction(self, interaction_type, type1, type2, parameters, bond = False):

        """Assigns a particle-particle interaction potential to a particle type pair. Calls :func:`pyrid.system.system_util.System.add_interaction`.
        
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
        
        parameters = convert_dict(parameters)
        
        self.System.add_interaction(interaction_type, type1, type2, parameters, bond)
        

#%%
    
    def add_up_reaction(self, reaction_type, educt_type, rate, product_types = None, product_loc = None, product_direction = None, radius = None):
        
        """Registeres a Uni-Particle (UP) reaction for an educt particle. Calls :func:`pyrid.system.system_util.System.add_up_reaction`.
        
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
        Uni-particle reactions are evaluated using a variant of the Gillespie stochastic simulation algorithm. Thereby, for a particle type, the time point of the 
        next uni-particle reaction occuring is calculated in advance and executed when the simualtion reaches the respective time point. 
        For unimolecular reactions we can draw the time point of the next reaction from a distribution using a variant of the Gillespie Stochastic 
        Simulation Algorithm (SSA) :cite:t:`Stiles2001`, :cite:t:`Erban2007`. For a single molecule having :math:`n` possible transition reactions/reaction paths each 
        having a reaction rate :math:`k_i`, let :math:`k_t = \sum_i^n k_i` be the total reaction rate. 
        
        Now, let :math:`\\rho(\\tau) d\\tau` be the probability that the next reaction occurs within :math:`[t+\\tau,t+\\tau+d\\tau)` and let :math:`g(\\tau)` 
        be the probability that no reaction occurs within :math:`[t,t+\\tau)`. The probability that a reaction occurs within the time interval :math:`d\\tau` 
        is simply :math:`k_t d\\tau`. Thereby
        
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
        
        A uni-particle reaction can consist of several reaction paths. Each particle type has exactly one uni-particle reaction id registered for which 
        the time point is drawn randomly based on the total reaction rate which is the sum of all path reaction rates. At the time point of the reaction occurring, 
        a path is randomly drawn from the set of possible paths weighted by the individual path reaction rates. 
        To get the reaction path we compare a second random number, uninformly distributed in (0,k_t), with the cumulative set of reaction rates :math:`(k_1, k_1+k_2, ..., k_t)`. 
        The comparison is done via a bisection algorithm (see rand_util.random_util.bisect_right()).
        
        To illustrate the above by an example: For particle 'A' a we may define two convcersion reactions where A -k1-> B, A -k2-> C. 
        In addition, we define a release reaction A -k3-> 2*D+5*E. The next reaction will occur at a time point T that is calculated based on the total rate k1+k2+k3. At time T, 
        one of the three reactions is executed. The probability for each reaction to be selected being k1/(k1+k2+k3), k2/(k1+k2+k3), k3/(k1+k2+k3).
        
        """

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

        """Registeres a Uni-Molecular (UM) reaction for an educt Molecule. Calls :func:`pyrid.system.system_util.System.add_um_reaction`.
        
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
        Uni-molecular reactions are evaluated using the Gillespie stochastic simulation algorithm. Thereby, for a molecule type, the time point of the next 
        uni-molecular reaction occuring is calculated in advance and executed when the simualtion reaches the respective time point. A uni-molecular reaction can 
        consist of several reaction paths. Each molecule type has exactly one uni-molecular reaction id registered for which the time point is drawn randomly based 
        on the total reaction rate which is the sum of all path reaction rates. At the time point of the reaction occuring, a path is randomly drawn from the set of 
        possible paths weighted by the individual path reaction rates. To illustrate the above by an example: For Molecule 'A' a we may define two convcersion reactions 
        where A -k1-> B, A -k2-> C. In addition, we define a fission reaction A -k3-> D+E. The next reaction will occur at a time point T that is calculated based on the 
        total rate k1+k2+k3. At time T, one of the three reactions is executed. The probability for each reaction to be selected being k1/(k1+k2+k3), k2/(k1+k2+k3), k3/(k1+k2+k3).
        
        """
        
        if product_types is not None:
            product_types = np.array(product_types)
        if product_loc is not None:
            product_loc = np.array(product_loc)
        if product_direction is not None:
            product_direction = np.array(product_direction)
            
        self.System.add_um_reaction(reaction_type, educt_type, rate, product_types, product_loc, product_direction, radius)
            
    #%%
    
    def add_bp_reaction(self, reaction_type, educt_types, product_types, rate, radius, interaction_type = None, interaction_parameters = None):

        """Registeres a Bi-Particle (BP) reaction for an educt particle. Calls :func:`pyrid.system.system_util.System.add_bp_reaction`.
        
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
        Bi-particle reactions are evaluated using the Doi scheme. Thereby, two educt particles react with a reaction rate k if the inter-particle distance is below 
        the reaction radius.
        
        `Binding Reactions`
        
        A particle pair will only enter a bound state if the binding reaction was succesfull. Otherwise two particles won't experience any pairwise interaction force! 
        A reaction occurs if two particles for whose type a binding reaction has been defined are within the reaction radius. The reaction probability is evaluated as:
            
        .. math::
            
            1-exp(\lambda \cdot \Delta t),
        
        where :math:`\lambda` is the reaction rate and :math:`\Delta t` is the integration time step.
        
        
        """
        
        educt_1_radius = self.System.particle_types[educt_types[0]][0]['radius']
        educt_2_radius = self.System.particle_types[educt_types[1]][0]['radius']
        
        if radius < educt_1_radius+educt_2_radius:
                warnings.warn('Warning: The reaction radius for the particle pair {0}, {1} is smaller than the sum of their radii. The reaction radius should not be smaller than {2:.3g}'.format(educt_types[0], educt_types[1], educt_1_radius+educt_2_radius))
        
        if interaction_parameters is not None:
            interaction_parameters = convert_dict(interaction_parameters)
        
        self.System.add_bp_reaction(reaction_type, np.array(educt_types), np.array(product_types), rate, radius, interaction_type, interaction_parameters)
        
    #%%
    
    def add_bm_reaction(self, reaction_type, educt_types, product_types, particle_pairs, pair_rates, pair_radii, placement_factor = 0.5):

        """Registeres a Bi-Molecular (BM) reaction for an educt molecule. Calls :func:`pyrid.system.system_util.System.add_bm_reaction`.
        
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
        placement_factor : `float64` [0,1]
            Only for fusion reactions: Affects where between the educt molecules the product molecule is placed. A factor of 0.5 means in the middle. 
            A smaller factor will shift the product towards the first educt, a larger value towards the second educt molecule. 
            A value different from 0.5 may increase accuracy in the case of a strong size difference between the two educts (a strong difference in the diffusion tensors). 
            Default = 0.5
        
        Notes
        -----
        Bi-molecular reactions are evaluated using the Doi scheme. Thereby, two educt particles react with a reaction rate k if the inter-particle distance is 
        below the reaction radius. PyRID only evaluates the eucludean distances between particles but not molecule centers. Thereby, also for bi-molecular reactions 
        educt particles need to be set. Since molecules can consist of several particles, a list of particle pairs can be passed and for each particle pair a reaction 
        rate and radius is defined. If one of the pair particle reactions is successful, the corresponding molecule that the particle belongs to is converted as defined 
        by the reaction type. If the molecule cosnsists of a single particle, the reaction will be handled just as the bi-particle reactions. The user needs to take care 
        of calculating accurate reaction radii and rates such that the desired reaction dynamics are simulated, which can become difficult for large molecules. 
        Therefore it is sometimes more convenient to place an interaction free particle at the origin of the molecule and execute the bimolecular reaction via this particle. 
        This method is equivalent to the case where one would evaluate reactions based on the distance between molecule centers.
        As for the uni-molecular reactions, several reaction paths can be defined. Each reaction path may involve different particle pairs of the educt molecules. 
        On a side note: Thereby, orientation dependent bimolecular reactions are in principle possible. However, since the reaction probability does not scale with 
        the distance, directionality is only really accounted for if the reaction radius is small compared to the molecule size.
        
        
        """
        
        for i,particle_types in enumerate(particle_pairs):
            educt_1_radius = self.System.particle_types[particle_types[0]][0]['radius']
            educt_2_radius = self.System.particle_types[particle_types[1]][0]['radius']
            
            if pair_radii[i] < educt_1_radius+educt_2_radius:
                warnings.warn('Warning: The reaction radius for the particle pair {0}, {1} is smaller than the sum of their radii. The reaction radius should not be smaller than {2:.3g}'.format(particle_types[0], particle_types[1], educt_1_radius+educt_2_radius))
            
        self.System.add_bm_reaction(reaction_type, np.array(educt_types), np.array(product_types), np.array(particle_pairs), np.array(pair_rates), np.array(pair_radii), placement_factor)
            
    #%%
    
    def register_molecule_type(self, molecule_name, particle_pos, particle_types, collision_type = 0, h_membrane = None):

        """Regsiters a molecule type. Calls :func:`pyrid.system.system_util.System.register_molecule_type`.
        
        Parameters
        ----------
        molecule_name : `str`
            Name of the molecule.
        particle_pos : `int[:,3]`
            Positions of each of the molecule's particles.
        particle_types : `array_like (dtype = 'U20')`
            Array of names for each of the molecule's particles.
        collision_type : `int {0,1}`
            Collision type of the molecule (Default = 0). 0: Collisions with compartments are handled via interaction force. 
            1: Collisions with compartments are handled by raytracing algorithm (recommended for molcules whose diffusion speed and/or interaction radius 
            is small compared to the chosen integration time step).
        h_membrane : `float64`
            Distance of the molecules´ center of diffusion from the membrane surface. Default = None
        
        
        """
        
        particle_pos = np.array(particle_pos)
        particle_types = np.array(particle_types, dtype = np.dtype('U20'))
        
        self.System.register_molecule_type(molecule_name, particle_pos, particle_types, collision_type, h_membrane)
        
    def set_diffusion_tensor(self, molecule_name, D_tt, D_rr):

        """Checks whether a given diffusion tensor is valid (physically meaningful) and sets the diffusion tensor for a given molecule type. 
        Calls :func:`pyrid.system.system_util.System.set_diffusion_tensor`.
        
        Parameters
        ----------
        molecule_name : `str`
            Some Information
        D_tt : `float[3,3]`
            Translational diffusion tensor ()
        D_rr : `float[3,3]`
            Rotational diffusion tensor

        
        Notes
        -----
        PyRID offers a module for the calculation of the diffusion tensors of rigid bead models (see molecules_util.hydro_util). It uses the Kirkwood-Riseman calculation with modified Oseen tensor as introduced by :cite:t:`Carrasco1999`, :cite:t:`Carrasco1999a`. The same method has been implemented in the `HYDRO++ <http://leonardo.inf.um.es/macromol/programs/hydro++/hydro++.htm>`_ tool, which is free to use but not open source! As such, you may also use HYDRO++ for the calculation of the diffusion tensors. Or even better, use both and tell me if you you get inconsistent results ;) !
        
        .. admonition:: Note
        
            Diffusion tensor should be a real positive semidefinite matrix!
        
        
        """
        
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

        """Sets the fixed concentration boundary given a concentartion for each defined molecule type and for each compartents volume and surface.
        Calls :func:`pyrid.system.system_util.System.fixed_concentration_at_boundary`.
        
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
        fixed_concentration_at_boundary() calculates some properties that are necessary to properly distribute molecules inside the simulation box that hit 
        the simulation box boundary from the outside (Thereby, 'virtual molecules' become `real` molecules in our simualtion). The number of hits per time step 
        a boundary of area A experiences is :math:`N = l_{perp}*A*C`. Where :math:`C` is the concentration in molecules per volume and :math:`l_{perp}` is the 
        average net displacement in one tiem step towards or away from any plane, where :math:`l_{perp} = \sqrt{(4*D*\Delta t/\pi)}` :cite:t:`Stiles2001`.
        
        
        """
        
        if location == 'Volume':
            self.System.fixed_concentration_at_boundary(molecule_name, C, comp_name, 0)
        elif location == 'Surface':
            self.System.fixed_concentration_at_boundary(molecule_name, C, comp_name, 1)
        else:
            raise NameError('Location must be either Volume or Surface!')            
        
    def add_molecules(self, Location, Compartment_Number, points, quaternion, points_type, face_ids = None):
        
        """Places molecules in the simulation box given their location, orientation, type, and compartment number.
        
        Parameters
        ----------
        Location : 'Volume' or 'Surface''
            Determines whether the molecules are surface or volume molecules.
        Compartment_Number : `int64`
            The number of the compartment in which to place the molecules
        points : `float64[:,3]`
            Molecule coordinates
        quaternion : `float64[:,4]`
            Orientation of the molecules represented by rotation /orientation quaternions.
        points_type : `int64[:]`
            Molecule type ids.
        face_ids : `int64[:]`
            Indices of the mesh triangle/faces on which the molecules are placed. Default = None

        
        Raises
        ------
        raise IndexError('Compartment number must be a positive integer!')
            Compartment number must be an integer value.
        raise IndexError('Compartment number must be greater zero!')
            Compartment numbers must be greater than zero.
        raise TypeError('Missing required argument face_ids')
            In the case of Location = 'Surface' face_ids must be passed.
        NameError('Location must be either Volume or Surface!')
            There are only two locations a molecule can assigned to: the surface or the volume of a compartment.
        
        
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
                raise IndexError('Compartment number must be a positive integer!')
                
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

        """Calculates the distribution of molecules in the simulation box / the surface or volume of a compartment using one of several different methods.
        The available methods are: 

        * 'PDS': The 'PDS' (Poisson-Disc-Sampling) method does resolve collisions based on the molecule radius but can only reach volume fractions of <30% for mono-disperse systems. Note that computation times can become very long if the target number of molecules is much larger than would fit into the compartment. The molecule density can also be increased by increasing max_trials at the cost of longer runtimes.In additions one can play around with the clustering factor that affects the distribution of molecules. A larger clustering factor will result in a more uniform distribution of molecules. See :func:`pyrid.system.distribute_vol_util.pds`
        * 'PDS_unfiform': The same holds true for 'PDS_unfiform' method, however, with the addition that the molecules are uniformly distributed. The method first creates a large number of uniformly distributed points (using the 'MC' method). From this sampling pool only those points that are a minimum distance apart from each other are kept. The molecule density can be increased by the multiplier parameter at the cost of longer runtimes.See :func:`pyrid.system.distribute_vol_util.pds_uniform` and :func:`pyrid.system.distribute_surface_util.pds`
        * 'Gauss': The 'Gauss' method can be used to normally distribute molecules around the centroid of a triangle (given by triangle_id) on a mesh surface. The distribution width can be set by the jitter parameter. See :func:`pyrid.system.distribute_surface_util.normal`
        * 'MC': The 'MC' (Monte-Carlo) method is fast and can reach arbitrary molecule densities, however, it does not resolve any collision. See :func:`pyrid.system.distribute_surface_util.mc` and :func:`pyrid.system.distribute_vol_util.mc`
        

        Parameters
        ----------
        Method : 'PDS', 'PDS uniform', 'Gauss', 'MC'
            Method for calculating the molecule distribution. 
        Location : 'Volume' or 'Surface'
            Determines of whether the molecules are distributed on the compartment surface or in its volume. 
            Note that the method 'pds_uniform' is only available for Location='Volume' and the method 'Gauss' only for Location='Surface'.
        Compartment_Number : `int64`
            Number of the mesh compartment. 0 means simulation box, i.e. outside any mesh compartment.
        Types : `int64[:]`
            List of molecule type ids.
        Number : `int64[:]`
            Number of molecules to distribute per type.
        clustering_factor : `float64`
            Only for Location='Volume' and Method='PDS'. The clustering factor affects the distribution of molecules. A larger clustering factor will result in a 
            more uniform distribution of molecules.
        max_trials : `int64`
            Only for Location='Volume' and Method='PDS'. Increases the number attempts to randomly place a molecule in the vicinity of another molecule.
        jitter : `float64`
            Only for Location='Surface' and Method='Gauss'. Width of the normal distribution.
        triangle_id : `int64`
            Only for Location='Surface' and Method='Gauss'. Index of the triangle around whose centroid molecules are normally distributed.
        multiplier : `int64`
            Only for (Location='Volume' and Method='PDS_uniform') or (Location='Surface' and Method='PDS'). Increases the sampling pool size.
        facegroup : `string`
            Only for Location='Surface' and Method='PDS' or 'MC'. Name of the face group on whose triangles to distribute the molecules.


        
        Raises
        ------
        raise NameError('Method {} does not exist!'.format(Method))
            Available methods are 'PDS', 'PDS uniform', 'Gauss', 'MC'.
        raise IndexError('Compartment number must be a positive integer!')
            Compartment number must be an integer value.
        raise IndexError('Compartment number must be greater zero!')
            Compartment numbers must be greater than zero.
        raise TypeError('Missing required argument face_ids')
            In the case of Location = 'Surface' face_ids must be passed.
        NameError('Location must be either Volume or Surface!')
            There are only two locations a molecule can assigned to: the surface or the volume of a compartment.
        
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
    
    def observe(self, Measure, molecules = 'All', reactions = 'All', obs_stride = 1, stepwise = True, binned= False):
        
        """Sets up an observer for a certain system property / measure.
        
        Parameters
        ----------
        Measure : 'Position', 'Energy', 'Pressure', 'Volume', 'Virial', 'Virial Tensor', 'Number', 'Orientation', 'Bonds', 'Reactions', 'Force', 'Torque'
            System property which to observe.
        molecules : `list` or 'All'
            List of molecule types. Default = 'All'
        reactions : `list` or 'All'
            List of reaction types to observe. Default = 'All'
        obs_stride : `int64`
            Stride by which observables are saved to the hdf5 file.
        stepwise : `boolean`
            If True, observables are saved in a stepwise manner, i.e. the current value at the respective timestep is saved. Default = True
        binned : `boolean`
            If True, observables are binned where the bin size is equal to stride.

        
        Raises
        ------
        warnings.warn('Warning: Binning is not supported for property {0}. PyRID will instead sample {0} stepwise!'.format(Measure))
            Binning is not supported for 'Force', 'Torque', 'Orientation', 'Position'.
        
        
        """
        
        
        if self.Observer is None:
            self.Observer = Observables(self)
            
        if Measure not in self.Measures:
            raise NameError('Measure {} not known!'.format(Measure))
          
            
        if Measure in ['Force', 'Torque', 'Orientation', 'Position'] and binned == True:
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

        """Sets up an observer for the radial distribution function.
        
        Parameters
        ----------
        rdf_pairs : `nested list`
            List of molecule type pairs for which to calculate the radial distribution function. Example: [['A','A'],['A','B']].
        rdf_bins : `int64[:]`
            Number of bins used for each respective molecule pair (resolution of the rdf histogram).
        rdf_cutoff : `float64[:]`
            Cutoff distance for each respective molecule pair (choose wisely).
        stride : `int64`
            Stride by which radial distribution function is sampled and saved to the hdf5 file. Note: calculation of the rdf is computationally expensive!

        
        
        """
        
        if self.Observer is None:
            self.Observer = Observables(self)
            
        self.Observer.observe_rdf(rdf_pairs, rdf_bins, rdf_cutoff, stride, self)
        
        print('Observing RDF.')
        
        
    #%%
                
    def add_release_site(self, Location, timepoint, compartment_number, Number = None, Types = None, origin = None, jitter = None, Molecules_id = None, Positions = None, Quaternion = None, triangle_id = None):
        
        """Adds a molecule release site that releases a number of molecules at a defined time point into the system. 
        The molecules are distributed normally around a point in the volume or a mesh surface.
        
        Parameters
        ----------
        Location : 'Volume' or 'Surface'
            Determines of whether the molecules are released on the compartment surface or in its volume.
        timepoint : `int64`
            Time step at which to release the molecules.
        compartment_number : `int64`
            Number of the mesh compartment. 0 means simulation box, i.e. outside any mesh compartment.
        Number : `int64[:]`
            Number of molecules per molecule type.
        Types : `list of strings`
            List of molecule types to release.
        origin : `float64[3]`
            Only if Location = 'Volume'. Position around which to distribute the molecules upon release.
        jitter : `float64`
            Initial width of the molecule distribution. Molecules are distributed normally around the origin.
        triangle_id : `int64`
            Only if Location='Surface'. Index of the triangle around whose centroid the molecules are distributed.
        Molecules_id : `int64[:]`
            Only for Location='Volume'. In case you want a custom distribution of molecules this list contains the type id of each molecule.
        Positions : `float64[:,3]`
            Only for Location='Volume'. In case you want a custom distribution of molecules this list contains the coordinates of each molecule.
        Quaternion : `float64[:,4]`
            Only for Location='Volume'. In case you want a custom distribution of molecules this list contains the orientation of each molecule.

        
        """
        
        release_dict = {}
        release_dict['compartment_id'] = compartment_number
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
        
        """prints / logs several different simulation parameters such as iterations per second, molecule number and pressure.
        Also prints a progress bar.
        
        Parameters
        ----------
        sim_time : `float64`
            Target simulation runtime.
        nsteps : `int64`
            Target number of simulation steps.
        j : `int64`
            Current time step.
        time_passed : `float`
            Current simulation runtime.
        it_per_s : `float64`
            Iterations per second.
        Np : `int64`
            Number of particles / beads in the system.
        Pressure_total : `float64`
            Current pressure.
        N_bonds : `int64`
            Current number of particle-particle bonds.
        Volume : `float64`
             Current simulation box volume.
        out_linebreak : `boolean`
            If True, after each progress print there is a line break.
        N : `int64`
            Current number of molecules in the system.
        width : `int64`
            Width of the progress bar.

        
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
        
        """Loads a system state from a checkpoint file.
        
        Parameters
        ----------
        file_name : `string`
            Checkpoint file name.
        index : `int64`
            Index of the checkpoint file (index goes from 0 to maximum number of saves).
        directory : `string`
            Directory of the checkpoint files. Default = 'checkpoints/'
        
        
        """
        
        file_extension = '.npz'
        if file_name[-4:] == '.npz':
            file_name = file_name[0:-4]
            
        self.RBs, self.Particles, self.HGrid = load_checkpoint(self.System, directory = directory, file = file_name + f'_{index}' + file_extension)

        self.System.box_lengths[0] *= 1
        self.System.volume = self.System.box_lengths.prod()
        
        
    #%%
    
    
    def run(self, progress_stride = 100, keep_time = False, out_linebreak = False, progress_bar_properties = None, start_reactions = 0):
        
        """Simulation loop. Runs the simulation.
        
        Parameters
        ----------
        progress_stride : `int64`
            Stride by which the progress bar is updated. Default = 100
        keep_time : `boolean`
            If true, different parts of the simulation loop are timed such as integration of the equations of motion or writing data to files. Default = False
        out_linebreak
            If True, after each progress print there is a line break. Default = False
        progress_bar_properties : `list of strings`
            List of properties to print/log with the progress bar during simulation. Available are: 
            'it/s', 'pu/s', 'Time passed', 'step', 'N', 'Pressure', 'Volume', 'Vol_Frac', 'Bonds' or 'All'.
        start_reactions = `int64`
            Set the time step from which reactions are allowed. Default = 0
        
        Raises
        ------
        raise ValueError('progress_bar_properties must be a list of strings!')
            If progress_bar_properties is not None or 'All' a list of strings must be passed.
        raise ValueError(str(prop)+'is not a supported property to be displayed in the progress bar!')
            A progress bar property has been passed that is not available.
        
        
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
        
    
