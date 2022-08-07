# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""


import numpy as np
import numba as nb
from numba.experimental import jitclass
from ..geometry.ray_march_util import ray_march_volume, ray_march_surface
from ..data_structures.dynamic_array_util import DenseArray, HolesArray
from ..reactions.update_reactions import delete_molecule


#%%

    
item_t_RB = np.dtype([('next', np.int64),('name', 'U20'),('id', np.int64),('type_id', np.int64), ('pos', np.float64, (3,)), ('dX', np.float64, (3,)), ('force', np.float64, (3,)), ('torque', np.float64, (3,)), ('topology', np.int64, (20,)),('topology_N', np.int64),('q', np.float64, (4,)), ('dq', np.float64, (4,)),('B', np.float64, (4,4)),('orientation_quat', np.float64, (3,3)),('mu_tb', np.float64, (3,3)),('mu_rb', np.float64, (3,3)),('mu_tb_sqrt', np.float64, (3,3)),('mu_rb_sqrt', np.float64, (3,3)),('Dtrans', np.float64),('Drot', np.float64),('radius', np.float64), ('loc_id', np.int64), ('compartment', np.int64), ('triangle_id', np.int64), ('pos_last', np.float64, (3,)), ('Theta_t', np.float64, (3,)), ('Theta_r', np.float64, (3,)), ('posL', np.float64, (3,)), ('collision_type', np.int64), ('next_transition', np.float64), ('h', np.int64), ],  align=True)

spec_holes_array = [
    ('n', nb.int64),
    ('capacity', nb.int64),
    ('Data', nb.typeof(np.empty(1, dtype = item_t_RB))),
    ('item_t', nb.typeof(item_t_RB)),
    ('i', nb.int64),
    ('occupied', DenseArray.class_type.instance_type),
    ('slot', nb.int64),
    ('continue_update', nb.types.boolean),
]

spec_RBs = [
    ('a', nb.int64),
]

@jitclass(spec_holes_array+spec_RBs)
class RBs(HolesArray):
    
    """
    A rigid body/bead model object is defined by a set of particles with fixed positions relative to the body center. 
    In our framework we use rigid bodies as a course grained representation of molecules.
    A rigid bead model has a center of diffusion around which the rigid bead model rotates in response to torque and rotational diffusion. 
    By default a surface molecule is bound to the mesh surface at the center of diffusion or shifted by an offset passed by the user.
    A rigid bodies topology is described by a list of particle indices. A RBs instance is initializeed without any molecules. Use the add_RB method to add molecules to the list.
    
    Attributes
    ----------
    Data : `array_like`
        Numpy structured array containing all data that define a moelcules state.
        `dtype = np.dtype([('next', np.uint64),('name', 'U20'),('id', np.int32),('type_id', np.int32), ('pos', np.float64, (3,)), ('dX', np.float64, (3,)), ('force', np.float64, (3,)), ('torque', np.float64, (3,)), ('topology', np.int32, (20,)),('topology_N', np.int32),('q', np.float64, (4,)), ('dq', np.float64, (4,)),('B', np.float64, (4,4)),('orientation_quat', np.float64, (3,3)),('mu_tb', np.float64, (3,3)),('mu_rb', np.float64, (3,3)),('mu_tb_sqrt', np.float64, (3,3)),('mu_rb_sqrt', np.float64, (3,3)),('Dtrans', np.float64),('Drot', np.float64),('radius', np.float64), ('loc_id', np.int32), ('compartment', np.int64), ('triangle_id', np.int32), ('pos_last', np.float64, (3,)), ('Theta_t', np.float64, (3,)), ('Theta_r', np.float64, (3,)), ('posL', np.float64, (3,)), ('collision_type', np.int64), ('next_transition', np.float64), ],  align=True)`
    pos : array 
        position of the center of mass
    
    Methods
    -------
    add_RB(name, compartment, System, Particles, vol_surf_id)
        Adds a new rigid body molecule to the Data array.
    next_um_reaction(System, i)
        Calculates the time of the next unimolecular reaction.
    set_triangle(tri_id, i)
        Set the triangle id of a surface molecule.
    add_particle(pi, i)
        Adds a particle to the molecule topology list.
    update_topology(System, Particles, i)
        Updates the positions of all particles in the molecule.         
    place_particles(pos, types, radii, System, Particles, i)
        Registeres the particle positions, radii and types of a molecule.  
    set_pos(Particles, System, pos, i)
        Sets the position of moelcule i to pos.       
    update_force_torque(Particles, i)
        Updates the total force and torque acting on molecule i. 
    update_B(i)        
        Updates matrix B which is used to update the moelcule's rotation quaternion. 
    calc_orientation_quat(i)
        Calculated the orientation matrix of the molecule from it's' rotation quaternion.
    set_orientation_quat(Particles, q, System, i)
        Rotates the molecule according to quaternion q.
    update_orientation_quat(i)
       Updates the molecules orientation matrix and the B-matrix.
    update_dq(System, i)
        Updates the rotational displacement dq.
    update_dX(System, i)
        Updates the translational displacement dX.
    update_particle_pos(Particles, System, i)
        Updates the position and orientation of volume molecule i (and all its particles).
    update_particle_pos_2D(System, Particles, i)
        Updates the position and orientation of surface molecule i (and all its particles).
    rescale_pos(Particles, System, mu, i)
        Rescales the position molecule i's origin by a factor mu (needed for the Berendsen barostat).

    """
    
    __init__HolesArray = HolesArray.__init__
    
    def __init__(self):
        self.__init__HolesArray(item_t_RB)
        
        
    def add_RB(self, name, compartment, System, Particles, vol_surf_id):
        
        """Adds a rigid bead molecule to the system.
        
        Parameters
        ----------
        name : `string`
            name of the molecule type
        compartment : `int64`
            Id of the compartment the moelcule sites in
        System : `object`
            Instance of System class
        Particles : `object`
            Instance of Particles class        
        
        
        """
        
        self.insert(np.zeros(1, dtype = item_t_RB)[0])
        
        self.Data[self.slot]['name'] = name
        self.Data[self.slot]['compartment'] = compartment
        self.Data[self.slot]['loc_id'] = vol_surf_id
        self.Data[self.slot]['id'] = self.slot-1
        self.Data[self.slot]['type_id'] = System.molecule_types[name].type_id
        self.Data[self.slot]['orientation_quat'][:,:] = np.eye(3)
        self.Data[self.slot]['q'][:] = np.zeros(4, dtype = np.float64)
        self.Data[self.slot]['q'][0] = 1.0
        self.Data[self.slot]['mu_tb'][:,:] = System.molecule_types[name].mu_tb
        self.Data[self.slot]['mu_rb'][:,:] = System.molecule_types[name].mu_rb
        self.Data[self.slot]['mu_tb_sqrt'][:,:] = System.molecule_types[name].mu_tb_sqrt
        self.Data[self.slot]['mu_rb_sqrt'][:,:] = System.molecule_types[name].mu_rb_sqrt
        self.Data[self.slot]['Dtrans'] = System.molecule_types[name].Dtrans
        self.Data[self.slot]['Drot'] = System.molecule_types[name].Drot
        #self.Data[self.slot]['radius'] = System.molecule_types[name].radius
        self.Data[self.slot]['collision_type'] = System.molecule_types[name].collision_type
        
        self.place_particles(System.molecule_types[name].pos, System.molecule_types[name].types, System.molecule_types[name].radii, System, Particles, self.slot-1, System.molecule_types[name].h_membrane)
        
        self.next_um_reaction(System, self.slot-1)
        
        System.N += 1
        System.Nmol[self.Data[self.slot]['type_id']] += 1
        
        
    def next_um_reaction(self, System, i):
        
        """Draws the time point of the next unimolecular diffusion for moelcule i.
        
        Parameters
        ----------
        System : `object`
            Instance of System class
        i : `int64`
            Molecule index
        
        Notes
        -----
        For unimolecular reactions we can draw the time point of the next reaction from a distribution using a variant of the Gillespie Stochastic Simulation Algorithm (SSA) :cite:t:`Stiles2001`, :cite:t:`Erban2007`. For a single molecule having :math:`n` possible transition reactions/reaction paths each having a reaction rate :math:`k_i`, let :math:`k_t = \sum_i^n k_i` be the total reaction rate. Then, we can draw the time point of the next reaction from:
            
        .. math::
            \\tau = \\frac{1}{k_t} \ln\Big[\\frac{1}{r}\Big],
            
        where :math:`r` is a random number uninformly distributed in (0,1).
        
        """
        
        i+=1
        
        type_id = self.Data[i]['type_id']
        mol_name = System.molecule_id_to_name[type_id]
        
        if System.molecule_types[str(mol_name)].um_reaction:
            
            total_rate = System.molecule_types[str(mol_name)].transition_rate_total
            
            self.Data[i]['next_transition'] = System.current_step*System.dt + 1/total_rate*np.log(1/np.random.rand())
        
        else:
            self.Data[i]['next_transition'] = 1e10
    
    def set_triangle(self, tri_id, i):
        
        """Sets the triangle id for surface molecule i.
        
        Parameters
        ----------
        triangle_id : `int64`
            Index of the triangle
        i : `int64`
            Molecule index
        
        """
        
        i+=1
        
        self.Data[i]['triangle_id'] = tri_id
        
        
    def add_particle(self, pi, i):
        
        """Adds a new particle to the rigid bead molecule topology.
        
        Parameters
        ----------
        pi : `int64`
            Particle index
        i : `int64`
            Molecule index
        
        Raises
        ------
        IndexError('Maximum number of particles per molecule (20) reached! To change this, currently you have to manualy set the maximum size in rigidbody_util.py')
            by default, currenztly, a maximum number of 20 particles per rigid bead molecule type is allowed. The user may incerase this limit by increasing the size of the field 'topology' in the structured numpy array dtype 'item_t_RB' (see rigi_body_util.py)
        
        """
        
        i += 1
        
        if self.Data[i]['topology_N'] == 19:
            raise IndexError('Maximum number of particles per molecule (20) reached! To change this, currently you have to manualy set the maximum size in rigidbody_util.py')
            
        n = self.Data[i]['topology_N']
        self.Data[i]['topology'][n] = pi
        self.Data[i]['topology_N'] += 1
            
        
    def update_topology(self, System, Particles, i):
        
        """Updates the positions of the particles that make up the topology of molecule i.
        
        Parameters
        ----------
        System : `object`
            Instance of System class
        Particles : `object`
            Instance of Particles class
        i : `int64`
            Molecule index
        
        """
        
        i+=1
        
        for pi0 in range(self.Data[i]['topology_N']):
            pi = self.Data[i]['topology'][pi0]
            
            Particles[pi]['pos_local'][0] = self.Data[i]['orientation_quat'][0][0]*Particles[pi]['coord_local'][0]+self.Data[i]['orientation_quat'][0][1]*Particles[pi]['coord_local'][1]+self.Data[i]['orientation_quat'][0][2]*Particles[pi]['coord_local'][2]
            
            Particles[pi]['pos_local'][1] = self.Data[i]['orientation_quat'][1][0]*Particles[pi]['coord_local'][0]+self.Data[i]['orientation_quat'][1][1]*Particles[pi]['coord_local'][1]+self.Data[i]['orientation_quat'][1][2]*Particles[pi]['coord_local'][2]
            
            Particles[pi]['pos_local'][2] = self.Data[i]['orientation_quat'][2][0]*Particles[pi]['coord_local'][0]+self.Data[i]['orientation_quat'][2][1]*Particles[pi]['coord_local'][1]+self.Data[i]['orientation_quat'][2][2]*Particles[pi]['coord_local'][2]
            
            
            Particles[pi]['pos'][0] = Particles[pi]['pos_local'][0] + self.Data[i]['pos'][0]
            Particles[pi]['pos'][1] = Particles[pi]['pos_local'][1] + self.Data[i]['pos'][1]
            Particles[pi]['pos'][2] = Particles[pi]['pos_local'][2] + self.Data[i]['pos'][2]
            
            if System.boundary_condition_id == 0:
                for dim in range(3):
                    if Particles[pi]['pos'][dim]>System.box_lengths[dim]/2:
                        Particles[pi]['pos'][dim] -= System.box_lengths[dim]
                    elif Particles[pi]['pos'][dim]<-System.box_lengths[dim]/2:
                        Particles[pi]['pos'][dim] += System.box_lengths[dim]
                        
                        
    def place_particles(self, pos, types, radii, System, Particles, i, h_membrane):
        
        """Places all the particles that make up the topology of rigid bead molecule i given a list of partcile types, positions and radii.
        
        Parameters
        ----------
        pos : `float64[:,3]`
            Array of particle positions
        types : `array_like`
            Array of strings of particle type names
        radii : `float64[:]`
            Array of particle radii
        System : `object`
            Instance of System class
        Particles : `object`
            Instance of Particles class        
        i : `int64`
            Molecule index
        h_membrane : `float64`
            Additive factor by which the particle positions are shifted along the z axis of the local coordinate frame. Thereby, the relative position to the center of diffusion is shifted. 'h_membrane' should only be used to change the position at which a surface molecule sits in the mesh surface.
        
        """
        
        i+=1
        
        n = len(pos)
        for k in range(n):
            
            Particles.add_particle(System, str(types[k]))
            index = Particles.slot-1

            Particles[index]['h'] = System.particle_types[str(types[k])][0]['h']
            Particles[index]['radius'] = radii[k]
            
            if self.Data[i]['loc_id'] == 0:
                Particles.set_coord(index, pos[k][0],pos[k][1],pos[k][2])
            else:
                Particles.set_coord(index, pos[k][0],pos[k][1],pos[k][2]+h_membrane)
                
            Particles[index]['pos_local'][:] = np.dot(self.Data[i]['orientation_quat'], Particles[index]['coord_local'])
            Particles[index]['pos'][:] = Particles[index]['pos_local'] + self.Data[i]['pos']
            
            if System.boundary_condition_id == 0:
                for dim in range(3):
                    if Particles[index]['pos'][dim]>System.box_lengths[dim]/2:
                        Particles[index]['pos'][dim] -= System.box_lengths[dim]
                    elif Particles[index]['pos'][dim]<-System.box_lengths[dim]/2:
                        Particles[index]['pos'][dim] += System.box_lengths[dim]   
                    
            Particles.set_rb_id(index, self.Data[i]['id'])
            
            self.add_particle(index, i-1)
            
            System.Np += 1
            
            
    def set_pos(self, Particles, System, pos, i):
        
        """Sets the position of rigid bead molecule i.
        
        Parameters
        ----------
        Particles : `object`
            Instance of Particles class  
        System : `object`
            Instance of System class
        pos : `float64[3]`
            Position of the molecule center
        i : `int64`
            Molecule index
        
        
        """
        
        i += 1
        
        self.Data[i]['pos'][:] = pos
        
        if System.boundary_condition_id == 0:
            for dim in range(3):
                if self.Data[i]['pos'][dim]>System.box_lengths[dim]/2:
                    self.Data[i]['pos'][dim]-=System.box_lengths[dim]
                elif self.Data[i]['pos'][dim]<-System.box_lengths[dim]/2:
                    self.Data[i]['pos'][dim]+=System.box_lengths[dim]

        for pi0 in range(self.Data[i]['topology_N']):
            pi = self.Data[i]['topology'][pi0]

            Particles.increase_pos(pi, pos[0], pos[1], pos[2]) 
            
            if System.boundary_condition_id == 0:
                for dim in range(3):
                    if Particles[pi]['pos'][dim]>System.box_lengths[dim]/2:
                        Particles[pi]['pos'][dim] -= System.box_lengths[dim]
                    elif Particles[pi]['pos'][dim]<-System.box_lengths[dim]/2:
                        Particles[pi]['pos'][dim] += System.box_lengths[dim]
                    
                    
    def update_force_torque(self, Particles, i):
        
        """Updates the total force and torque acting on rigid bead molecule i by summing over all forces and torques of its particles.
        
        Parameters
        ----------
        Particles : `object`
            Instance of Particles class  
        i : `int64`
            Molecule index
        
        
        """
        
        i+=1
        
        self.Data[i]['force'][:] = 0.0
        self.Data[i]['torque'][:] = 0.0
        for pi0 in range(self.Data[i]['topology_N']):
            pi = self.Data[i]['topology'][pi0]
            self.Data[i]['force'][0] += Particles[pi]['force'][0]
            self.Data[i]['force'][1] += Particles[pi]['force'][1]
            self.Data[i]['force'][2] += Particles[pi]['force'][2]
            
            self.Data[i]['torque'][0]+=Particles[pi]['pos_local'][1]*Particles[pi]['force'][2]-Particles[pi]['pos_local'][2]*Particles[pi]['force'][1]
            self.Data[i]['torque'][1]+=Particles[pi]['pos_local'][2]*Particles[pi]['force'][0]-Particles[pi]['pos_local'][0]*Particles[pi]['force'][2]      
            self.Data[i]['torque'][2]+=Particles[pi]['pos_local'][0]*Particles[pi]['force'][1]-Particles[pi]['pos_local'][1]*Particles[pi]['force'][0]
            
            Particles.clear_force(pi)
            Particles[pi]['number_reactions'] = 0
        
        
    def update_B(self, i):
        
        """Calculates matrix B used in the quaternion propagator (see RBs.dq()).
        
        Parameters
        ----------
        i : `int64`
            Molecule index.

        
        Notes
        -----
        When propagating the cartesian coordinates of the rigid bead moelcule's center of diffusion we need to account for the orientation of the moelcule. This is readily done by applying the rotation matrix A to the molecule's mobility tensor (Note: to rotate a tensor we need to apply the rotation matrix twice: :math:`\\mu^{lab} = A \cdot \\mu^{local} \cdot A^{-1}`). However, things are more complicated when deriving the propagator for the rotation quaternion. Here, the rotation matrix is partially replaced by matrix :math:`\\boldsymbol{B}` which accounts for the fact that we are using a four dimensional coordinate set of quaternions instead of three euler angles. In other terms, :math:`\\boldsymbol{B}` relates the angular velocity in the local frame to the quaternion velocity ( :math:`\\frac{d\\boldsymbol{q}}{dt} = \\boldsymbol{B} \\boldsymbol{\\omega}^{local}` ) :cite:p:`Ilie2015`.
        The matrix :math:`\\boldsymbol{B}` is given by:
            
        .. math::
            :label: B-matrix
            
            \\boldsymbol{B}
            = \\frac{1}{2}
            \\begin{pmatrix}
            q_0 & -q_1 & -q_2 & -q_3 \\\\
            q_1 & q_0 & -q_3 & q_2 \\\\
            q_2 & q_3 & q_0 & -q_1 \\\\
            q_3 & -q_2 & q_1 & q_0 \\\\
            \\end{pmatrix},
            
        where :math:`\\boldsymbol{q}` is the current rotational quaternion of molecule i.
        
        """
        
        i += 1
        
        self.Data[i]['B'][0][0] = self.Data[i]['q'][0]/2
        self.Data[i]['B'][0][1] = -self.Data[i]['q'][1]/2
        self.Data[i]['B'][0][2] = -self.Data[i]['q'][2]/2
        self.Data[i]['B'][0][3] = -self.Data[i]['q'][3]/2
        
        self.Data[i]['B'][1][0] = self.Data[i]['q'][1]/2
        self.Data[i]['B'][1][1] = self.Data[i]['q'][0]/2
        self.Data[i]['B'][1][2] = -self.Data[i]['q'][3]/2
        self.Data[i]['B'][1][3] = self.Data[i]['q'][2]/2
        
        self.Data[i]['B'][2][0] = self.Data[i]['q'][2]/2
        self.Data[i]['B'][2][1] = self.Data[i]['q'][3]/2
        self.Data[i]['B'][2][2] = self.Data[i]['q'][0]/2
        self.Data[i]['B'][2][3] = -self.Data[i]['q'][1]/2
        
        self.Data[i]['B'][3][0] = self.Data[i]['q'][3]/2
        self.Data[i]['B'][3][1] = -self.Data[i]['q'][2]/2
        self.Data[i]['B'][3][2] = self.Data[i]['q'][1]/2
        self.Data[i]['B'][3][3] = self.Data[i]['q'][0]/2            
        
    
    def calc_orientation_quat(self, i):
        
        """Calculates the orientation/rotation matrix of moelcule i from it's rotational quaternion q.
        
        Parameters
        ----------
        i : `int64`
            Molecule index
        
        
        Notes
        -----
        
        The rotation matrix corresponding to a unit rotation quaternion :math:`\\boldsymbol{q} = [q_0, q_1, q_2, q_3]` can be calculated by:
        
        .. math::
            :label: RotationMatrix
            
            \\boldsymbol{A}
            = 
            \\begin{pmatrix}
            2(q_0^2+q_1^2)-1 & 2(q_1 q_2-q_0 q_3) & 2(q_1 q_3-q_0 q_2) \\\\
            2(q_1 q_2-q_0 q_3) & 2(q_0^2+q_2^2)-1 & 2(q_2 q_3-q_0 q_1) \\\\
            2(q_1 q_3-q_0 q_2) & 2(q_2 q_3-q_0 q_1) & 2(q_0^2+q_3^2)-1 \\\\
            \\end{pmatrix}.  
            
        (Note: There exist several equivalent ways to calculate the rotation matrix, which share the way the off-diagonal elements are calculated but slightly differ in the diagonal elements)
        
        """
            
        i+=1
    
        self.Data[i]['orientation_quat'][0][0] = 2*(self.Data[i]['q'][0]**2+self.Data[i]['q'][1]**2)-1
        self.Data[i]['orientation_quat'][0][1] = 2*(self.Data[i]['q'][1]*self.Data[i]['q'][2]-self.Data[i]['q'][0]*self.Data[i]['q'][3])
        self.Data[i]['orientation_quat'][0][2] = 2*(self.Data[i]['q'][1]*self.Data[i]['q'][3]+self.Data[i]['q'][0]*self.Data[i]['q'][2])
        
        self.Data[i]['orientation_quat'][1][0] = 2*(self.Data[i]['q'][1]*self.Data[i]['q'][2]+self.Data[i]['q'][0]*self.Data[i]['q'][3])
        self.Data[i]['orientation_quat'][1][1] = 2*(self.Data[i]['q'][0]**2+self.Data[i]['q'][2]**2)-1
        self.Data[i]['orientation_quat'][1][2] = 2*(self.Data[i]['q'][2]*self.Data[i]['q'][3]-self.Data[i]['q'][0]*self.Data[i]['q'][1])
        
        self.Data[i]['orientation_quat'][2][0] = 2*(self.Data[i]['q'][1]*self.Data[i]['q'][3]-self.Data[i]['q'][0]*self.Data[i]['q'][2])
        self.Data[i]['orientation_quat'][2][1] = 2*(self.Data[i]['q'][2]*self.Data[i]['q'][3]+self.Data[i]['q'][0]*self.Data[i]['q'][1])
        self.Data[i]['orientation_quat'][2][2] = 2*(self.Data[i]['q'][0]**2+self.Data[i]['q'][3]**2)-1
            
        
    def set_orientation_quat(self, Particles, q, System, i):
        
        """Sets the orientation of rigid bead molecule i given a rotation quaternion q.
        
        Parameters
        ----------
        Particles : `object`
            Instance of Particles class  
        q : `float64[4]`
            Rotation quanternion
        System : `object`
            Instance of System class
        i : `int64`
            Molecule index
        
        
        """
            
        self.Data[i+1]['q'][:] = q
        self.update_B(i)
        self.calc_orientation_quat(i)
        
        self.update_topology(System, Particles, i)
                        
            

    def update_orientation_quat(self, i):
        
        """Updates the rotation matrix of rigid bead molecule i, however, without updating the molecule and particle positions.
        
        Parameters
        ----------
        i : `int64`
            Molecule index
        
        """
        
        self.update_B(i)
        
        self.calc_orientation_quat(i)
        
    
    def update_dq(self, System, i):
        
        """Calculates the rotational displacemnt of molecule i using the Brownian dynamics algorithm for anisotropic particles introduced by :cite:t:`Ilie2015`.
        
        Parameters
        ----------
        System : `obj`
            Instance of System class.
        i : `int`
            Index of the molecule.
        
        Notes
        -----
        The rotational displacement of an anisotropic particles due to Brownian motion and an external torque :math:`\\boldsymbol{T}` can be expressed by :cite:p:`Ilie2015`:
            
        .. math::
            :label: dq
            
            \\begin{align*}
            q_{a}(t+\\Delta t) - q_{a}(t) = & B_{a \\alpha} \\mu_{\\alpha \\beta}^{rb} A_{\\gamma \\beta} T_{\\gamma} \\Delta t \\\\
            & + B_{a \\alpha}(\\sqrt{\\boldsymbol{\\mu}^{rb}})_{\\alpha \\beta} \\Theta_{\\beta}^q \\sqrt{2 k_B T \\Delta t} + \\lambda_q q_a,
            \\end{align*}
        
        where :math:`\\boldsymbol{A}` is the rotation matrix and :math:`\\boldsymbol{\mu}^{rb}` the rotational mobility tensor of the molecule in the body fixed frame (as opposed to the lab frame of reference). :math:`\\boldsymbol{\\Theta}^q` is a normal distributed random vector describing the random rotational movement of the molecule due to collisions with the fluid molecules. :math:`\\boldsymbol{\mu}^{rb}` should be a real, positive semidefinite matrix to have proper physical meaning :cite:p:`Niethammer2006`. In this case, there exists a unique square root of the matrix :math:`\\sqrt{\\boldsymbol{\\mu}^{rb}}`, which can be found via diagonalization (see geometry_util.Transform:Util.sqrtM()). Latin indices run from 0 to 3 and Greek indices run from 1 to 3.
        
        The matrix :math:`\\boldsymbol{B}` is given by:
            
        .. math::
            :label: B-matrix_0
            
            \\boldsymbol{B}
            = \\frac{1}{2q^4}
            \\begin{pmatrix}
            q_0 & -q_1 & -q_2 & -q_3 \\\\
            q_1 & q_0 & -q_3 & q_2 \\\\
            q_2 & q_3 & q_0 & -q_1 \\\\
            q_3 & -q_2 & q_1 & q_0 \\\\
            \\end{pmatrix}.         
            
        For infinitesimal time steps :math:`\\boldsymbol{q}` preserves its unit length. However, for finite time step :math:`\\Delta t` an additional constraint needs to be added to ensure that :math:`\\boldsymbol{q}` keeps its unit length. This is realized here by a constraint force in the last term, directed along :math:`\\boldsymbol{q}`. By solving the Langrange multiplier :math:`\lambda_q` from the condition :math:`q(t+\Delta t) = 1` we get the strength of the force. Denoting :math:`\\tilde{q}(t+ \\Delta t)` as the uncopnstrained quaternion, :math:`\lambda_q` is obtained from solving the quadratic equation
        
        .. math::
            :label: LagrangeMult
            
            \\lambda_q^2 + 2 \\lambda_q \\boldsymbol{q} \cdot \\tilde{q}(t+ \\Delta t)+ \\tilde{q}^2(t+ \\Delta t) = 1
            
        Simple rescaling of the quaternion will change the sampled phase space distribution. However, in practice, the resulting error is marginal and since rescaling is fatser than solving the quadratic equation each time step, we currently do the rescaling. However, the more accurate method is mentioned here for compeleness.
        
        """
        
        i+=1
        
        dq = self.Data[i]['dq']
        B = self.Data[i]['B']
        mu_rb = self.Data[i]['mu_rb']
        mu_rb_sqrt = self.Data[i]['mu_rb_sqrt']
        torque = self.Data[i]['torque']
        Theta_r = self.Data[i]['Theta_r']
        q = self.Data[i]['q']
        orientation_quat = self.Data[i]['orientation_quat']
        
        if self.Data[i]['loc_id'] == 0:
            ai = 1
        else: # Surface molecule
            ai = 3
        

        dq[:] = 0.0
        # Theta_r = np.random.normal(loc=0.0, scale=1.0, size = 3)
        Theta_r[0] = np.random.normal(loc=0.0, scale=1.0)
        Theta_r[1] = np.random.normal(loc=0.0, scale=1.0)
        Theta_r[2] = np.random.normal(loc=0.0, scale=1.0)
        for a in range(4):
            for alpha in range(ai,4):
                for beta in range(ai,4):
                    for gamma in range(1,4): 
                        
                        dq[a] += B[a][alpha]*mu_rb[alpha-1][beta-1]*orientation_quat[gamma-1][beta-1]*torque[gamma-1]*System.dt
                    
                    #TODO: If mu_rb is a diagonal matrix, we could shorten this loop:
                    #(9 - > 3 iterations).
                    dq[a] += B[a][alpha]*mu_rb_sqrt[alpha-1][beta-1]*Theta_r[beta-1]*np.sqrt(2*System.kbt*System.dt)
                    
        # q[:] += dq
        q[0] += dq[0]
        q[1] += dq[1]
        q[2] += dq[2]
        q[3] += dq[3]
        
        # TODO: renormalizing the quaternion actually introduces a bias. Better use Lagrange multiplier to apply restriction!
        # In practice, however, the bias seems to be fairly low!
        norm = np.linalg.norm(q)
        q[0] /= norm  
        q[1] /= norm  
        q[2] /= norm  
        q[3] /= norm  
        
        # print('dq:', self.dq)
        # print('T:', torque)
            


    def update_dX(self, System, i):
        
        """Calculates the translational displacemnt of molecule i using the Brownian dynamics algorithm for anisotropic particles introduced by :cite:t:`Ilie2015`.
        
        Parameters
        ----------
        System : `obj`
            Instance of System class.
        i : `int`
            Index of the molecule.
        
        Notes
        -----
        The translational displacement of an anisotropic particles due to Brownian motion and an external force :math:`\\boldsymbol{F}` can be expressed by :cite:p:`Ilie2015`:
            
        .. math::
            :label: dX
            
            \\begin{align*}
            R_{\\alpha}(t+\\Delta t) - R_{\\alpha}(t) = & A_{\\alpha \\gamma} \\mu_{\\gamma \\delta}^{tb} A_{\\beta \\delta} F_{\\beta} \\Delta t \\\\
            & + A_{\\alpha \\gamma}(\\sqrt{\\boldsymbol{\\mu}^{tb}})_{\\gamma \\beta} \\Theta_{\\beta}^t \\sqrt{2 k_B T \\Delta t},
            \\end{align*}
            
        where :math:`\\boldsymbol{A}` is the rotation matrix and :math:`\\boldsymbol{\mu}^{tb}` the translational mobility tensor of the molecule in the body fixed frame (as opposed to the lab frame of reference). :math:`\\boldsymbol{\\Theta}^t` is a normal distributed random vector describing the random movement of the molecule due to collisions with the fluid molecules. :math:`\\boldsymbol{\mu}^{tb}` should be a real, positive semidefinite matrix to have proper physical meaning :cite:p:`Niethammer2006` . In this case, there exists a unique square root of the matrix :math:`\\sqrt{\\boldsymbol{\\mu}^{tb}}`, which can be found via diagonalization (see geometry_util.Transform:Util.sqrtM()). Latin indices run from 0 to 3 and Greek indices run from 1 to 3.
        
        """
        
        i+=1
        
        dX = self.Data[i]['dX']
        mu_tb = self.Data[i]['mu_tb']
        mu_tb_sqrt = self.Data[i]['mu_tb_sqrt']
        force = self.Data[i]['force']
        Theta_t = self.Data[i]['Theta_t']
        pos = self.Data[i]['pos']
        orientation_quat = self.Data[i]['orientation_quat']
        
        
        if self.Data[i]['loc_id'] == 0:
            bi = 4
        else: # Surface molecule
            bi = 3
            
        dX[:] = 0.0
        # Theta_t = np.random.normal(loc=0.0, scale=1.0, size = 3)
        Theta_t[0] = np.random.normal(loc=0.0, scale=1.0)
        Theta_t[1] = np.random.normal(loc=0.0, scale=1.0)
        Theta_t[2] = np.random.normal(loc=0.0, scale=1.0)
        for alpha in range(3):
            for gamma in range(1,bi):
                for beta in range(1,4):
                    for delta in range(1,bi):
                    #TODO: If mu_tb is a diagonal, we could shorten this loop:
                    #(9 - > 3 iterations).
                        
                        dX[alpha] += orientation_quat[alpha][gamma-1]*mu_tb[gamma-1][delta-1]*orientation_quat[beta-1][delta-1]*force[beta-1]*System.dt
                    
                    dX[alpha] += orientation_quat[alpha][gamma-1]*mu_tb_sqrt[gamma-1][beta-1]*Theta_t[beta-1]*np.sqrt(2*System.kbt*System.dt)



        self.Data[i]['pos_last'][0] = pos[0]
        self.Data[i]['pos_last'][1] = pos[1]
        self.Data[i]['pos_last'][2] = pos[2]
        
        
                
    #%%
                
            
    def update_particle_pos(self, Particles, System, i):
        
        """Updates the position of volume molecule i. 
        
        Parameters
        ----------
        Particles : `object`
            Instance of Particles class  
        System : `obj`
            Instance of System class.
        i : `int`
            Index of the molecule.
        
        See Also
        --------
        :func:`~pyrid.geometry.intersections_util.ray_march_volume`

        """
        
        i+=1
            
        if self.Data[i]['collision_type'] == 0: # ALl collisions are handled during force update
            
            dX = self.Data[i]['dX']
            pos = self.Data[i]['pos']

            if System.boundary_condition_id == 0:
                
                pos[0] += dX[0]
                pos[1] += dX[1]
                pos[2] += dX[2]
                
                for dim in range(3):
                    if self.Data[i]['pos'][dim]>System.box_lengths[dim]/2:
                        self.Data[i]['pos'][dim]-=System.box_lengths[dim]
                    elif self.Data[i]['pos'][dim]<-System.box_lengths[dim]/2:
                        self.Data[i]['pos'][dim]+=System.box_lengths[dim]
                    
                self.update_topology(System, Particles, i-1)
                
            if System.boundary_condition_id == 1:
                 
                pos[0] += dX[0]
                pos[1] += dX[1]
                pos[2] += dX[2]

                self.update_topology(System, Particles, i-1)
                
            elif System.boundary_condition_id == 2:
                
                pos[0] += dX[0]
                pos[1] += dX[1]
                pos[2] += dX[2]
                
                crossed_border = False
                for dim in range(3):
                    if self.Data[i]['pos'][dim]>System.box_lengths[dim]/2:
                        crossed_border = True
                    elif self.Data[i]['pos'][dim]<-System.box_lengths[dim]/2:
                        crossed_border = True
                       
                if crossed_border:
                    delete_molecule(System, self, Particles, i-1)
                    return
                    
                self.update_topology(System, Particles, i-1)
                
    
        else:
            
            if System.mesh == True:
            
                # pos_init = np.copy(self.Data[i]['pos'])
                # dX_init = np.copy(self.Data[i]['dX'])
                # ---------------
                            
                # self.resolve_collisions(System, Particles, i-1)
                passed = ray_march_volume(self.Data[i]['pos'], self.Data[i]['dX'], System)
                
                if passed == False:
                    delete_molecule(System, self, Particles, i-1)
                else:
                    self.update_topology(System, Particles, i-1)
                
                # inside_System_before = isPointInside_fast(pos_init, 0, System)
                # inside_System_after = isPointInside_fast(self.Data[i]['pos'], 0, System)
                
                # if inside_System_before == False and inside_System_after == True and 0>self.Data[i]['pos'][2]>-System.box_lengths[2]/2:
                #     print(i, pos_init, dX_init)
                #     raise ValueError('crossed triangle boundary!')
                
            else:
                
                dX = self.Data[i]['dX']
                pos = self.Data[i]['pos']
    
                if System.boundary_condition_id == 0:
                    
                    pos[0] += dX[0]
                    pos[1] += dX[1]
                    pos[2] += dX[2]
                    
                    for dim in range(3):
                        if self.Data[i]['pos'][dim]>System.box_lengths[dim]/2:
                            self.Data[i]['pos'][dim]-=System.box_lengths[dim]
                        elif self.Data[i]['pos'][dim]<-System.box_lengths[dim]/2:
                            self.Data[i]['pos'][dim]+=System.box_lengths[dim]
                        
                    self.update_topology(System, Particles, i-1)
                    
                    
                if System.boundary_condition_id == 1:
                     
                    for dim in range(3):
                        if pos[dim]+dX[dim]>System.box_lengths[dim]/2:
                            
                            boundary = System.box_lengths[dim]/2
                            t = (boundary - pos[dim])/dX[dim]
                            self.Data[i]['pos'][:] += t*dX
                            self.Data[i]['dX'][:] -= t*dX
                            self.Data[i]['dX'][0] *= -1
                            
                        elif pos[dim]+dX[dim]<-System.box_lengths[dim]/2:
                            
                            boundary = -System.box_lengths[dim]/2
                            t = (boundary - pos[dim])/dX[dim]
                            self.Data[i]['pos'][:] += t*dX
                            self.Data[i]['dX'][:] -= t*dX
                            self.Data[i]['dX'][0] *= -1
                    
                    pos[0] += dX[0]
                    pos[1] += dX[1]
                    pos[2] += dX[2]

                    self.update_topology(System, Particles, i-1)
                    
                elif System.boundary_condition_id == 2:
                    
                    pos[0] += dX[0]
                    pos[1] += dX[1]
                    pos[2] += dX[2]
                    
                    crossed_border = False
                    for dim in range(3):
                        if self.Data[i]['pos'][dim]>System.box_lengths[dim]/2:
                            crossed_border = True
                        elif self.Data[i]['pos'][dim]<-System.box_lengths[dim]/2:
                            crossed_border = True
                           
                    if crossed_border:
                        delete_molecule(System, self, Particles, i-1)
                        return
                        
                    self.update_topology(System, Particles, i-1)
                

                

            
                    
    #%%

    def update_particle_pos_2D(self, System, Particles, i):
        
        """Updates the position of surface molecule i. 
        
        Parameters
        ----------
        Particles : `object`
            Instance of Particles class  
        System : `obj`
            Instance of System class.
        i : `int`
            Index of the molecule.
        
        See Also
        --------
        :func:`~pyrid.geometry.intersections_util.ray_march_surface`
        
        """
        
        i+=1
           
        
        triangle_id = ray_march_surface(self.Data[i]['pos'], self.Data[i]['q'], self.Data[i]['dX'], self.Data[i]['triangle_id'], System, update_quat = True)
        
        if triangle_id==-2:
            print('i, step: ', i, System.current_step)
            raise ValueError('Unable to Raymarch on mesh surface!')
        
        if triangle_id >= 0:
            self.Data[i]['triangle_id'] = triangle_id
            self.update_orientation_quat(i-1)
            self.update_topology(System, Particles, i-1)  
        else:
            delete_molecule(System, self, Particles, i-1)
    
    #%%
    
    def rescale_pos(self, Particles, System, mu, i):
        
        """Rescales the position of molecule i by a multiplicative factor mu. This method is only used for the Berendsen barostat.
        
        Parameters
        ----------
        Particles : `object`
            Instance of Particles class  
        System : `obj`
            Instance of System class.
        mu : `float64`
            Multiplicative factor by which to scale the molecule position.
        i : `int`
            Index of the molecule.
        
        Raises
        ------
        NotImplementedError (just an example)
            Brief explanation of why/when this exception is raised
        
        Returns
        -------
        dtype
            Some information
        
        """
        
        i+=1
        
        self.Data[i]['pos'][0] *= mu
        self.Data[i]['pos'][1] *= mu
        self.Data[i]['pos'][2] *= mu
        
        for pi0 in range(self.Data[i]['topology_N']):
            pi = self.Data[i]['topology'][pi0]
            
            Particles[pi]['pos'][0] = Particles[pi]['pos_local'][0] + self.Data[i]['pos'][0]
            Particles[pi]['pos'][1] = Particles[pi]['pos_local'][1] + self.Data[i]['pos'][1]
            Particles[pi]['pos'][2] = Particles[pi]['pos_local'][2] + self.Data[i]['pos'][2]
                        
            if System.boundary_condition_id == 0:
                for dim in range(3):
                    if Particles[pi]['pos'][dim]>System.box_lengths[dim]/2:
                        Particles[pi]['pos'][dim] -= System.box_lengths[dim]
                    elif Particles[pi]['pos'][dim]<-System.box_lengths[dim]/2:
                        Particles[pi]['pos'][dim] += System.box_lengths[dim]
    
    




#%%

# if __name__=='__main__':
    
