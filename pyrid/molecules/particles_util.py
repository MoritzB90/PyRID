# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""


import numpy as np
import numba as nb
from numba.experimental import jitclass
# from contextlib import contextmanager
from ..data_structures.dynamic_array_util import DenseArray, HolesArray

    
#%%

    
item_t = np.dtype([('next', np.int64), ('pos', (np.float64, (3,))), ('pos_local', (np.float64, (3,))), ('coord_local', (np.float64, (3,))), ('force', (np.float64, (3,))), ('rb_id', np.int64), ('type', 'U20'), ('type_id', np.int64), ('radius', np.float64), ('number_reactions', np.int64), ('reactions_head', np.int64 , (2,)), ('bound', bool), ('bound_with', np.int64), ('cutoff', np.float64), ('h', np.int64), ('next_transition', np.float64),],  align=True)


spec_holes_array = [
    ('n', nb.int64),
    ('capacity', nb.int64),
    ('Data', nb.typeof(np.empty(1, dtype = item_t))),
    ('item_t', nb.typeof(item_t)),
    ('i', nb.int64),
    ('occupied', DenseArray.class_type.instance_type),
    ('slot', nb.int64),
]

spec_particles = [
    ('a', nb.int64),
]

@jitclass(spec_holes_array+spec_particles)
class Particles(HolesArray):
    
    """
    The Particles class stored all data for the particles in the simulation. Particles are the 'atoms' in PyRID. Molecules can be constructed from combining several particles to a rigid bead model. Particles can interact with each other via energy potentials and react with each other by user defined bimolecular reactions.
    
    Attributes
    ----------
    n : `int64`
        length of the particles array
    Data : `array_like`
        Numpy structured array containing all data that define a particle state.
        dtype: np.dtype([('next', np.int64), ('pos', (np.float64, (3,))), ('pos_local', (np.float64, (3,))), ('coord_local', (np.float64, (3,))), ('force', (np.float64, (3,))), ('rb_id', np.int64), ('type', 'U20'), ('type_id', np.int64), ('radius', np.float64), ('number_reactions', np.int64), ('reactions_head', np.int64 , (2,)), ('bound', bool), ('bound_with', np.int64), ('cutoff', np.float64), ('h', np.int64), ('next_transition', np.float64),],  align=True)
    
    Methods
    -------
    add_particle(System, type_name)
        Adds a new particle to the array
    next_up_reaction(System, i)
        Calculates the time point of the next uni-particle reaction.
    set_pos(k, x,y,z)
        Sets the position of particle i given x,y,z
    increase_pos(i, dx,dy,dz)
        Incerases the particle position by dx,dy,dz
    set_pos_local(i, xl,yl,zl)
        Sets the position of particle i in the local coordinate frame
    increase_pos_local(i, dxl,dyl,dzl)
        Increases the position of particle i in the local coordinate frame
    set_force(i, fx, fy, fz)
        Sets the force vector of particle i
    clear_force(i)
        Resets the force vector of particle i to [0,0,0]
    increase_force(i, dfx, dfy, dfz)
        Increases the force vector of particle i by dfx, dfy, dfz
    set_coord(i, xc,yc, zc)
        Sets the coordinates of particle i in the local reference frame
    set_rb_id(i, rb_id)
        Sets the id of the rigid bead molecule that particle i is part of
    set_type(i,  particle_type, System)
        Sets the type name and id of particle i
    increase_number_reactions(i)
        Increases the number of reactions particle i partakes in by 1
    decrease_number_reactions(i)
        Decreases the number of reactions particle i partakes in by 1
    clear_number_reactions(i)
        Sets the number of reactions particle i partakes in to zero
    
        
    """
    
    __init__HolesArray = HolesArray.__init__
    
    def __init__(self):
        self.__init__HolesArray(item_t)
        
    def add_particle(self, System, type_name):
        
        """Adds a new particle to the array
        
        Parameters
        ----------
        System : `object`
            Instance of System class
        type_name : `string`
            Name of particle type
        
        """
    
        self.insert(np.zeros(1, dtype = item_t)[0])
        self.Data[self.slot]['bound_with'] = -1
        
        self.set_type(self.slot-1,  type_name, System)
        
        self.next_up_reaction(System, self.slot-1)

    ############################################

    def next_up_reaction(self, System, i):
        
        """Calculates the time point of the next uni-particle reaction.
        
        Parameters
        ----------
        System : `object`
            Instance of System class
        i : `int64`
            Particle index
        
        
        """
        
        i+=1
        
        type_id = self.Data[i]['type_id']
        name = System.particle_id_to_name[type_id]
        
        if System.particle_types[str(name)][0]['UP_reaction']:
            
            total_rate = System.particle_types[str(name)][0]['transition_rate_total']
            
            self.Data[i]['next_transition'] = System.current_step*System.dt + 1/total_rate*np.log(1/np.random.rand())
        
        else:
            self.Data[i]['next_transition'] = 1e10
            
    def set_pos(self, i, x,y,z):

        """Sets the position of particle i given x,y,z
        
        Parameters
        ----------
        i : `int64`
            Particle index
        x : `float64`
            Particle position x value
        y : `float64`
            Particle position y value
        z : `float64`
            Particle position z value
        
        """

        i+=1
        
        self.Data[i]['pos'][0] = x
        self.Data[i]['pos'][1] = y
        self.Data[i]['pos'][2] = z
        
    def increase_pos(self, i, dx,dy,dz):

        """Incerases the particle position by dx,dy,dz
        
        Parameters
        ----------
        i : `int64`
            Particle index
        dx : `float64`
            Particle position differental along x axis
        dy : `float64`
            Particle position differental along y axis
        dz : `float64`
            Particle position differental along z axis
        
        """

        i+=1
        
        self.Data[i]['pos'][0] += dx
        self.Data[i]['pos'][1] += dy
        self.Data[i]['pos'][2] += dz    
        
    ############################################
    def set_pos_local(self, i, xl,yl,zl):

        """Sets the position of particle i in the local coordinate frame
        
        Parameters
        ----------
        i : `int64`
            Particle index
        xl : `float64`
            Particle x position in local frame
        yl : `float64`
            Particle y position in local frame
        zl : `float64`
            Particle z position in local frame
        
        """
        
        i+=1
        
        self.Data[i]['pos_local'][0] = xl
        self.Data[i]['pos_local'][1] = yl
        self.Data[i]['pos_local'][2] = zl
        
        
    def increase_pos_local(self, i, dxl,dyl,dzl):

        """Increases the position of particle i in the local coordinate frame
        
        Parameters
        ----------
        i : `int64`
            Particle index
        dxl : `float64`
            Particle position differental along x axis in local frame
        dyl : `float64`
            Particle position differental along y axis in local frame
        dzl : `float64`
            Particle position differental along z axis in local frame
        
        """
        
        i+=1
        
        self.Data[i]['pos_local'][0] += dxl
        self.Data[i]['pos_local'][1] += dyl
        self.Data[i]['pos_local'][2] += dzl
        
        
    ############################################
        

    def set_force(self, i, fx, fy, fz):
        
        """Sets the force vector of particle i
        
        Parameters
        ----------
        i : `int64`
            Particle index
        fx : `float64`
            Particle force along x axis
        fy : `float64`
            Particle force along y axis
        fz : `float64`
            Particle force along z axis
        
        """
        
        i+=1
        
        self.Data[i]['force'][0] = fx
        self.Data[i]['force'][1] = fy
        self.Data[i]['force'][2] = fz
        
        
    def clear_force(self, i):

        """Resets the force vector of particle i to [0,0,0]
        
        Parameters
        ----------
        i : `int64`
            Particle index
        
        """

        i+=1
        
        self.Data[i]['force'][:] = 0.0

    def increase_force(self, i, dfx, dfy, dfz):

        """Increases the force vector of particle i by dfx, dfy, dfz
        
        Parameters
        ----------
        i : `int64`
            Particle index
        dfx : `float64`
            Particle force differental along x axis
        dfy : `float64`
            Particle force differental along y axis
        dfz : `float64`
            Particle force differental along z axis
        
        """
        
        i+=1
        
        self.Data[i]['force'][0] += dfx
        self.Data[i]['force'][1] += dfy
        self.Data[i]['force'][2] += dfz
        
    ##############################################
    
    def set_coord(self, i, xc,yc, zc):

        """Sets the coordinates of particle i in the local reference frame
        
        Parameters
        ----------
        i : `int64`
            Particle index
        xc : `float64`
            Particle x coordinate in local frame
        yc : `float64`
            Particle y coordinate in local frame
        zc : `float64`
            Particle z coordinate in local frame
        
        """
        
        i+=1
        
        self.Data[i]['coord_local'][0] = xc
        self.Data[i]['coord_local'][1] = yc
        self.Data[i]['coord_local'][2] = zc
    
    def set_rb_id(self, i, rb_id):

        """Sets the id of the rigid bead molecule that particle i is part of
        
        Parameters
        ----------
        i : `int64`
            Particle index
        rb:id : `int64`
            Type index of the rigid bead molecule
        
        
        """
        
        i+=1
        
        self.Data[i]['rb_id'] = rb_id
        
    def set_type(self, i,  particle_type, System):

        """Sets the type name and id of particle i
        
        Parameters
        ----------
        i : `int64`
            Particle index
        particle_type : `string`
            Name of the particle type
        System : `object`
            Instance of System class
        
        
        """
        
        i+=1
        
        type_id = System.particle_types[particle_type][0]['id']
        
        self.Data[i]['type'] = particle_type
        self.Data[i]['type_id'] = type_id
    
    #%%
    
    def increase_number_reactions(self, i):

        """Increases the number of reactions particle i partakes in by 1
        
        Parameters
        ----------
        i : `int64`
            Particle index
        
        """
        
        i+=1
        
        self.Data[i]['number_reactions'] += 1

    def decrease_number_reactions(self, i):

        """Decreases the number of reactions particle i partakes in by 1
        
        Parameters
        ----------
        i : `int64`
            Particle index
        
        """
        
        i+=1
        
        self.Data[i]['number_reactions'] -= 1
        
    def clear_number_reactions(self, i):

        """Sets the number of reactions particle i partakes in to zero
        
        Parameters
        ----------
        i : `int64`
            Particle index
        
        """
        
        i+=1
        
        self.Data[i]['number_reactions'] = 0
        


#%%

# if __name__=='__main__':
    
