# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numpy as np
import numba as nb
import os

#%%

def read_xyz(file_path_pos, file_path_id = None):
    
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
    
    global start, end, Types
    
    lines = filter(None, (line.rstrip().replace('\t', ' ').replace('\n', ' ').split() for line in open(file_path_pos, 'r')))
    lines = list(lines)

    if file_path_id is not None:
        item_t = np.dtype([('atom', 'U24'), ('x', np.float64), ('y', np.float64), ('z', np.float64), ('id', np.int64), ('rb_id', np.int64)],  align=True)
    else:
        item_t = np.dtype([('atom', 'U24'), ('x', np.float64), ('y', np.float64), ('z', np.float64)],  align=True)
    

    Types_Set = set()
    
    array  = None
    Data = []
    i=0
    for c,l in enumerate(lines):
        if len(l)==1:
            i = -1
            if array is not None:
                Data.append(array)
            array = np.zeros(int(l[0]), dtype = item_t)
        else:
            for j,value in enumerate(l):
                array[i][j] = value
                Types_Set.add(array['atom'][i])
        
        # if i%100000==0:
        #     print('lines read (pos):', i, '/', len(lines))
            
        i+=1
        
        if c==len(lines)-1:
            Data.append(array)
        
    #---------
    
    if file_path_id is not None:
        lines = filter(None, (line.rstrip().replace('\t', ' ').replace('\n', ' ').split() for line in open(file_path_id, 'r')))
        lines = list(lines)
    
        i = 0
        k = -1
        for l in lines:
            if len(l)==1:
                i = -1
                k+=1
                # print(k, len(Data))
            else:
                for j,value in enumerate(l):
                    # print(value)
                    Data[k][i][4+j] = value
            
            # if i%100000==0:
            #     print('lines read (id):', i, '/', len(lines))
                
            i+=1
        
        
    Types = list(Types_Set)
    Types.sort()
    Types.reverse()
    
    print(Types)
    
    return Data, Types


#%%

item_t_pos = np.dtype([('type', 'U20'),('x', np.float64),('y', np.float64),('z', np.float64)],  align=True)

@nb.njit
def pos_data(Particles, System):
    
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
    
    # data = np.zeros(System.Np, dtype=item_t)
    # data = np.zeros(len(Particles.occupied), dtype=item_t_pos)
    data = np.zeros(Particles.occupied.n, dtype=item_t_pos) # Note: Numba does not support structured arrays to be intialized to anything but zeros (no empty, no ones)!
    
    # for i,pi in enumerate(Particles.occupied):
    for i in range(Particles.occupied.n):
        pi = Particles.occupied[i]
        
        if Particles[pi]['bound']==True:
            data[i]['type'] = Particles[pi]['type']+'_True'
        else:
            data[i]['type'] = Particles[pi]['type']+'_False'
        data[i]['x'] = Particles[pi]['pos'][0]
        data[i]['y'] = Particles[pi]['pos'][1]
        data[i]['z'] = Particles[pi]['pos'][2]
        
    return data

#%%

item_t_ID = np.dtype([('id', np.int64), ('rb_id', np.int64)],  align=True)

@nb.njit
def data_id(Particles, System):
    
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
    
    # data = np.zeros(System.Np, dtype=item_t)
    # data = np.zeros(len(Particles.occupied), dtype=item_t_ID)
    data = np.zeros(Particles.occupied.n, dtype=item_t_ID)
    
    # for i,pi in enumerate(Particles.occupied):
    for i in range(Particles.occupied.n):
        pi = Particles.occupied[i]
        
        data[i]['id'] = pi
        data[i]['rb_id'] = Particles[pi]['rb_id']
        
    return data


#%%

def write(j, Simulation, System, Particles):
    
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
    
    if j%Simulation.stride==0:
        if Simulation.write_trajectory==True:    
    
            open(Simulation.file_path / (Simulation.file_name+'.xyz'), "a").write(str(Particles.occupied.n)+'\n\n')
            with open(Simulation.file_path / (Simulation.file_name+'.xyz'), "a") as f:
                np.savetxt(f, pos_data(Particles, System), fmt = ('%s','%f','%f','%f'), delimiter='\t')
                
                
            open(Simulation.file_path / (Simulation.file_name+'.pid'), "a").write(str(Particles.occupied.n)+'\n\n')
            with open(Simulation.file_path / (Simulation.file_name+'.pid'), "a") as f:
                np.savetxt(f, data_id(Particles, System), fmt = ('%d'))            
    
    
            if System.barostat[0]['active'] == True:
                open(Simulation.file_path / (Simulation.file_name+'.box'), "a").write('{0} {1} {2}\n'.format(System.box_lengths[0],System.box_lengths[1],System.box_lengths[2]))
                
#%%


def write_init(Simulation, System):
    
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
    
    if Simulation.write_trajectory==True:
        
        try:
            os.makedirs(Simulation.file_path) 
        except FileExistsError:
            # directory already exists
            pass
        
        Radii = {}#{'Patch_2_True_shape': 0.01, 'Patch_2_False_shape': 0.01, 'Patch_1_True_shape': 0.01, 'Patch_1_False_shape': 0.01, 'Core_1_False_shape': 0.015}
        Radii_interaction = {}
        
        for ptype in System.particle_types:
            for bound in [True, False]:
                Radii[ptype+'_'+str(bound)+'_shape'] = System.particle_types[ptype][0]['radius']
                Radii_interaction[ptype+'_'+str(bound)+'_shape'] = System.particle_types[ptype][0]['cutoff']
                
        f = open(Simulation.file_path / (Simulation.file_name+'.prop'), "w")
        f.write('Radii = {}\n'.format(Radii))
        f.write('Radii_interaction = {}\n'.format(Radii_interaction))
        f.write('kB = {}\n'.format(System.kB))
        f.write('Temp = {}\n'.format(System.Temp))
        f.write('eta = {}\n'.format(System.eta))
        f.write('dt = {}\n'.format(System.dt))
        f.write('mesh_scale = {}\n'.format(System.mesh_scale))
        f.write('box_lengths = np.array([{0},{1},{2}])\n'.format(System.box_lengths[0],System.box_lengths[1],System.box_lengths[2]))
        f.close()
        
    if Simulation.write_trajectory==True:
        
        f = open(Simulation.file_path / (Simulation.file_name+'.xyz.tcl'), "w")
        f.write("""
        mol delete top
        """
        'mol load xyz '+Simulation.file_name+'.xyz'
        """
        mol delrep 0 top
        display resetview
        """)
        
        for ptype in System.particle_types:
            for bound in [True, False]:
                f.write("""
                mol representation VDW {0} 16.0
                mol selection name {1}
                mol material Opaque
                mol color ColorID {2}
                mol addrep top
                """.format(System.particle_types[ptype][0]['radius']/2, ptype+'_'+str(bound), System.particle_types[ptype][0]['id']))
            
        f.write("""
        animate goto 0
        color Display Background white
        molinfo top set {center_matrix} {{{1 0 0 0}{0 1 0 0}{0 0 1 0}{0 0 0 1}}}
        """
        'set cell [pbc set {'+'{} {} {}'.format(System.box_lengths[0],System.box_lengths[1],System.box_lengths[2])+'} -all]'
        """
        pbc box -center origin -color black -width 1
        """
        )
        f.close()
        
#%%

    if Simulation.write_trajectory==True:
        f = open(Simulation.file_path / (Simulation.file_name+'.xyz'), "w")
        f.close()
        f = open(Simulation.file_path / (Simulation.file_name+'.pid'), "w")
        f.close()
        f = open(Simulation.file_path / (Simulation.file_name+'.box'), "w")
        f.close()