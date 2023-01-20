# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numpy as np
import h5py
import os
import math
from pathlib import Path

from ..evaluation.rdf_util import create_rb_hgrid, radial_distr_function

#%%


class Observables(object):
    
    """
    The Observables class keeps track of all the observables and writes these to an hdf5 file.
    
    Attributes
    ----------
    binned : `dict`
        Some Information
    attribute_2 : dtype
        Some Information
    
    Methods
    -------
    method_1(arguments)
        Some general information about this method
    
    """
    
    def __init__(self, Simulation):
        
        try:
            os.makedirs(Simulation.file_path / 'hdf5') 
        except FileExistsError:
            # directory already exists
            pass
 
        try:
            os.makedirs(Simulation.fig_path) 
        except FileExistsError:
            # directory already exists
            pass
        
        # -------------
        # Binning Data
        # -------------
        
        self.binned = {}
        self.binned['Volume'] = 0.0
        self.binned['Energy'] = np.zeros(Simulation.System.Epot.shape)
        self.binned['Bonds'] = np.zeros(Simulation.System.N_bonds.shape)
        self.binned['Number'] = np.zeros(Simulation.System.Nmol.shape)
        self.binned['Pressure'] = np.zeros(Simulation.System.Pressure.shape)
        self.binned['Virial'] = np.zeros(Simulation.System.virial_scalar.shape)
        self.binned['Virial Tensor'] = np.zeros(Simulation.System.virial_tensor.shape)
        
        observables_types = np.dtype([('Volume', bool),('Energy', bool),('Number', bool),('Pressure', bool),('Virial', bool),('Virial Tensor', bool),('Reactions', bool),('Bonds', bool),('Force', bool),('Torque', bool),('Position', bool),('Orientation', bool),],  align=True)

        self.Observing = np.zeros(2, dtype = observables_types)
        
        # -------------
        
        self.Time_passed = []
        
        self.observables_setup = dict()
        self.observing_rdf = False

        hdf = h5py.File(Simulation.file_path / 'hdf5' / (Simulation.file_name+'.h5'), 'w')
        
        #------------------
        # System Setup
        #------------------
        
        hdf.create_group('Setup/')
        # hdf['Setup'].attrs['box_lengths'] = Simulation.System.box_lengths
        # hdf['Setup'].attrs['Temp'] = Simulation.System.Temp
        # hdf['Setup'].attrs['eta'] = Simulation.System.eta
        # hdf['Setup'].attrs['dt'] = Simulation.System.dt
        
        hdf['Setup'].create_dataset('box_lengths', data=Simulation.System.box_lengths)
        hdf['Setup'].create_dataset('Temp', data=Simulation.System.Temp)
        hdf['Setup'].create_dataset('kbt', data=Simulation.System.kbt)
        hdf['Setup'].create_dataset('eta', data=Simulation.System.eta)
        hdf['Setup'].create_dataset('dt', data=Simulation.System.dt)
        hdf['Setup'].create_dataset('nsteps', data=Simulation.nsteps)
        if Simulation.System.barostat[0]['active']==True:
            hdf.create_group('Setup/barostat')
            hdf['Setup/barostat'].create_dataset('Tau_P', data=Simulation.System.barostat[0]['Tau_P'])
            hdf['Setup/barostat'].create_dataset('P0', data=Simulation.System.barostat[0]['P0'])
            
        hdf.create_group('Setup/units')
        for measure in Simulation.units:
            hdf['Setup/units'].create_dataset(measure, data=Simulation.units[measure])
        
        #------------------
        # Molecule data
        #------------------
        
        # hdf.create_group('Molecules/')
        
        if len(Simulation.System.molecule_types)==0:
            raise AttributeError('No molecules defined!')
        else:
            for type1 in Simulation.System.molecule_types:
                    
                hdf.create_group('Molecules/'+str(type1))        
                
                hdf['Molecules/'+str(type1)].create_dataset('types', data=np.array(Simulation.System.molecule_types[str(type1)].types, dtype = 'S'))
                hdf['Molecules/'+str(type1)].create_dataset('radii', data=Simulation.System.molecule_types[type1].radii)
                hdf['Molecules/'+str(type1)].create_dataset('pos', data=Simulation.System.molecule_types[type1].pos)
                hdf['Molecules/'+str(type1)].create_dataset('volume', data=Simulation.System.molecule_types[type1].volume)
                hdf['Molecules/'+str(type1)].create_dataset('mu_tb', data=Simulation.System.molecule_types[type1].mu_tb)
                hdf['Molecules/'+str(type1)].create_dataset('mu_rb', data=Simulation.System.molecule_types[type1].mu_rb)
            
        #------------------
        
        hdf.close()
        

    def update_bins(self, Simulation):
        
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
        
        
        if self.Observing[1]['Volume']:
            self.binned['Volume'] += Simulation.System.volume
            
        if self.Observing[1]['Number']:
            for mol_name in self.observables_setup['Number']['items']:   
                mol_id = Simulation.System.molecule_types[mol_name].type_id
                self.binned['Number'][mol_id] += Simulation.System.Nmol[mol_id]
            
        if self.Observing[1]['Energy']:
            for mol_name in self.observables_setup['Energy']['items']:   
                mol_id = Simulation.System.molecule_types[mol_name].type_id
                self.binned['Energy'][mol_id] += Simulation.System.Epot[mol_id]
        
        if self.Observing[1]['Bonds']:
            for type1_id, type2_id in self.bond_pairs:
                self.binned['Bonds'][type1_id][type2_id] += Simulation.System.N_bonds[type1_id][type2_id]
            
        if self.Observing[1]['Pressure']:
            for mol_name in self.observables_setup['Pressure']['items']:   
                mol_id = Simulation.System.molecule_types[mol_name].type_id
                self.binned['Pressure'][mol_id] += Simulation.System.Pressure[mol_id]         

        
        if self.Observing[1]['Virial']:
            for mol_name in self.observables_setup['Virial']['items']:   
                mol_id = Simulation.System.molecule_types[mol_name].type_id
                self.binned['Virial'][mol_id] += Simulation.System.virial_scalar[mol_id]+Simulation.System.virial_scalar_Wall[mol_id] 

                
        if self.Observing[1]['Virial Tensor']:
            for mol_name in self.observables_setup['Virial Tensor']['items']:   
                mol_id = Simulation.System.molecule_types[mol_name].type_id
                self.binned['Virial Tensor'][mol_id] += Simulation.System.virial_tensor[mol_id]#+Simulation.System.virial_tensor_Wall[mol_id] 
            
            
                    
    #%%
    
    def observe(self, Property, stride, Simulation, types = None, reactions = None, stepwise = True, binned= False):#, save = True, keep_list = False):
        
        
        self.Observing[0][Property] = True
        if binned:
            self.Observing[1][Property] = True
            
        self.observables_setup[Property] = dict()
        
        self.observables_setup[Property]['stride'] = stride
        # self.observables_setup[Property]['save'] = save
        # self.observables_setup[Property]['keep_list'] = keep_list
        self.observables_setup[Property]['nsteps'] = int((Simulation.nsteps)/stride)
        self.observables_setup[Property]['items'] = []
        # self.observables_setup[Property]['trace'] = dict()
        self.observables_setup[Property]['binned'] = binned
        self.observables_setup[Property]['stepwise'] = stepwise
        self.observables_setup[Property]['current_step'] = 0
        
        # if save==True:
        hdf = h5py.File(Simulation.file_path / 'hdf5' / (Simulation.file_name+'.h5'), 'a')
            
        if stepwise == True:
            group = hdf.create_group('stepwise/'+Property, track_order=False)
            group.attrs['stride'] = stride
        if binned == True:
            group = hdf.create_group('binned/'+Property, track_order=False)
            group.attrs['stride'] = stride
        
            
        if Property in ['Force', 'Torque', 'Orientation', 'Position']:
            
            for type1 in types:
                if str(type1) not in Simulation.System.molecule_types:
                    raise KeyError('Molecule type not found')
                    
                self.observables_setup[Property]['items'].append(str(type1))
                
                # if keep_list == True:
                #     self.observables_setup[Property]['trace'][str(type1)] = []
                # if save==True:
                if stepwise == True:
                    group = hdf.create_group('stepwise/'+Property+'/'+str(type1), track_order=False)
                if binned == True:
                    group = hdf.create_group('binned/'+Property+'/'+str(type1), track_order=False)
                    
        elif Property in ['Volume']:
            
            # if keep_list == True:
            #     self.observables_setup[Property]['trace'] = []
            # if save== True:
            if stepwise == True:
                dataset = hdf.create_dataset('stepwise/Volume/'+Property, shape=(0,), dtype='f8', maxshape=(None,))
            if binned == True:
                dataset = hdf.create_dataset('binned/Volume/'+Property, shape=(0,), dtype='f8', maxshape=(None,))
                    
                    
        elif Property in ['Energy','Pressure', 'Virial', 'Number', 'Virial Tensor']:
                
                
            if Property in ['Pressure', 'Virial', 'Energy']:
                shape = (0,)
                max_shape = (None,)
                dtype = 'f8'
            elif Property in ['Virial Tensor']:
                shape = (0,3,3)
                max_shape = (None,3,3)
                dtype = 'f8'
            elif Property in ['Number']:
                shape = (0,)
                max_shape = (None,)
                dtype = 'i8'
                
            for type1 in types:
                if str(type1) not in Simulation.System.molecule_types:
                    raise KeyError('Molecule type not found')
                    
                self.observables_setup[Property]['items'].append(str(type1))
                
                # if keep_list == True:
                #     self.observables_setup[Property]['trace'][str(type1)] = []
                # if save== True:
                if stepwise == True:
                    dataset = hdf.create_dataset('stepwise/'+Property+'/'+str(type1), shape=shape, dtype=dtype, maxshape=max_shape)
                if binned == True:
                    dataset = hdf.create_dataset('binned/'+Property+'/'+str(type1), shape=shape, dtype='f8', maxshape=max_shape)     
                            
                            
        elif Property in ['Reactions']:
            
            for reaction_type_index in reactions:
                
                # reaction_type_id = Simulation.System.Reactions_Dict[reaction_type_index].reaction_type_id
                bimol = Simulation.System.Reactions_Dict[reaction_type_index].bimol
                
                if bimol:
                    educts_names = []
                    for educt_id in Simulation.System.Reactions_Dict[reaction_type_index].educts:
                        if Simulation.System.Reactions_Dict[reaction_type_index].reaction_educt_type == 'Molecule':
                            educts_names.append(Simulation.System.molecule_id_to_name[educt_id])
                        else:
                            educts_names.append(Simulation.System.particle_id_to_name[educt_id])
                    educts = educts_names[0]+'+'+educts_names[1]
                else:
                    educt_id = Simulation.System.Reactions_Dict[reaction_type_index].educts[0]
                    if Simulation.System.Reactions_Dict[reaction_type_index].reaction_educt_type == 'Molecule':
                        educts = Simulation.System.molecule_id_to_name[educt_id]
                    else:
                        educts = Simulation.System.particle_id_to_name[educt_id]
                
                self.observables_setup[Property]['items'].append([reaction_type_index,educts])
                
                if Simulation.System.Reactions_Dict[reaction_type_index].reaction_educt_type == 'Molecule' and bimol:
                    
                    particle_educts = Simulation.System.Reactions_Dict[reaction_type_index].particle_educts
                    particle_educts = Simulation.System.particle_id_to_name[particle_educts[0]]+'+'+Simulation.System.particle_id_to_name[particle_educts[1]]
                    
                    if stepwise == True:
                        hdf.create_dataset('stepwise/'+Property+'/'+particle_educts+'/n_total', shape=(0,), dtype='i8', maxshape=(None,))
                    if binned == True:
                        hdf.create_dataset('binned/'+Property+'/'+particle_educts+'/n_total', shape=(0,), dtype='i8', maxshape=(None,))
                else:
                    if stepwise == True:
                        hdf.create_dataset('stepwise/'+Property+'/'+educts+'/n_total', shape=(0,), dtype='i8', maxshape=(None,))
                    if binned == True:
                        hdf.create_dataset('binned/'+Property+'/'+educts+'/n_total', shape=(0,), dtype='f8', maxshape=(None,))
                
                for i in range(len(Simulation.System.Reactions_Dict[reaction_type_index].paths)):
                    
                    reaction_type = Simulation.System.Reactions_Dict[reaction_type_index].paths[i]['type']
                    
                    path = 'path_'+str(i)
                    
                    if Simulation.System.Reactions_Dict[reaction_type_index].reaction_educt_type == 'Molecule' and bimol:
                    
                        if stepwise == True:
                            dataset = hdf.create_dataset('stepwise/'+Property+'/'+particle_educts+'/'+reaction_type+'/'+educts+'/'+path, shape=(0,), dtype='i8', maxshape=(None,))
                        if binned == True:
                            dataset = hdf.create_dataset('binned/'+Property+'/'+particle_educts+'/'+reaction_type+'/'+educts+'/'+path, shape=(0,), dtype='f8', maxshape=(None,))
                    
                    else:
                        
                        if stepwise == True:
                            dataset = hdf.create_dataset('stepwise/'+Property+'/'+educts+'/'+reaction_type+'/'+path, shape=(0,), dtype='i8', maxshape=(None,))
                        if binned == True:
                            dataset = hdf.create_dataset('binned/'+Property+'/'+educts+'/'+reaction_type+'/'+path, shape=(0,), dtype='f8', maxshape=(None,))
                        
                    
                    
        elif Property in ['Bonds']:

            ii = np.where(Simulation.System.reaction_args['bond'])
            
            self.bond_pairs = list(zip(ii[0],ii[1]))
            
            for type1_id, type2_id in self.bond_pairs:
                
                type1_name = Simulation.System.particle_id_to_name[type1_id]
                type2_name = Simulation.System.particle_id_to_name[type2_id]
                
                bond_pair = type1_name+'.'+type2_name
                
                if stepwise == True:
                    dataset = hdf.create_dataset('stepwise/'+Property+'/'+bond_pair, shape=(0,), dtype='i8', maxshape=(None,))
                if binned == True:
                    dataset = hdf.create_dataset('binned/'+Property+'/'+bond_pair, shape=(0,), dtype='i8', maxshape=(None,))     
                                
            
        # if save==True:
        hdf.close()    
            
        print('Observing {}.'.format(Property))
        
    #%%
            
            
    def observe_rdf(self, rdf_pairs, rdf_bins, rdf_cutoff, stride, Simulation): #, save = True, keep_list = False):
        
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
        
        hdf = h5py.File(Simulation.file_path / 'hdf5' / (Simulation.file_name+'.h5'), 'a')
        
        self.observing_rdf =True
        self.rdf_nsteps = int((Simulation.nsteps)/stride)
        self.rdf_current_step = 0
        
        self.rdf_cutoff = np.array(rdf_cutoff)
        self.rdf_bins = np.array(rdf_bins)
        self.rdf_stride = stride
        
        self.rdf_pairs = rdf_pairs
        self.rdf_molecules = set(sum(rdf_pairs,[])) # [[mol1, mol2], [mol2, mol3]] -> {mol1, mol2, mol3}
        
        self.rdf_hgrid = create_rb_hgrid(Simulation, Simulation.RBs, self.rdf_molecules, Simulation.System.box_lengths, Simulation.System.N, Simulation.System)
        
        self.rdf_measure_pair = np.zeros((Simulation.System.ntypes, Simulation.System.ntypes), dtype = np.int64)
        self.rdf_hist = dict()
        for i in range(len(rdf_pairs)):
            type1, type2 = rdf_pairs[i]
            
            type1_id = Simulation.System.molecule_types[str(type1)].type_id
            type2_id = Simulation.System.molecule_types[str(type2)].type_id
            
            if (str(type1) not in Simulation.System.molecule_types) or (str(type2) not in Simulation.System.molecule_types):
                raise KeyError('Molecule type not found')
            
            self.rdf_hist[type1+'.'+type2] = np.zeros(rdf_bins[i])
            self.rdf_measure_pair[type1_id][type2_id] = True
            
            dataset = hdf.create_group('RDF/'+str(type1)+'.'+str(type2), track_order=False)
            dataset.attrs['stride'] = stride
            dataset.attrs['rdf_cutoff'] = rdf_cutoff[i]
            dataset.attrs['rdf_bins'] = rdf_bins[i]
            
        hdf.close() 
        
            
    #%%
    
    def update(self, Simulation, hdf, RBs):
        
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
        
        for Property in self.observables_setup:
            
            observable = self.observables_setup[Property]
            
            if Simulation.current_step % observable['stride'] == 0:
                
                # hdf = h5py.File(Simulation.file_path+'hdf5/'+Simulation.file_name+'.h5', 'a')
                
                if Property in ['Force', 'Torque', 'Orientation', 'Position']:
                    
                    digits = int(math.log10(observable['nsteps']))+1
                    step = observable['current_step']
                    step_id = f'{step:0{digits}}'
                    
                    
                    for mol_name in observable['items']:   
                        mol_id = Simulation.System.molecule_types[mol_name].type_id
                        
                        Mask = RBs[RBs.occupied[0:RBs.occupied.n]]['type_id']==mol_id
                        
                        # if observable['stepwise'] == True:
                        if Property == 'Force':
                            dataset = hdf.create_dataset('stepwise/'+Property+'/'+str(mol_name)+'/'+step_id, data=RBs[RBs.occupied[0:RBs.occupied.n]]['force'][Mask])
                        elif Property == 'Torque':
                            dataset = hdf.create_dataset('stepwise/'+Property+'/'+str(mol_name)+'/'+step_id, data=RBs[RBs.occupied[0:RBs.occupied.n]]['torque'][Mask])
                        elif Property == 'Orientation':
                            dataset = hdf.create_dataset('stepwise/'+Property+'/'+str(mol_name)+'/'+step_id, data=RBs[RBs.occupied[0:RBs.occupied.n]]['q'][Mask])
                        elif Property == 'Position':
                            dataset = hdf.create_dataset('stepwise/'+Property+'/'+str(mol_name)+'/'+step_id, data=RBs[RBs.occupied[0:RBs.occupied.n]]['pos'][Mask])
                            
                        dataset.attrs['time step'] = Simulation.current_step
                    
                    observable['current_step'] += 1
                    
                    
                elif Property in ['Volume']:
                    
                    if observable['stepwise'] == True:
                        Dset = hdf['stepwise/Volume/'+Property]
                        Dset.resize((Dset.shape[0] + 1), axis = 0)
                        if Property == 'Volume':
                            Dset[-1:] = Simulation.System.volume
                    if observable['binned'] == True:
                        Dset = hdf['binned/Volume/'+Property]
                        Dset.resize((Dset.shape[0] + 1), axis = 0)
                        Dset[-1:] = self.binned[Property]       
                        
                        self.binned[Property] = 0.0
                
                elif Property in ['Energy', 'Pressure', 'Virial', 'Virtial Tensor', 'Number']:
                    
                    for mol_name in observable['items']: 
                        mol_id = Simulation.System.molecule_types[mol_name].type_id
                        
                        if observable['stepwise'] == True:
                            Dset = hdf['stepwise/'+Property+'/'+str(mol_name)]
                            Dset.resize((Dset.shape[0] + 1), axis = 0)
                            if Property == 'Energy':
                                Dset[-1:] = Simulation.System.Epot[mol_id]
                            elif Property == 'Pressure':
                                Dset[-1:] = Simulation.System.Pressure[mol_id]
                            elif Property == 'Virial':
                                Dset[-1:] = Simulation.System.virial_scalar[mol_id]+Simulation.System.virial_scalar_Wall[mol_id]
                            elif Property == 'Virial Tensor':
                                Dset[-1:] = Simulation.System.virial_tensor[mol_id]
                            elif Property == 'Number':
                                Dset[-1:] = Simulation.System.Nmol[mol_id]
                        if observable['binned'] == True:
                            Dset = hdf['binned/'+Property+'/'+str(mol_name)]
                            Dset.resize((Dset.shape[0] + 1), axis = 0)
                            Dset[-1:] = self.binned[Property][mol_id]
                            
                            if Property == 'Virial Tensor':
                                self.binned[Property][:,:,:] = 0.0
                            else:       
                                self.binned[Property][:] = 0.0
                    
                
                elif Property in ['Reactions']:
                    
                    for reaction_type_index,educts in observable['items']: 
                        
                        bimol = Simulation.System.Reactions_Dict[reaction_type_index].bimol
                        
                        if Simulation.System.Reactions_Dict[reaction_type_index].reaction_educt_type == 'Molecule' and bimol:
                            
                            particle_educts = Simulation.System.Reactions_Dict[reaction_type_index].particle_educts
                            particle_educts = Simulation.System.particle_id_to_name[particle_educts[0]]+'+'+Simulation.System.particle_id_to_name[particle_educts[1]]
                    
                            if observable['stepwise'] == True:
                                Dset = hdf['stepwise/'+Property+'/'+particle_educts+'/n_total']
                                Dset.resize((Dset.shape[0] + 1), axis = 0)
                                Dset[-1:] = Simulation.System.Reactions_Dict[reaction_type_index].n_total
                            if observable['binned'] == True:
                                Dset = hdf['binned/'+Property+'/'+particle_educts+'/n_total']
                                Dset.resize((Dset.shape[0] + 1), axis = 0)
                                Dset[-1:] = Simulation.System.Reactions_Dict[reaction_type_index].n_total_binned#/observable['stride']
                                
                                Simulation.System.Reactions_Dict[reaction_type_index].n_total_binned = 0                            
                    
                        else:
                            if observable['stepwise'] == True:
                                Dset = hdf['stepwise/'+Property+'/'+educts+'/n_total']
                                Dset.resize((Dset.shape[0] + 1), axis = 0)
                                Dset[-1:] = Simulation.System.Reactions_Dict[reaction_type_index].n_total
                            if observable['binned'] == True:
                                Dset = hdf['binned/'+Property+'/'+educts+'/n_total']
                                Dset.resize((Dset.shape[0] + 1), axis = 0)
                                Dset[-1:] = Simulation.System.Reactions_Dict[reaction_type_index].n_total_binned#/observable['stride']
                                
                                Simulation.System.Reactions_Dict[reaction_type_index].n_total_binned = 0
                        
                        for i in range(len(Simulation.System.Reactions_Dict[reaction_type_index].paths)):
                            
                            reaction_type = Simulation.System.Reactions_Dict[reaction_type_index].paths[i]['type']
                            
                            path = 'path_'+str(i)
                            
                            if Simulation.System.Reactions_Dict[reaction_type_index].reaction_educt_type == 'Molecule' and bimol:
                    
                                if observable['stepwise'] == True:
                                    
                                    Dset = hdf['stepwise/'+Property+'/'+particle_educts+'/'+reaction_type+'/'+educts+'/'+path]
                                    Dset.resize((Dset.shape[0] + 1), axis = 0)
                                    Dset[-1:] = Simulation.System.Reactions_Dict[reaction_type_index].paths[i]['n_success']
                                    
                                if observable['binned'] == True:
                                    
                                    Dset = hdf['binned/'+Property+'/'+particle_educts+'/'+reaction_type+'/'+educts+'/'+path]
                                    Dset.resize((Dset.shape[0] + 1), axis = 0)
                                    Dset[-1:] = Simulation.System.Reactions_Dict[reaction_type_index].paths[i]['n_success_binned']#/observable['stride']
                                    
                                    Simulation.System.Reactions_Dict[reaction_type_index].paths[i]['n_success_binned'] = 0
                                
                            else:
                                if observable['stepwise'] == True:
                                    
                                    Dset = hdf['stepwise/'+Property+'/'+educts+'/'+reaction_type+'/'+path]
                                    Dset.resize((Dset.shape[0] + 1), axis = 0)
                                    Dset[-1:] = Simulation.System.Reactions_Dict[reaction_type_index].paths[i]['n_success']
                                    
                                if observable['binned'] == True:
                                    
                                    Dset = hdf['binned/'+Property+'/'+educts+'/'+reaction_type+'/'+path]
                                    Dset.resize((Dset.shape[0] + 1), axis = 0)
                                    Dset[-1:] = Simulation.System.Reactions_Dict[reaction_type_index].paths[i]['n_success_binned']#/observable['stride']
                                    
                                    Simulation.System.Reactions_Dict[reaction_type_index].paths[i]['n_success_binned'] = 0
                                    
                                    
                elif Property in ['Bonds']:
                    
                    for type1_id, type2_id in self.bond_pairs:
                        
                        type1_name = Simulation.System.particle_id_to_name[type1_id]
                        type2_name = Simulation.System.particle_id_to_name[type2_id]
                        
                        bond_pair = type1_name+'.'+type2_name
                        
                        if observable['stepwise'] == True:
                            Dset = hdf['stepwise/'+Property+'/'+bond_pair]
                            Dset.resize((Dset.shape[0] + 1), axis = 0)
                            Dset[-1:] = Simulation.System.N_bonds[type1_id][type2_id]
                        if observable['binned'] == True:
                            Dset = hdf['binned/'+Property+'/'+bond_pair]
                            Dset.resize((Dset.shape[0] + 1), axis = 0)
                            Dset[-1:] = self.binned[Property][type1_id][type2_id]
                            
                            self.binned[Property][:,:] = 0.0
                                
                #%%
                
                # hdf.close() 
                    
    #%%
    
    
    def update_rdf(self, Simulation, hdf):
        
        
        if Simulation.current_step % self.rdf_stride == 0:
            
            for i in range(len(self.rdf_pairs)):
                
                mol1, mol2 =self.rdf_pairs[i]
                
                typeI_id = Simulation.System.molecule_types[str(mol1)].type_id
                typeJ_id = Simulation.System.molecule_types[str(mol2)].type_id
                
                rdf_hist = self.rdf_hist[mol1+'.'+mol2]
                
                if Simulation.System.Nmol[typeI_id]>0 and Simulation.System.Nmol[typeJ_id]>0:
                
                    rdf_hgrid = self.rdf_hgrid
                    rdf_cutoff = self.rdf_cutoff[i]
                    rdf_bins = self.rdf_bins[i]
                    
                    radial_distr_function(Simulation.System, np.array([typeI_id,typeJ_id]), Simulation.RBs, rdf_hgrid, rdf_cutoff, rdf_bins, rdf_hist)
                    
                    dr_rdf = rdf_cutoff / float(rdf_bins)
                    r = np.linspace(0, rdf_cutoff, rdf_bins)
                    
                    rdf_hist = rdf_hist/(4/3*np.pi*(r**3-(r-dr_rdf)**3))/(Simulation.System.Nmol[typeJ_id]/Simulation.System.box_lengths.prod())/(Simulation.System.Nmol[typeI_id])
                
                else:
                    
                    rdf_hist[:] = 0.0
                    
                digits = int(math.log10(self.rdf_nsteps))+1
                step = self.rdf_current_step
                step_id = f'{step:0{digits}}'
                
                dataset = hdf.create_dataset('RDF/'+str(mol1)+'.'+str(mol2)+'/'+step_id, data=rdf_hist)
                dataset.attrs['time step'] = Simulation.current_step
                
            self.rdf_current_step += 1
            
            
            