# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numpy as np
import h5py
import os
# import math
# import itertools
from pathlib import Path

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from itertools import product, combinations
import seaborn as sns
sns.set(style='ticks')
import pandas as pd
from pyvis.network import Network

col=sns.color_palette("colorblind", 10)
from matplotlib.font_manager import FontProperties
fontLgd = FontProperties()
fontLgd.set_size('x-small')

# import plotly.graph_objects as go
import plotly.io as pio
#pio.renderers.default = 'svg'
pio.renderers.default = 'browser'


# from ..evaluation import direct_coexistence_method_util as dcm
from ..evaluation import diffusion_util as diff
from ..math.transform_util import unique_pairing

# from ..observables_util.plot_util import plot_observable

class Evaluation(object):
    
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
    
    def __init__(self, file_name = 'PyRID', path = None) :
        
        
        self.file_name = file_name
            
        if path is None:
            self.path = Path(os.getcwd()) / 'Files' / self.file_name
            self.fig_path = Path(os.getcwd()) / 'Figures'
        else:
            self.path = Path(path) / self.file_name
            self.fig_path = Path(path) / 'Figures'
            
            try:
                os.makedirs(self.fig_path) 
            except FileExistsError:
                # directory already exists
                pass
            
        self.Observables = {}
        self.rdf = {}
        
        self.Measures = ['Position', 'Energy', 'Pressure', 'Volume', 'Virial', 'Virial Tensor', 'Number', 'Orientation', 'Bonds', 'Reactions', 'Force', 'Torque', 'RDF']
        
        self.time_MSD = {}
        self.MSD_data = {}
        
        self.time_P2 = {}
        self.P2_data = {}
        self.P2_t = {}
        
        self.molecules_colors = np.array([[0.12156863, 0.46666667, 0.70588235, 1.        ],
           [1.        , 0.49803922, 0.05490196, 1.        ],
           [0.17254902, 0.62745098, 0.17254902, 1.        ],
           [0.83921569, 0.15294118, 0.15686275, 1.        ],
           [0.58039216, 0.40392157, 0.74117647, 1.        ],
           [0.54901961, 0.3372549 , 0.29411765, 1.        ],
           [0.89019608, 0.46666667, 0.76078431, 1.        ],
           [0.49803922, 0.49803922, 0.49803922, 1.        ],
           [0.7372549 , 0.74117647, 0.13333333, 1.        ],
           [0.09019608, 0.74509804, 0.81176471, 1.        ],
           [0.4       , 0.76078431, 0.64705882, 1.        ],
           [0.98823529, 0.55294118, 0.38431373, 1.        ],
           [0.55294118, 0.62745098, 0.79607843, 1.        ],
           [0.90588235, 0.54117647, 0.76470588, 1.        ],
           [0.65098039, 0.84705882, 0.32941176, 1.        ],
           [1.        , 0.85098039, 0.18431373, 1.        ],
           [0.89803922, 0.76862745, 0.58039216, 1.        ],
           [0.70196078, 0.70196078, 0.70196078, 1.        ],
           [0.86      , 0.3712    , 0.34      , 1.        ],
           [0.8288    , 0.86      , 0.34      , 1.        ],
           [0.34      , 0.86      , 0.3712    , 1.        ],
           [0.34      , 0.8288    , 0.86      , 1.        ],
           [0.3712    , 0.34      , 0.86      , 1.        ],
           [0.86      , 0.34      , 0.8288    , 1.        ]])
        
        self.molecules_colors_rgb = np.array(np.round(self.molecules_colors[:,0:3]*255), dtype = np.int64)
        
        self.molecules_colors_hex = ['#%02x%02x%02x' % tuple(rgb) for rgb in self.molecules_colors_rgb]
            
        self.reactions_color = dict()
        self.reactions_color['bind'] = '#000000'
        self.reactions_color['fusion'] = self.molecules_colors_hex[0]
        self.reactions_color['enzymatic_mol'] = self.molecules_colors_hex[1]
        self.reactions_color['enzymatic'] = self.molecules_colors_hex[2]
        self.reactions_color['absorption'] = self.molecules_colors_hex[3]
        self.reactions_color['conversion_mol'] = self.molecules_colors_hex[4]
        self.reactions_color['conversion'] = self.molecules_colors_hex[5]
        self.reactions_color['fission'] = self.molecules_colors_hex[6]
        self.reactions_color['production'] = self.molecules_colors_hex[7]
        self.reactions_color['release'] = self.molecules_colors_hex[8]
        self.reactions_color['decay'] = self.molecules_colors_hex[9]
        
        self.color_pallete = dict()
        self.color_pallete['Energy'] = 'rocket'
        self.color_pallete['Pressure'] = 'viridis'
        self.color_pallete['Number'] = 'crest'
        self.color_pallete['Reactions'] = "cubehelix"
        self.color_pallete['Bonds'] = "icefire"
        self.color_pallete['Volume'] = 'Blues'
        self.color_pallete['RDF'] = "mako"
        self.color_pallete['ODF'] = "flare"
        self.color_pallete['Virial'] = 'magma'
        self.color_pallete['Virial Tensor'] = "cubehelix"
        self.color_pallete['Force'] = 'rocket'
        self.color_pallete['Torque'] = 'viridis'
        self.color_pallete['Orientation'] = 'Spectral'
        self.color_pallete['Position'] = "icefire"
        
        
    def load_file(self, file_name, path = None):
        
        if '.h5' in file_name:
            self.file_name = file_name[0:-3]
        else:
            self.file_name = file_name
            
        if path is None:
            self.path = Path(os.getcwd()) / 'Files/hdf5' / (self.file_name+'.h5')
            self.fig_path = Path(os.getcwd()) / 'Figures'
        else:
            self.path = Path(path) / (self.file_name+'.h5')
            self.fig_path = Path(path) / 'Figures'
            
            try:
                os.makedirs(self.fig_path) 
            except FileExistsError:
                # directory already exists
                pass
            
        hdf = h5py.File(self.path, 'r', track_order=True)
        
        self.units = dict()
        for measure in hdf['Setup']['units']:
            self.units[measure] = hdf['Setup']['units'][measure][()].decode()
            
        
        self.Temp = hdf['Setup']['Temp'][()]
        self.box_lengths = hdf['Setup']['box_lengths'][()]
        self.dt = hdf['Setup']['dt'][()]
        self.eta = hdf['Setup']['eta'][()]
        self.kbt = hdf['Setup']['kbt'][()]
        self.nsteps = hdf['Setup']['nsteps'][()]
        
        self.read_molecule_data()
        
        hdf.close()
        
    def read_molecule_data(self):
        
        hdf = h5py.File(self.path, 'r', track_order=True)
        
        self.Molecules = dict()
        
        for molecule in hdf['Molecules']:
            
            self.Molecules[molecule] = dict()
            self.Molecules[molecule]['pos'] = hdf['Molecules'][molecule]['pos'][()]
            self.Molecules[molecule]['radii'] = hdf['Molecules'][molecule]['radii'][()]
            self.Molecules[molecule]['volume'] = hdf['Molecules'][molecule]['volume'][()]
            self.Molecules[molecule]['mu_rb'] = hdf['Molecules'][molecule]['mu_rb'][()]
            self.Molecules[molecule]['mu_tb'] = hdf['Molecules'][molecule]['mu_tb'][()]
            
        hdf.close()
        
    #%%
    
    def load_hdf(self):
        
        self.hdf = h5py.File(self.path, 'r', track_order=True)
    
    def close_hdf(self):
        
        self.hdf.close()
        
        
    #%%
    
    def MSD(self, time_interval, stride, Simulation, molecule):
        
        self.read_observable('Position', sampling = 'stepwise', molecules = [molecule], steps = 'All')
        
        position_trace = []
        for step in self.Observables['stepwise']['Position'][molecule]:
            position_trace.append(self.Observables['stepwise']['Position'][molecule][step])
            
        position_trace = np.array(position_trace)
        pos_stride = self.Observables['stride']['Position'] 
        Delta_t = pos_stride*self.dt
        
        self.MSD_data[molecule], self.time_MSD[molecule] = diff.MSD(position_trace, Delta_t, time_interval, stride, molecule)
        
        

    def plot_MSD(self, Simulation, molecule, save_fig = False, fig_name = None, fig_path=None):
        
        sns.set_style("whitegrid")
        
        MSD_x, MSD_y, MSD_z = self.MSD_data[molecule]
        
        D_tt = Simulation.System.molecule_types[molecule].D_tt
        D_trans = np.trace(D_tt)/3
        
        plt.figure(figsize = (4,3), dpi=150)
        
        plt.scatter(self.time_MSD[molecule], MSD_x, marker='o', facecolor = 'none', edgecolors = 'k', label = 'x', linewidth = 1)
        plt.scatter(self.time_MSD[molecule], MSD_y, marker='d', facecolor = 'none', edgecolors = 'r', label = 'y', linewidth = 1)
        plt.scatter(self.time_MSD[molecule], MSD_z, marker='s', facecolor = 'none', edgecolors = 'g', label = 'z', linewidth = 1)
        
        plt.plot(self.time_MSD[molecule], 2*D_trans*self.time_MSD[molecule], 'k', linewidth = 2, label = 'Theory')
        plt.legend(prop=fontLgd)
        plt.xlabel('time in {}'.format(Simulation.System.time_unit))
        plt.ylabel('MSD')
        
        if save_fig == True:
            if fig_name is None:
                fig_name = Simulation.file_name + '_' + molecule
            if fig_path is None:
                fig_path = Simulation.fig_path
                
            plt.savefig(fig_path / (fig_name+'_MSD.png'), bbox_inches="tight", dpi=300)
            

    #%%

    def P2(self, time_interval, stride, Simulation, molecule, theory_only = False, Delta_t = None):
        
        self.read_observable('Orientation', sampling = 'stepwise', molecules = [molecule], steps = 'All')
        
        D_rr = Simulation.System.molecule_types[molecule].D_rr
        
        orientation_trace = []
        for step in self.Observables['stepwise']['Orientation'][molecule]:
            orientation_trace.append(self.Observables['stepwise']['Orientation'][molecule][step])
            
        orientation_trace = np.array(orientation_trace)
        orientation_stride = self.Observables['stride']['Orientation'] 
        Delta_t = orientation_stride*self.dt
        
        result =  diff.P2(orientation_trace, Delta_t, D_rr, time_interval, stride, Simulation, molecule, theory_only = theory_only)

        if theory_only == False:
            self.P2_data[molecule], self.P2_t[molecule], self.time_P2[molecule] = result
            P2_1_t, P2_2_t, P2_3_t = self.P2_t[molecule]
            P2_1, P2_2, P2_3 = self.P2_data[molecule]
        else:
            self.P2_t[molecule], self.time_P2[molecule] = result
            P2_1_t, P2_2_t, P2_3_t = self.P2_t[molecule]
        
    
    def plot_P2(self, Simulation, molecule, theory_only = False, save_fig = False, fig_name = None, fig_path=None, limits = None):
        
        sns.set_style("whitegrid")
        
        P2_1_t, P2_2_t, P2_3_t = self.P2_t[molecule]
        P2_1, P2_2, P2_3 = self.P2_data[molecule]
        
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        plt.figure(figsize = (4,3), dpi=150)
        
        plt.plot(self.time_P2[molecule], P2_1_t, color = 'grey', label = 'Theory')
        plt.plot(self.time_P2[molecule], P2_2_t, color = 'grey')
        plt.plot(self.time_P2[molecule], P2_3_t, color = 'grey')
        
        if theory_only == False:
            plt.scatter(self.time_P2[molecule], P2_1, facecolors='none', edgecolors = colors[0], label = '1')
            plt.scatter(self.time_P2[molecule], P2_2, facecolors='none', edgecolors = colors[1], label = '2')
            plt.scatter(self.time_P2[molecule], P2_3, facecolors='none', edgecolors = colors[2], label = '3')
        
        plt.legend(prop=fontLgd)
        
        if limits is not None:
            plt.xlim(limits[0][0], limits[0][1])
            plt.ylim(limits[1][0], limits[1][1])
            
        plt.xlabel('time in {}'.format(Simulation.System.time_unit))
        plt.ylabel(r'$\langle P_{\hat{u}}(t) \rangle$')
        plt.yscale('log',base=10) 
        
        if save_fig == True:
            if fig_name is None:
                fig_name = Simulation.file_name + '_' + molecule
            if fig_path is None:
                fig_path = Simulation.fig_path
                
            plt.savefig(fig_path / (fig_name+'_RotDiff_Pu.png'), bbox_inches="tight", dpi=300)
                
    
    #%%
    
    # def plot_observable(self, measure, molecules = 'All', step = 0, save_fig = False, fig_name = None , fig_path = None, binned = False):
    
    #     plot_observable(self, measure, molecules = molecules, step = step, save_fig = save_fig, fig_name = fig_name , fig_path = fig_path, binned = binned)
        
        
    def plot_rdf(self, mol_pairs, steps = [0], save_fig = False, fig_name = None , fig_path = None, average = False):
        
        sns.set_style("whitegrid")
        
        hdf = h5py.File(self.path, 'r', track_order=True)
        
        if average:
            plt.figure()
            
        for mol1, mol2 in mol_pairs:
            
            if steps == 'All':
                steps = list(hdf['RDF'][mol1+'.'+mol2].keys())
            else:
                digits = len(list(hdf['RDF'][mol1+'.'+mol2].keys())[-1])
                steps_temp = []
                for step in steps:
                    # digits = int(math.log10(self.nsteps))+1
                    step_id = f'{step:0{digits}}'
                    steps_temp.append(step_id)
                steps = steps_temp
            
            rdf_bins = hdf['RDF'][mol1+'.'+mol2].attrs['rdf_bins']
            rdf_cutoff = hdf['RDF'][mol1+'.'+mol2].attrs['rdf_cutoff']
            dr_rdf = rdf_cutoff / float(rdf_bins)
            r = np.linspace(0, rdf_cutoff, rdf_bins)+dr_rdf/2
            
            time_steps = []
            for step in steps:
        
                if mol1+'.'+mol2 not in self.rdf:
                    self.rdf[mol1+'.'+mol2] = dict()
                    
                time_step = str(self.dt*hdf['RDF'][mol1+'.'+mol2][step].attrs['time step']) # Need to use string, because for some reason seaborn doesnt handle numerical values for the hue value as expected!
                time_steps.append(time_step)
                
                if time_step not in self.rdf[mol1+'.'+mol2]:
                    self.rdf[mol1+'.'+mol2][time_step] = hdf['RDF'][mol1+'.'+mol2][step][()]
                
                
            df = pd.DataFrame(self.rdf[mol1+'.'+mol2])[time_steps]
            df['r'] = r
            dfm = df.melt('r', var_name = 'time ({})'.format(self.units['Time']), value_name = 'g(r)')
            
            fig, ax = plt.subplots(figsize=(4,3), dpi=150)
            
            plt.axhline(1.0, color= 'k', linewidth = 1, linestyle = '--')
            
            if average:
                g = sns.lineplot(x="r", y='g(r)', data = dfm)
            else:
                g = sns.lineplot(x="r", y='g(r)', hue='time ({})'.format(self.units['Time']), data = dfm, palette = 'rocket', ci = 'sd')

            plt.title(mol1+'.'+mol2)
            if average ==False:
                g.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='time ({})'.format(self.units['Time']), prop=fontLgd)
            plt.xlabel(r'r in {}'.format(self.units['Length']))
            plt.ylabel('g(r)')

            if save_fig == True:
                if fig_name is None:
                    fig_name = self.file_name + ''
                if fig_path is None:
                    fig_path = self.fig_path
                plt.savefig(fig_path / (fig_name+'_RDF_'+mol1+'-'+mol2+'.png'), bbox_inches="tight", dpi = 300)

      
        hdf.close()
        
        return fig, ax
        
    #%%
    
    def read_observable(self, measure, sampling = None, molecules = 'All', educts = 'All', Reaction_Type = None, steps = 'All', file_path = None):
        
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
        

        hdf = h5py.File(self.path, 'r', track_order=True)
        
        if sampling is None:
            if measure in ['Position', 'Orientation', 'Torque', 'Force', 'Virial', 'Virial Tensor', 'Number','Energy', 'Pressure', 'Bonds', 'Volume']:
                sampling = 'stepwise'
            elif measure == 'Reactions':
                sampling = 'binned'
            
        stride = hdf[sampling][measure].attrs['stride']
        
        if 'stride' not in self.Observables:
            self.Observables['stride'] = {}
            self.Observables['stride'][measure] = stride
        elif measure not in self.Observables['stride']:
            self.Observables['stride'][measure] = stride
            
        if sampling not in self.Observables:
            self.Observables[sampling] = {}
            self.Observables[sampling][measure] = {}
        elif measure not in self.Observables[sampling]:
                self.Observables[sampling][measure] = {}
                    
        if measure in ['Position', 'Orientation', 'Torque', 'Force', 'Virial', 'Virtial Tensor', 'Number','Energy', 'Pressure']:
            
            if molecules == 'All':
                molecules = list(hdf[sampling][measure].keys())
                
            if measure in ['Position', 'Orientation', 'Torque', 'Force']:
                timesteps_seperated = True # For properties such as position, the data for each time step is saved seperately in an array.
                if steps == 'All':
                    steps = list(hdf[sampling][measure][molecules[0]].keys())
                else:
                    digits = len(list(hdf[sampling][measure][molecules[0]].keys())[-1])
                    steps_temp = []
                    for step in steps:
                        # digits = int(math.log10(self.nsteps))+1
                        step_id = f'{step:0{digits}}'
                        steps_temp.append(step_id)
                    steps = steps_temp
                    
            elif measure in ['Virial', 'Virtial Tensor', 'Number','Energy', 'Pressure']:
                timesteps_seperated = False
                
            self.Observables[sampling][measure]['time'] = self.dt*stride*np.arange(len(hdf[sampling][measure][molecules[0]]))
            
            for molecule in molecules:
                
                if timesteps_seperated:
                    
                    if molecule not in self.Observables[sampling][measure]:
                        self.Observables[sampling][measure][molecule] = {}
                        
                    for step in steps:
                        # time_step = hdf[sampling][measure][molecule][step].attrs['time step']
                        # if time_step not in self.Observables[sampling][measure][molecule]:
                        if step not in self.Observables[sampling][measure][molecule]:
                            self.Observables[sampling][measure][molecule][int(step)] = hdf[sampling][measure][molecule][step][()]
                       
                else:
                    if molecule not in self.Observables[sampling][measure]:

                        self.Observables[sampling][measure][molecule] = hdf[sampling][measure][molecule][()]
            
                                                                                                                   
        elif measure in ['Volume']:
            
            self.Observables[sampling][measure]['time'] = self.dt*stride*np.arange(len(hdf[sampling][measure][measure]))
            
            self.Observables[sampling][measure][measure] = hdf[sampling][measure][measure][()]
                   
        elif measure == 'Reactions':
        
            if Reaction_Type in ['bind', 'enzymatic', 'conversion', 'conversion_rb', 'decay_rb', 'production_rb', 'release']:
            
                for educt in hdf[sampling]['Reactions'].keys():
                    for reaction_type in hdf[sampling]['Reactions'][educt].keys():
                        
                        if reaction_type == Reaction_Type:
                            
                            if reaction_type not in self.Observables[sampling][measure]:
                                self.Observables[sampling][measure][reaction_type] = {}
                                
                            if educt not in self.Observables[sampling][measure][reaction_type]:
                                self.Observables[sampling][measure][reaction_type][educt] = {}
                            
                            for reaction_path in hdf[sampling]['Reactions'][educt][reaction_type].keys():

                                if reaction_path not in self.Observables[sampling][measure][reaction_type][educt]:
                                    
                                    self.Observables[sampling][measure][Reaction_Type][educt][reaction_path] = hdf[sampling]['Reactions'][educt][reaction_type][reaction_path][()]
                                    
            elif Reaction_Type in ['fusion', 'enzymatic_rb']:

                for particle_educt in hdf[sampling]['Reactions'].keys():
                    for reaction_type in hdf[sampling]['Reactions'][particle_educt].keys():
                        
                        if reaction_type == Reaction_Type:
                            
                            if reaction_type not in self.Observables[sampling][measure]:
                                self.Observables[sampling][measure][reaction_type] = {}
                            
                            for educt in hdf[sampling]['Reactions'][particle_educt][reaction_type].keys():
                                
                                if educt not in self.Observables[sampling][measure][reaction_type]:
                                    self.Observables[sampling][measure][reaction_type][educt] = {}
                                
                                if particle_educt not in self.Observables[sampling][measure][reaction_type][educt]:
                                    self.Observables[sampling][measure][reaction_type][educt][particle_educt] = {}
                                    
                                for reaction_path in hdf[sampling]['Reactions'][particle_educt][reaction_type][educt].keys():
    
                                    if reaction_path not in self.Observables[sampling][measure][reaction_type][educt][particle_educt]:
                                        
                                        self.Observables[sampling][measure][Reaction_Type][educt][particle_educt][reaction_path] = hdf[sampling]['Reactions'][particle_educt][reaction_type][educt][reaction_path][()]                
                
        
        elif measure == 'Bonds':
            
            for bond_pair in hdf[sampling]['Bonds'].keys():
                
                self.Observables[sampling][measure][bond_pair] = hdf[sampling]['Bonds'][bond_pair][()]            
            
                                    
        hdf.close()
    
    #%%
    
    def plot_observable(self, measure, molecules = 'All', Reaction_Type = None, educt = None, bond_pairs = 'All', particle_educt = None, step = 0, save_fig = False, fig_name = None , fig_path = None, sampling = None, formats = ['png'], show = True):
        
        # Measures = ['Position', 'Energy', 'Pressure', 'Volume', 'Virial', 'Virial Tensor', 'Number', 'Orientation', 'Bonds', 'Reactions', 'Force', 'Torque', 'RDF']
        
        sns.set_style("whitegrid")
        
        if sampling is None:
            if measure in ['Position', 'Orientation', 'Torque', 'Force', 'Virial', 'Virial Tensor', 'Number','Energy', 'Pressure', 'Bonds', 'Volume']:
                sampling = 'stepwise'
            elif measure == 'Reactions':
                sampling = 'binned'
                
        self.read_observable(measure, sampling = sampling, molecules = molecules, Reaction_Type = Reaction_Type, steps = [step])
        
        stride = self.Observables['stride'][measure]
        
        if molecules == 'All':
            molecules = [molecule for molecule in self.Observables[sampling][measure] if molecule != 'time']
                
        fig, ax = plt.subplots(figsize=(4,3), dpi=150)
        
        if measure in ['Energy', 'Pressure', 'Virial', 'Number']:
             
            df = pd.DataFrame(self.Observables[sampling][measure])[['time']+molecules]
            dfm = df.melt('time', var_name = 'Molecules', value_name = measure)
            
            g = sns.lineplot(x="time", y=measure, hue='Molecules', data = dfm, palette = self.color_pallete[measure])
    
            plt.xlabel('Time in {}'.format(self.units['Time']))
            
            legend = g.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Molecules', prop=fontLgd)
            plt.setp(legend.get_title(),fontsize='x-small')
            
        elif measure in ['Force', 'Torque', 'Position', 'Orientation']:
            
            sampling = 'stepwise'
            
            ax.text(0.05, 0.95, 'time point: {0:1.1f} {1}'.format(step*stride*self.dt, self.units['Time']), transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=1.0))
            
            df = pd.DataFrame(self.Observables[sampling][measure][molecules[0]][step])
            # df.columns = ['x', 'y', 'z']
            
            g = sns.lineplot(data = df, palette = self.color_pallete[measure])
            
            plt.title('Molecule: {}'.format(molecules[0]))
            plt.xlabel('Molecules')
                
            legend = g.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Dimension', prop=fontLgd)
            plt.setp(legend.get_title(),fontsize='x-small')
            
        elif measure in ['Volume']:
            
            df = pd.DataFrame(self.Observables[sampling][measure])[['time', 'Volume']]
            
            g = sns.lineplot(x="time", y=measure, data = df, palette = self.color_pallete[measure])
    
            plt.xlabel('Time in {}'.format(self.units['Time']))            
            
        elif measure == 'Reactions':
            
            if Reaction_Type in ['bind', 'enzymatic', 'conversion', 'conversion_rb', 'decay_rb', 'production_rb', 'release']:
                df = pd.DataFrame(self.Observables[sampling][measure][Reaction_Type][educt])
                
                plt.title(educt+' | '+Reaction_Type)
                
            elif Reaction_Type in ['fusion', 'enzymatic_rb']:
                df = pd.DataFrame(self.Observables[sampling][measure][Reaction_Type][educt][particle_educt])
                
                educt_1 , educt_2 = educt.split('+')
                particle_educt_1 , particle_educt_2 = particle_educt.split('+')
                plt.title(f'{educt_1}({particle_educt_1})+{educt_2}({particle_educt_2})'+' | '+Reaction_Type)
                
            df['time'] = self.dt*stride*np.arange(int(self.nsteps/stride)+1)
            
            dfm = df.melt('time', var_name = 'paths', value_name = measure)
            
            # g = sns.lineplot(x="time", data = df, palette = self.color_pallete[measure])
            
            g = sns.lineplot(x="time", y=measure, hue='paths', data = dfm, palette = self.color_pallete[measure], linewidth = 1)
            
            
            plt.xlabel('Time in {}'.format(self.units['Time']))
            
            
        elif measure == 'Bonds':
            
            if bond_pairs == 'All':
                bond_pairs = self.Observables[sampling]['Bonds'].keys()
            
            df = pd.DataFrame(self.Observables[sampling][measure])[bond_pairs]
            df['time'] = self.dt*stride*np.arange(int(self.nsteps/stride)+1)
            
            dfm = df.melt('time', var_name = 'bond pairs', value_name = measure)
            
            g = sns.lineplot(x="time", y=measure, hue='bond pairs', data = dfm, palette = self.color_pallete[measure])
    
            plt.xlabel('Time in {}'.format(self.units['Time']))
            
            legend = g.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='bond pairs', prop=fontLgd)
            plt.setp(legend.get_title(),fontsize='x-small')            
            
        
        if sampling == 'stepwise':
            plt.ylabel(measure+' in {}'.format(self.units[measure]))
        elif sampling == 'binned':
            plt.ylabel(measure+' in {0}/({1}{2})'.format(self.units[measure], stride*self.dt, self.units['Time']))
        
        if save_fig == True:
            if fig_name is None:
                fig_name = self.file_name + ''
            if fig_path is None:
                fig_path = self.fig_path
            for frmt in formats:
                if measure == 'Reactions':
                    plt.savefig(fig_path / (fig_name+'_'+measure+'_'+Reaction_Type+f'.{frmt}'), bbox_inches="tight", dpi = 300)
                else:
                    plt.savefig(fig_path / (fig_name+'_'+measure+f'.{frmt}'), bbox_inches="tight", dpi = 300)
            
        if show:
            plt.show()
            
        return fig, ax
    
    #%%
            
    def plot_reactions_graph(self, Simulation, graph_type = 'Bimolecular', graph_subtype = ''):
        
        
        # np.dtype([('defined', bool), ('bond', bool), ('id', np.int64), ('enzyme', np.int64), ('rate', np.float64), ('radius', np.float64), ('cutoff', np.float64), ('type', 'U20'), ('type_id', np.int64), ('type_BP_BM', 'U2'),],  align=True)
        
        self.graph_types = ['Bimolecular', 'Biparticle', 'Unimolecular', 'Interactions']
        
        self.graph_subtypes = ['pair relations', 'product relations']
        
        Nodes = dict()
        Nodes['index'] = []
        Nodes['name'] = []
        Nodes['color'] = []
        N_particles = len(Simulation.System.reaction_args)
        
        Edges = dict()
        Edges['indices'] = []
        Edges['rate'] = []
        Edges['color'] = []
        Edges['label'] = []
        
        Product_Node_id = {}
        
        
        if graph_type == 'Interactions':
 
            for i in range(N_particles):
                for j in range(N_particles-i):
                    j = (N_particles-1)-j
                    
                    if Simulation.System.reaction_args[i][j]['defined']:
                        
                        reaction_id = Simulation.System.reaction_args[i][j]['id']
                        bond = Simulation.System.reaction_args[i,j]['bond']
                        rate = Simulation.System.Reactions_Dict[reaction_id].rate
                        radius = Simulation.System.Reactions_Dict[reaction_id].radius
                        reaction_educt_type = Simulation.System.Reactions_Dict[reaction_id].reaction_educt_type
                        educts = Simulation.System.Reactions_Dict[reaction_id].educts
                                
                        if bond == True:
                            
                            educts_id_i = []
                            educts_id_j = []
                            pname_i = Simulation.System.particle_id_to_name[i]
                            pname_j = Simulation.System.particle_id_to_name[j]
                            
                            for molecule in Simulation.System.molecule_types:
                                   
                                if pname_i in Simulation.System.molecule_types[molecule].types:
                                    educts_id_i.append(Simulation.System.molecule_types[molecule].type_id)
                                    
                                if pname_j in Simulation.System.molecule_types[molecule].types:
                                    educts_id_j.append(Simulation.System.molecule_types[molecule].type_id)

                            educt_pairs = []
                            for ei in educts_id_i:
                                for ej in educts_id_j:
                                    if (ei,ej) not in educt_pairs and (ej,ei) not in educt_pairs:
                                        educt_pairs.append((ei,ej))
                                        
                            for educts in educt_pairs:
                                
                                Nodes['index'].append(int(educts[0]))
                                i_name = Simulation.System.molecule_id_to_name[educts[0]]
                                Nodes['name'].append(i_name)
                                Nodes['color'].append(self.molecules_colors_hex[educts[0]%24])
                                
                                if educts[0]== educts[1]:
                                    educt_1_idx = 96000+int(educts[1])
                                else:
                                    educt_1_idx = int(educts[1])
                                    
                                Nodes['index'].append(educt_1_idx)
                                j_name = Simulation.System.molecule_id_to_name[educts[1]]
                                Nodes['name'].append(j_name)
                                Nodes['color'].append(self.molecules_colors_hex[educts[1]%24])
                                
                                path = Simulation.System.Reactions_Dict[reaction_id].paths[0]
                                    
                                rate = path['rate']
                                reaction_type = path['type']
                                
                                Edges['indices'].append((int(educts[0]),educt_1_idx))
                                Edges['rate'].append(rate)
                                Edges['label'].append('({0},{1})'.format(rate, radius))
                                Edges['color'].append(self.reactions_color[reaction_type])
        
        
        if graph_type == 'Unimolecular':
            
            node_idx = 96000
            for reaction_id in Simulation.System.Reactions_Dict:
                reaction_educt_type = Simulation.System.Reactions_Dict[reaction_id].reaction_educt_type
                bimol = Simulation.System.Reactions_Dict[reaction_id].bimol
                
                if bimol == False and reaction_educt_type == 'Molecule':
                    
                    educts = Simulation.System.Reactions_Dict[reaction_id].educts
                    
                    Nodes['index'].append(int(educts[0]))
                    i_name = Simulation.System.molecule_id_to_name[educts[0]]
                    Nodes['name'].append(i_name)
                    Nodes['color'].append(self.molecules_colors_hex[educts[0]%24])   
                    
                    for path in Simulation.System.Reactions_Dict[reaction_id].paths:
                        
                        n_products = path['n_products']
                        product_ids = path['products_ids'][0:n_products]
                        rate = path['rate']
                        reaction_type = path['type']
                        
                        p_name = ''
                        unique, counts = np.unique(product_ids, return_counts=True)
                        for p_id, n in zip(unique,counts):
                            p_name += '{0}⋅{1}+'.format(n, Simulation.System.molecule_id_to_name[p_id])
                        p_name = p_name[0:-1]
                        
                        Nodes['index'].append(node_idx)
                        Nodes['name'].append(p_name)
                        Nodes['color'].append(self.molecules_colors_hex[unique[0]%24])
                                    
                        Edges['indices'].append((int(educts[0]),node_idx))
                        Edges['rate'].append(rate)  
                        Edges['label'].append('{0}'.format(rate))           
                        Edges['color'].append(self.reactions_color[reaction_type])
                        
                        node_idx += 1
        
        if graph_type == 'Bimolecular':
            
            if graph_subtype == '':
                graph_subtype = 'product_relations'
                
            if graph_subtype == 'educt_relations':
                
                for i in range(N_particles):
                    for j in range(N_particles-i):
                        j = (N_particles-1)-j
                        
                        if Simulation.System.reaction_args[i][j]['defined']:
                            
                            reaction_id = Simulation.System.reaction_args[i][j]['id']
                            bond = Simulation.System.reaction_args[i,j]['bond']
                            rate = Simulation.System.Reactions_Dict[reaction_id].rate
                            radius = Simulation.System.Reactions_Dict[reaction_id].radius
                            reaction_educt_type = Simulation.System.Reactions_Dict[reaction_id].reaction_educt_type
                            educts = Simulation.System.Reactions_Dict[reaction_id].educts
                            
                            if reaction_educt_type == 'Molecule':
                                
                                Nodes['index'].append(int(educts[0]))
                                i_name = Simulation.System.molecule_id_to_name[educts[0]]
                                Nodes['name'].append(i_name)
                                Nodes['color'].append(self.molecules_colors_hex[educts[0]%24])
                                
                                if educts[0] == educts[1]:
                                    educt_1_idx = 96000+int(educts[1])
                                else:
                                    educt_1_idx = int(educts[1])
                                    
                                Nodes['index'].append(educt_1_idx)
                                j_name = Simulation.System.molecule_id_to_name[educts[1]]
                                Nodes['name'].append(j_name)
                                Nodes['color'].append(self.molecules_colors_hex[educts[1]%24])
                                
                                for path in Simulation.System.Reactions_Dict[reaction_id].paths:
                                    
                                    rate = path['rate']
                                    reaction_type = path['type']
                                    
                                    # print(i_name, j_name)
                                    Edges['indices'].append((int(educts[0]),educt_1_idx))
                                    Edges['rate'].append(rate)
                                    Edges['label'].append('({0},{1})'.format(rate, radius))
                                    Edges['color'].append(self.reactions_color[reaction_type])

                                    
                                    
                # print(Edges)
                                    
            elif graph_subtype == 'product_relations':
                
                prod_node_idx = 96000
                for i in range(N_particles):
                    for j in range(N_particles-i):
                        j = (N_particles-1)-j
                        
                        if Simulation.System.reaction_args[i][j]['defined']:
                            
                            reaction_id = Simulation.System.reaction_args[i][j]['id']
                            rate = Simulation.System.Reactions_Dict[reaction_id].rate
                            radius = Simulation.System.Reactions_Dict[reaction_id].radius
                            reaction_educt_type = Simulation.System.Reactions_Dict[reaction_id].reaction_educt_type
                            educts = Simulation.System.Reactions_Dict[reaction_id].educts
                            
                            if reaction_educt_type == 'Molecule':
                                
                                node_idx = int(unique_pairing(educts[0],educts[1]))
                                Nodes['index'].append(int(node_idx))
                                i_name = Simulation.System.molecule_id_to_name[educts[0]]
                                j_name = Simulation.System.molecule_id_to_name[educts[1]]
                                Nodes['name'].append(i_name + ' + ' + j_name)
                                Nodes['color'].append(self.molecules_colors_hex[educts[0]%24])
                                
                                p0 = node_idx
                                # node_idx += 1
                                
                                for path in Simulation.System.Reactions_Dict[reaction_id].paths:
                                    
                                    n_products = path['n_products']
                                    product_ids = path['products_ids'][0:n_products]
                                    rate = path['rate']
                                    reaction_type = path['type']
                                    
                                    p_name = ''
                                    for p_id in product_ids:
                                        p_name += Simulation.System.molecule_id_to_name[p_id]+'+'
                                    p_name = p_name[0:-1]
                                    
                                    if p_name in Product_Node_id:
                                        p1 = Product_Node_id[p_name]
                                    else:
                                        Product_Node_id[p_name] = int(prod_node_idx)
                                        p1 = int(prod_node_idx)
                                        prod_node_idx += 1
                                    
                                    Nodes['index'].append(p1)
                                    Nodes['name'].append(p_name)
                                    Nodes['color'].append(self.molecules_colors_hex[p_id%24])
                                    
                                    Edges['indices'].append((p0,p1))
                                    Edges['rate'].append(rate)  
                                    Edges['label'].append('({0},{1})'.format(rate, radius))
                                    Edges['color'].append(self.reactions_color[reaction_type])
                                    
                                    
                                    
        # print(Edges)
                
                
                
        
        #%%
        
        if graph_type == 'Interactions':
            arrow_type = "to from"
        elif graph_type == 'Unimolecular':
            arrow_type = "to"
        elif graph_type == 'Bimolecular':
            if graph_subtype == 'product_relations':
                arrow_type = "to"
            if graph_subtype == 'educt_relations':
                arrow_type = "to from"
            
        # g.add_node('A', shape = 'ellipse', opacity = 0.5, fixed = True, font = '40px arial black', heightConstraint = 100, widthConstraint = 100, margin = 0)
        # g.add_node('B', opacity = 0.5)
        # g.add_edge('A','B', arrows = "to from", dashes = True, length = 200, shadow = True, smooth = True)
        # g.add_edge('B','A', arrows = "to from", dashes = True, length = 200)
        # g.show('tmp.html')
        
        # g = Network(height="100%", width="100%", bgcolor="#222222", font_color="white", directed = True)
        g = Network(height="70%", width="50%", bgcolor="white", font_color="black", directed = True)
        
        for i in range(len(Nodes['index'])):
            g.add_node(Nodes['index'][i], title = Nodes['name'][i], label = Nodes['name'][i], shape = 'box', color = Nodes['color'][i], opacity = 1.0, font = '20px arial black')#, heightConstraint = 50, widthConstraint = 50)
            
        # g.add_nodes(list(Nodes['index']), title = list(Nodes['name']), label = list(Nodes['name']), shape=['box']*len(Nodes['index']), color = Nodes['color'], size = [10]*len(Nodes['index']))
        
        #, weight = Edges['rate'][i]/max(Edges['rate']), value = Edges['rate'][i]/max(Edges['rate'])
        
        for i in range(len(Edges['indices'])):
            
            g.add_edge(Edges['indices'][i][0],Edges['indices'][i][1], title=Edges['rate'][i], label=str(Edges['label'][i]), color = Edges['color'][i], arrows = arrow_type, dashes = False, shadow = True, smooth = True)
            
            # g.add_edge(Edges['indices'][i][0],Edges['indices'][i][1], title=Edges['rate'][i], label=str(Edges['label'][i]), color = Edges['color'][i], arrows = "to from")
        
        # g.toggle_physics(False)
        
        # g.show_buttons(filter_=['edges', 'nodes', 'physics'])
        
        g.set_edge_smooth("dynamic")
        g.barnes_hut(gravity=-3000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.05,
        damping=0.5,
        overlap=0)
        
        try:
            os.makedirs(self.fig_path / 'Graphs') 
        except FileExistsError:
            # directory already exists
            pass
            
        # g.set_options('{"layout": {"randomSeed":0}}')
        g.show(str(self.fig_path / 'Graphs' / (self.file_name+'_'+graph_type+'_'+graph_subtype+'.html')))
        
#%%

# if __name__ == '__main__':
    
