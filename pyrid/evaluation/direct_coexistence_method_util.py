# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
fontLgd = FontProperties()
fontLgd.set_size('x-small')
import seaborn as sns
import h5py
from scipy.optimize import curve_fit


#%%

def center_profile(Histograms0, box_lengths, axes,cells_axes):
    
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
    
    center = np.sum(np.linspace(0,box_lengths[axes],cells_axes)*Histograms0)/np.sum(Histograms0)
    Dx = box_lengths[axes]/2-center
    return Dx
    
    
    

def calc_profile(path, moltype, box_lengths, axes, cells_axes, section):
    
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
        
    hdf = h5py.File(path, 'r', track_order=True)
    
    Vol_mol = np.float64(hdf['Molecules'][moltype]['volume'])
    
    time_points = list(hdf['Position'][moltype])
            
    cell_length_axes = box_lengths[axes]/cells_axes
        
    Histograms = np.zeros(cells_axes)
    for j in range(section[0],section[1]):
        
        Positions = np.array(hdf['Position'][moltype][str(time_points[j])])
        
        # axes_name = ['X', 'Y', 'Z']
        ax123 = set([0,1,2])
        ax123.remove(axes)
        ax123 = list(ax123)
        Vol_cell = cell_length_axes*box_lengths[ax123[0]]*box_lengths[ax123[1]]
        
        Histograms0 = np.zeros(cells_axes)
        for pos in Positions:
            
            i = int((pos[axes]+box_lengths[axes]/2) / cell_length_axes)
            
            Histograms0[i] += Vol_mol
        
        
        Histograms0/=Vol_cell
        
        Dx = center_profile(Histograms0, box_lengths, axes,cells_axes)
        
        Histograms0 = np.zeros(cells_axes)
        for pos in Positions:
            
            i = int((pos[axes]+Dx+box_lengths[axes]/2) / cell_length_axes)
            
            if i>=0 and i<len(Histograms):
                Histograms0[i] += Vol_mol
        
        
        Histograms0/=Vol_cell        
        
        Histograms += Histograms0
        
    Histograms/=len(range(section[0],section[1]))
    
    hdf.close() 
    
    return Histograms

def calc_phase_diagram(Histograms, cutoff):
    
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
    
    ii_dense = np.where(Histograms>=np.max(Histograms)*cutoff[0])
    ii_dilute = np.where(Histograms<np.max(Histograms)*cutoff[1])
    
    print('dense phase: ', np.mean(Histograms[ii_dense]))
    print('diliute phase: ', np.mean(Histograms[ii_dilute]))
    
    # dense = np.mean(Histograms[ii_dense])
    # dilute = np.mean(Histograms[ii_dilute])
    
    lenH = len(Histograms)
    center = int(lenH/2)
    
    dense = np.mean(Histograms[int(center-lenH/7):int(center+lenH/7)])
    dilute = (np.mean(Histograms[0:int(lenH/5)])+np.mean(Histograms[-int(lenH/5):]))/2
    
    return dense, dilute


def density_hyperbolic_tangent(z,z0,d, dense, dilute):
    
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
    
    return (dense+dilute)/2-(dense-dilute)/2*np.tanh((z-z0)/d)
    
#%%

# z = np.linspace(0,0.2,100)
# rho = density_hyperbolic_tangent(z,0.1,0.01, 0.4,0.001)

# plt.figure()
# plt.plot(z,rho)

#%%

def critical_Temp(eps_csw, d, eps_c, x):
    
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
    
    '''
    Based on: Silmore 2017, "Vapour–liquid phase equilibrium and surface tension of fully flexible Lennard–Jones chains"
    '''
    
    # kbt = 2.4
    # return d*(1-(kbt/eps_csw)/Tc)

    return d*(1-(eps_c/eps_csw))**(0.325)#(1/8.0)


def critical_Density(eps_csw, s2, phi_c):
    
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
    
    '''
    Based on: Silmore 2017, "Vapour–liquid phase equilibrium and surface tension of fully flexible Lennard–Jones chains"
    '''
    
    # kbt = 2.4
    # return d*(1-(kbt/eps_csw)/Tc)
    # eps_c = Esp_popt[geometry][1]
    return phi_c+s2*(1/critical_Density.eps_c-1/eps_csw)


def critical_point_fit(pp_strength, dense, dilute):
    
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
    
    popt_temp, pcov_temp = curve_fit(critical_Temp, np.array(pp_strength), (np.array(dense)-np.array(dilute)), p0=(1.0,10.0, 1.0), bounds=([0.0,0.0, 1.0],[np.inf,np.inf,np.inf]), maxfev=1000)
    
    critical_Density.eps_c = popt_temp[1]
    
    popt_density, pcov_density = curve_fit(critical_Density, np.array(pp_strength), (np.array(dense)+np.array(dilute))/2.0, p0=(1.0,10.0), bounds=([0.0,0.0],[ np.inf,np.inf]), maxfev=1000)
    
    return popt_temp, popt_density


#%%

# if __name__ == '__main__':
    
