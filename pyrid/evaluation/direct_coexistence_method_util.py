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
    
    """Centers the molecule density profile.
    
    Parameters
    ----------
    Histogram0 : `float64[:]`
        Density profile.
    box_lengths : `float64[3]`
        Simulation box lengths
    axes : 0, 1 or 2
        Coordinate axis corresponding to the density profile (axis along which the profile has been sampled).
    cells_axes : `int64`
        Number of cells / bins along the chosen axis.

    
    Returns
    -------
    float64
        Value by which to shift the density profile such that it is centered.
    
    """
    
    center = np.sum(np.linspace(0,box_lengths[axes],cells_axes)*Histograms0)/np.sum(Histograms0)
    Dx = box_lengths[axes]/2-center
    return Dx
    
    
    

def calc_profile(path, moltype, box_lengths, axes, cells_axes, section):
    
    """Calculates the density profile of a molecule population along a given coordinate axis.
    
    Parameters
    ----------
    path : `string`
        directory of the hdf5 file.
    moltype : `string`
        Molecule type
    box_lengths : `float64[3]`
        Simulation box lengths
    axes : 0, 1 or 2
        Coordinate axis along which to calculate the density profile.
    cells_axes : `int64`
        Number of cells /  bins by which to divide the simulation box along the chosen axis.
    section : `int64[2]`
        Time interval (section) over which the density profile is averaged.
    
    
    Returns
    -------
    float64[:]
        Density profile / histogram
    
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
    
    """Calculates the phase diagram from a given density profile by identifying the dense and the dilute phase.
    
    Parameters
    ----------
    Histogram : `float64[:]`
        Density profile.
    cutoff : `float64`
        Fraction of the maximum density at which to distinguish between dilute and dense phase.
    
    
    Returns
    -------
    tuple(float64, float64)
        Volume fractions of the dense and the dilute phase.
    
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
    
    """The hyperbolic tangent function can be used to fit the density profile.
    
    Parameters
    ----------
    z : `float64[:]`
        Location / Distance values along z-axis.
    z0 : `float64`
        Location shift.
    d : `float64`
        fitting parameter
    dense: `float64`
        Volume fraction dense phase.
    dilute : `float64`
        Volume fraction dilute phase.
    
    
    Returns
    -------
    float64[:]
        Fit of the density profile.
    
    """
    
    return (dense+dilute)/2-(dense-dilute)/2*np.tanh((z-z0)/d)
    
#%%

# z = np.linspace(0,0.2,100)
# rho = density_hyperbolic_tangent(z,0.1,0.01, 0.4,0.001)

# plt.figure()
# plt.plot(z,rho)

#%%

def critical_Temp(eps_csw, d, eps_c, x):
    
    """Equation to estimate the critical temperature (inverse interaction strength). critical_Temp() is called by critical_point_fit() for fitting. Based on: Silmore 2017, "Vapour–liquid phase equilibrium and surface tension of fully flexible Lennard–Jones chains"
    
    Parameters
    ----------
    eps_csw : `float64`
        Interaction energy constant
    d : `float64`
        fitting parameter
    eps_c : `float64`
        Critical interaction energy constant of the highest-valency molecule (interaction energy at the critical point where the two phase regime ends).
    
    
    Returns
    -------
    float64
        Critical temperature (inverse interaction strength)
    
    """
    
    
    # kbt = 2.4
    # return d*(1-(kbt/eps_csw)/Tc)

    return d*(1-(eps_c/eps_csw))**(0.325)#(1/8.0)


def critical_Density(eps_csw, s2, phi_c):
    
    """Equation to estimate the critical density / volume fraction. critical_Density() is called by critical_point_fit() for fitting. Based on: Silmore 2017, "Vapour–liquid phase equilibrium and surface tension of fully flexible Lennard–Jones chains"
    
    Parameters
    ----------
    eps_csw : `float64`
        Interaction energy constant
    s2 : `float64`
        fitting parameter
    phi_c : `float64`
        Volume fraction of the condensed (dense) phase.

    
    Returns
    -------
    float64
        Critical density.
    
    """
    
    
    # kbt = 2.4
    # return d*(1-(kbt/eps_csw)/Tc)
    # eps_c = Esp_popt[geometry][1]
    return phi_c+s2*(1/critical_Density.eps_c-1/eps_csw)


def critical_point_fit(pp_strength, dense, dilute):
    
    """Estimates the critical temperature (inverse interaction strength) and density / volume fraction from a selection of phase diagram points.
    
    Parameters
    ----------
    pp_strength : `float64[:]`
        List of particle-particle interaction strengths.
    dense : `float64[:]`
        Volume fractions of the dense phase (condensate) corresponding to particle-particle interaction strengths kept in pp_strength.
    dilute : `float64[:]`
        Volume fractions of the dilute phase corresponding to particle-particle interaction strengths kept in pp_strength.
    
    
    Returns
    -------
    tuple(float64, float64)
        Estimates for the critical temperature (inverse interaction strength) and the critical density (volume fraction).
    
    """
    
    popt_temp, pcov_temp = curve_fit(critical_Temp, np.array(pp_strength), (np.array(dense)-np.array(dilute)), p0=(1.0,10.0, 1.0), bounds=([0.0,0.0, 1.0],[np.inf,np.inf,np.inf]), maxfev=1000)
    
    critical_Density.eps_c = popt_temp[1]
    
    popt_density, pcov_density = curve_fit(critical_Density, np.array(pp_strength), (np.array(dense)+np.array(dilute))/2.0, p0=(1.0,10.0), bounds=([0.0,0.0],[ np.inf,np.inf]), maxfev=1000)
    
    return popt_temp, popt_density


#%%

# if __name__ == '__main__':
    
