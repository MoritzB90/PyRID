# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""


import numpy as np
from scipy.optimize import root
import warnings

#%%

def k_macro(D1, D2, k_micro, R_react):
    
    """
    Calculates the macroscopic reaction rate for a bimolecular reaction from the educts diffusion coefficients, the reaction radius and the microscopic reaction rate.
    
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
    
    return 4*np.pi*(D1+D2)*(R_react-np.sqrt((D1+D2)/k_micro)*np.tanh(R_react*np.sqrt(k_micro/(D1+D2))))

#%%

def k_micro(System, mol_type_1, mol_type_2, k_macro, R_react, loc_type = 'Volume'):
    
    """
    Calculates the microscopic reaction rate for a bimolecular reaction given the educts diffusion coefficients, the reaction radius and the macroscopic reaction rate.
    
    Parameters
    ----------
    System : `object`
        Instance of System class.
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
    The method used here is only valid for volume moelcules, i.e. diffusion in 3D. For surface molecules a value is also returned, however with a warning.
    
    Returns
    -------
    `float64`
        Microscopic reaction rate
    """
    
    if loc_type == 'Surface':
        D1 = System.molecule_types[mol_type_1].Dtrans_2D
        D2 = System.molecule_types[mol_type_2].Dtrans_2D
        
        print('The estimate of the microscopic reaction rate from the macroscopic reaction rate and the reaction radius is only precise for volume molecules. While PyRID allows users to get an estimate for the 2D case, please be aware that the result may not be accurate!')
        
    elif loc_type == 'Volume':
        D1 = System.molecule_types[mol_type_1].Dtrans
        D2 = System.molecule_types[mol_type_2].Dtrans
        
    def k_macro_root(x, D1, D2, k_macro, R_react):
    
        return 4*np.pi*(D1+D2)*(R_react-np.sqrt((D1+D2)/abs(x[0]))*np.tanh(R_react*np.sqrt(abs(x[0])/(D1+D2))))-k_macro
    
    lamb = np.sqrt(4*(D1+D2)*System.dt) # Diffusion length constant
    if R_react < 10*lamb:
        warnings.warn('Warning: The reaction radius is very close to the diffusion length constan for the molecule pair {0}, {1}. Thereby, many bi-molecular reactions could get overlooked! I recommend to choose a reaction radius that is above {2:.3g}'.format(mol_type_1, mol_type_2, 10*lamb))
    
    # R_react at k_micro -> oo
    R_react_infty = k_macro/(4*np.pi*(D1+D2))
    if R_react < R_react_infty:
        warnings.warn('Warning: The reaction radius is too small to be able to match the macroscopic reaction rate! The minimum reaction radius for an infinite microscopic reaction rate that is in agreement with the given macroscopic reaction rate is {0:.3g}'.format(R_react_infty))
        
    k_micro_0 = k_macro/(4/3*np.pi*R_react**3)
    k_micro_sol = root(k_macro_root, [k_micro_0], args = (D1, D2, k_macro, R_react,), method='lm')

    if k_micro_sol.x[0]*System.dt>0.1:
        warnings.warn('Warning: The reaction rate for the bi-molecule reaction of the educts {0}, {1} is larger 1/(10*dt). k*dt = {2:.3g}. The discretization errors may become too large! Please increase the reaction radius or lower the macroscopic reaction rate or the integration time step!'.format(mol_type_1, mol_type_2, k_micro_sol.x[0]*System.dt))
    
    
    return abs(k_micro_sol.x[0])