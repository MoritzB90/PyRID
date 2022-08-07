# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numpy as np
import numba as nb


@nb.njit
def wall(r, d, k):
    
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
    
    force=-k*(r-d)
    u=k/2*(r-d)**2
        
    return u, force


@nb.njit
def harmonic_repulsion(r, args):
    
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
    
    d=args[0]
    k=args[1]
    
    if r<d:
        force=-k*(r-d)
        u=k/2*(r-d)**2
    else:
        force=0.0
        u=0.0
        
    return u, force

setattr(harmonic_repulsion, 'name', 'harmonic_repulsion')


@nb.njit
def piecewise_harmonic(r, args):
    
    """Calculates potential energy and force for a weak piecewise harmonic potential.
    
    Parameters
    ----------
    rc : `float`
        cutoff radius
    k : `float`
        force constant (units of kJ/length_unit^2).
    h : `float`
        depth of potential well (units of kJ).
    d : `float`
        radius of the repulsive part!
            

    Notes
    -----
    
    The weak piecewise harmonic interaction potential is also used in the particle-based reaction-diffusion simulator `ReaDDy <https://readdy.github.io/system.html#potentials>`_ :cite:t:`Hoffmann2019` as it is well suited for brownian dynamics simulations, which allow larger time steps than conventional molecular dynamics simualtions, however forces acting on the particles must not change too much in bewteen two integration step. Therefore, soft potentials are used. Hard potentials like the PHS may also be used. In this case, however, the integration time step needs to be chosen much smaller, reducing the benefit of simulating overdamped langevin dynamics to a degree!
    
    .. math::
    
       U_{ha}(r)
       = 
        \\begin{cases}
            \\frac{1}{2}k(r-(d_1+d_2))^2-h,& \\text{if } r<(d_1+d_2), \\\\
            \\frac{h}{2}(\\frac{r_c-(d_1+d_2)}{2})^{-2}(r-(d_1+d_2))^2-h,& \\text{if } d \le r < d + \\frac{r_c-(d_1+d_2)}{2}, \\\\
            -\\frac{h}{2}(\\frac{r_c-(d_1+d_2)}{2})^{-2}(r-r_c)^2,& \\text{if } d + \\frac{r_c-(d_1+d_2)}{2} \le r < r_c, \\\\
            0,              & \\text{otherwise}
        \\end{cases}
    

    Returns
    -------
    (float64, float64)
        returns the potential energy and force of the interaction
    
    """

    rc=args[0]   
    k=args[1]
    h=args[2]
    d=args[3]
    
    if (r<d):
        force=-k*(r-d)
        u=k/2*(r-d)**2-h
    elif ((d<=r)&(r<d+(rc-d)/2)):
        force=-h*((rc-d)/2)**(-2)*(r-d) 
        u=h/2*((rc-d)/2)**(-2)*(r-d)**2-h
    elif ((d+(rc-d)/2<=r)&(r<rc)):
        force=h*((rc-d)/2)**(-2)*(r-rc)
        u=-h/2*((rc-d)/2)**(-2)*(r-rc)**2
        
    return u, force

setattr(piecewise_harmonic, 'name', 'piecewise_harmonic')

@nb.njit
def screened_electrostatics(r, args):
    
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
    
    return 0.0, 0.0

@nb.njit
def Lennard_Jones(r, args):
    
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
  
    epsilon=args[1]
    sigma=args[2]
    exp_large=args[3]   
    exp_small=args[4]
        
    term_1 = (sigma / r) ** exp_large
    term_2 = (sigma / r) ** exp_small

    u = 4*epsilon * (term_1 - term_2)
    force = 4*epsilon * (exp_large * term_1 - exp_small * term_2) / r

    return u, force


@nb.njit
def FENE(r, args):
    
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
    
    return 0.0, 0.0

@nb.njit
def Weeks_Chandler_Anderson(r, args):
    
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
    
    return 0.0, 0.0
    
@nb.njit
def repulsive_membrane(r, args):
    
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
    
    return 0.0, 0.0
    
@nb.njit 
def CSW(r, args):
    
    """Calculates potential energy and force for a Continuous Square-Well (CSW) potential.
    
    Parameters
    ----------
    rw : `float`
        Radius of the attractive well
    eps_cw : `float`
        Interaction energy constant (units of kJ).
    alpha : `float`
        Steepness of the attractive well (the larger alpha, the better the approximation to a square well).
    
    
    Notes
    -----
    The CSW potential has been introduced in :cite:t:`Espinosa2014` and is a continuous approximation of the square-well potential and thereby suited for moelcular dynamics simulations :cite:t:`Espinosa2019`.
    
    .. math::
    
       U_{CSW}(r) = - \\frac{\epsilon_{CSW}}{2} \Big[1 - \\tanh\Big(\\frac{r-r_w}{\\alpha}\Big)\Big]
    
    Returns
    -------
    (float64, float64)
        returns the potential energy and force of the interaction
    
    """
    
    rw = args[0] #cutoff
    eps_csw = args[1] #depth
    alpha = args[2] #slope
    
    
    u = -1/2*eps_csw*(1-np.tanh((r-rw)/alpha))
    force = -eps_csw/(2*alpha*np.cosh((rw-r)/alpha)**2)
    
    return u, force

setattr(CSW, 'name', 'CSW')



@nb.njit 
def PHS(r, args):
    
    """Calculates potential energy and force for a Pseudo Hard Sphere (PHS) potential.
    
    Parameters
    ----------
    EpsR : `float`
        Interaction energy constant (units of kJ).
    lr : `float`
        Exponent of the repulsive term.
    la : `float`
        Exponent of the attractive term.
    
    Notes
    -----
    The Pseudo Hard Sphere interaction potential is a continuous approximation of the hard sphere and thereby suited for molecular dynamics simulations :cite:t:`Jover2012`, :cite:t:`Espinosa2019`.
    
    .. math::
    
       U_{HS}
       =
       \\Biggl \\lbrace 
       { 
       \lambda_r (\\frac{\lambda_r}{\lambda_a})^{\lambda_a} \epsilon_R [(\\frac{\sigma}{r})^{\lambda_r}-(\\frac{\sigma}{r})^{\lambda_a}]+\epsilon_R,\\text{ if } 
          { r < (\\frac{\lambda_r}{\lambda_a}) \sigma }
       \\atop 
       0, \\text{ if } 
          { r < (\\frac{\lambda_r}{\lambda_a}) \sigma }
       }
    
    Returns
    -------
    (float64, float64)
        returns the potential energy and force of the interaction
    
    """
    
    sigma = args[0] #cutoff
    EpsR = args[1] #depth
    lr = args[2] #slope
    la = args[3] #slope
    
    if r<(lr/la)*sigma:
        E = lr*(lr/la)**la*EpsR
        exp_r = (sigma/r)**lr
        exp_a = (sigma/r)**la
        
        u = E*(exp_r-exp_a)+EpsR
        force = E*(lr*exp_r-la*exp_a)/r
        
    else:
        u = 0.0
        force = 0.0
        
    
    return u, force

setattr(PHS, 'name', 'PHS')


def WLC_force(z, Lp, L0):
    
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
    
    kbt = 1
    F = kbt/(4*Lp)*(1/(1-z/L0)**2-1+4*z/L0)
    return F


#%%

@nb.njit
def execute_force(ID, r, args):
    
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
    
    if ID==0:
        return harmonic_repulsion(r, args)
    elif ID==1:
        return piecewise_harmonic(r, args)
    elif ID==2:
        return screened_electrostatics(r, args)
    elif ID==3:
        return Lennard_Jones(r, args)
    elif ID==4:
        return Weeks_Chandler_Anderson(r, args)
    elif ID==5:
        return repulsive_membrane(r, args)
    elif ID==6:
        return CSW(r, args)
    elif ID==7:
        return PHS(r, args)