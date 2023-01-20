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

plt.rcParams.update({'font.size': 14})

#%%

# ------------------------------------
# MSD
# ------------------------------------


@nb.njit
def calc_MSD_population(time_steps,position):
    
    """Calculates the Mean Squared Distance (MSD) from the position data (one dimension x, y or z) of a molecule population. 
    
    Parameters
    ----------
    time_steps : `int64[:]`
        Time steps.
    position : `float64[t,N]`
        Position vectors for t timesteps of a molecule population of size N (one dimension x, y or z).
    
    Returns
    -------
    `list`
        MSD in one dimension (x, y or z).
    
    """
    
    MSD=nb.typed.List()
    for t in time_steps:
        # if t%100==0:
        #     print(t)
        MSD_t0=0
        count = 0
        for i in np.arange(0,len(position[:,0])-t,1):
            MSD_t0 += np.mean((position[i+t,:]-position[i,:])**2)
            count += 1
        
        MSD.append(MSD_t0/count)
        
    return MSD    


#%%

def MSD(position_trace, Delta_t, time_interval, stride, molecule):
    
    """Calculates the MSD for each dimension (x,y, and z) of a molecule population.
    
    Parameters
    ----------
    position_trace : `float64[t,N,3]`
        Position data for a molecule population of size N at t different time steps.
    Delta_t : `int64`
        Time step.
    time_interaval : `int64`
        The time step until which the molecule positions are sampled for the MSD calculation. Note: Currently sampling always starts at time step 0.
    stride : `int64`
        Stride with which the molecule positions are sampled.
    molecule : `string`
        Name of the molecule type.
    
    Returns
    -------
    tuple(list(float64[:],float64[:],float64[:]), float64[:])
        MSD in each dimension and the corresponding time vector.
    
    """


    time_steps=np.arange(0,time_interval, stride)
    
    pos_x = position_trace[:,:,0]
    pos_y = position_trace[:,:,1]
    pos_z = position_trace[:,:,2]
    
    MSD_x = np.array(calc_MSD_population(time_steps, pos_x))
    MSD_y = np.array(calc_MSD_population(time_steps, pos_y))
    MSD_z = np.array(calc_MSD_population(time_steps, pos_z))
    
            
    return [MSD_x, MSD_y, MSD_z], time_steps*Delta_t

#%%

# ------------------------------------
# P2
# ------------------------------------

@nb.njit
def calcP2_anisotropic(time_steps,orientation_trace):
    
    """Calculates the rotational time correlation function P2 regarding one rotation axis from the orientations of a molecule population.
    
    Parameters
    ----------
    time_steps : `int64[:]`
        Time steps.
    orientation_trace : `float64[t,N]`
        Orientations regarding one axis of a molecule population of size N at t different time steps.
    

    Returns
    -------
    `float64[:]`
        P2 regarding one rotation axis.
    
    """
    
    # <P2(t)> = 3/2 <u(t0)*u(t0+t)>_t0 - 1/2
    
    stride = 1#int(len(orientation_trace)/100)
    
    P2=np.zeros(len(time_steps), dtype = np.float64)
    for j,t in enumerate(time_steps):
        # if t%100==0:
        #     print(t)
        P2_t0=0
        count = 0
        for i in np.arange(0,len(orientation_trace)-t,stride):
            P2_t0 += np.dot(orientation_trace[i+t], orientation_trace[i])**2
            count += 1
        
        P2[j] = 3/2*P2_t0/count-1/2
        
    return P2



@nb.njit
def calc_orientation_quat(q):
    
    """Calculates the rotation matrix from a rotation / orientation quaternion.
    
    Parameters
    ----------
    q : `float64[4]`
        Rotation quaternion.
    
    
    Returns
    -------
    `float64[3,3]`
        Rotation / Orientation matrix.
    
    """
        
    orientation_quat =np.zeros((3,3))

    orientation_quat[0][0] = 2*(q[0]**2+q[1]**2)-1
    orientation_quat[0][1] = 2*(q[1]*q[2]-q[0]*q[3])
    orientation_quat[0][2] = 2*(q[1]*q[3]+q[0]*q[2])
    
    orientation_quat[1][0] = 2*(q[1]*q[2]+q[0]*q[3])
    orientation_quat[1][1] = 2*(q[0]**2+q[2]**2)-1
    orientation_quat[1][2] = 2*(q[2]*q[3]-q[0]*q[1])
    
    orientation_quat[2][0] = 2*(q[1]*q[3]-q[0]*q[2])
    orientation_quat[2][1] = 2*(q[2]*q[3]+q[0]*q[1])
    orientation_quat[2][2] = 2*(q[0]**2+q[3]**2)-1
    
    return orientation_quat
        


@nb.njit
def calcP2_anisotropic_Population(time_interval, stride, orientation_trace):
    
    """Calculates the rotational time correlation function P2 around each of the three rotation axis of a molecule population.
    
    Parameters
    ----------
    time_interaval : `int64`
        Target time step. Note: Currently sampling always starts at time step 0.
    stride : `int64`
        Stride with which the molecule orientations are sampled.
    orientation_trace : `float64[t,N,4]`
        Orientations (in quaternion representation) of a molecule population of size N at t different time steps.
    
    
    Returns
    -------
    tuple(float64[:], float64[:], float64[:])
        P2 for each rotation axis.
    
    """
    
    time_steps = np.arange(0,time_interval, stride)
    
    P2_1 = np.zeros(len(time_steps), dtype = np.float64)
    P2_2 = np.zeros(len(time_steps), dtype = np.float64)
    P2_3 = np.zeros(len(time_steps), dtype = np.float64)
    
    ni = orientation_trace.shape[1]
    for i in range(ni):
    
        orientation_1 = np.zeros((orientation_trace.shape[0],3), dtype = np.float64)
        orientation_2 = np.zeros((orientation_trace.shape[0],3), dtype = np.float64)
        orientation_3 = np.zeros((orientation_trace.shape[0],3), dtype = np.float64)
        for j,q in enumerate(orientation_trace[:,i,:]):
            orientation_quat = calc_orientation_quat(q)
        
            ex =np.array([1.0,0.0,0.0])
            orientation_1[j][:] = np.dot(orientation_quat,ex)
            
            ey =np.array([0.0,1.0,0.0])
            orientation_2[j][:] = np.dot(orientation_quat,ey)
            
            ez =np.array([0.0,0.0,1.0])
            orientation_3[j][:] = np.dot(orientation_quat,ez)
        
        P2_1+= calcP2_anisotropic(time_steps,orientation_1)
        P2_2+= calcP2_anisotropic(time_steps,orientation_2)
        P2_3+= calcP2_anisotropic(time_steps,orientation_3)
    
    P2_1 /= ni
    P2_2 /= ni
    P2_3 /= ni
    
    return P2_1, P2_2, P2_3


#%%

def calc_A_B(u, Drot, D_rr, Delta):
    
    """Calculates the parameters A (F) and B (G) used in the theoretical prediction of the rotational time correlation function P2.
    
    Parameters
    ----------
    u : `float64[3]`
        Basis vector.
    Drot : `float64`
        Rotational diffusion constant.
    D_rr : `float64[3,3]`
        Rotational diffusion tensor.
    Delta : `float64`
        Delta.
    
    
    Returns
    -------
    tuple(float64, float64)
        A (F) and B (G).
    
    """
    
    Sum = 0
    range_3 = np.array([0,1,2])
    for alpha in range(3):
        
        beta = range_3[alpha-2]
        gamma = range_3[alpha-1]
        Sum += D_rr[alpha][alpha]*(u[alpha]**4+2*(u[beta]*u[gamma])**2)
        
    A = -1/3+np.sum(u**4)
    B = 1/Delta*(-Drot+Sum)
    
    return A, B

def calc_P_theory(t, u, D_rr):
    
    """Theoretical prediction of the rotational time correlation function P2.
    
    Parameters
    ----------
    t : `float64[:]`
        Time vector
    u : `float64[3]`
        Basis vector
    D_rr : `float64[3,3]`
        Rotational diffusion tensor.
    
    
    Returns
    -------
    float64[:]
        P2 (Theory)
    
    """
    
    """
    reference: Torre J G et al. 1999, "Calculation of NMR relaxation, covolume, and scattering-related properties of bead models using the SOLPRO computer program", Eur Biophys J.
    """
    
    Drot = np.trace(D_rr)/3
    
    w, v = np.linalg.eig(D_rr)
    
    D_1 = w[0]
    D_2 = w[1]
    D_3 = w[2]
    Delta = np.sqrt((D_1-D_2)**2+(D_3-D_2)*(D_3-D_1))
    
    Ti = np.zeros(5)
    
    Ti[0] = 1/(6*Drot-2*Delta)
    Ti[1] = 1/(3*(Drot+D_1))
    Ti[2] = 1/(3*(Drot+D_2))
    Ti[3] = 1/(3*(Drot+D_3))
    Ti[4] = 1/(6*Drot+2*Delta)
    
    A,B = calc_A_B(u, Drot, D_rr, Delta)
    
    ai = np.zeros(5)
    
    ai[0] = 3/4*(A+B)
    ai[1] = 3*(u[1]*u[2])**2
    ai[2] = 3*(u[0]*u[2])**2
    ai[3] = 3*(u[0]*u[1])**2
    ai[4] = 3/4*(A-B)
    
    P = 0
    for i in range(5):
        P += ai[i]*np.exp(-t/Ti[i])
        
    return P

#%%

def P2(orientation_trace, Delta_t, D_rr, time_interval, stride, Simulation, molecule, theory_only = False):
    
    """Calculates the rotational time correlation function P2 from the orientation traces of a molecule population 
    as well as the corresponding theoretical prediction for validation purposes.
    
    Parameters
    ----------
    orientation_trace : `float64[t,N,4]`
        Orientations (in quaternion representation) of a molecule population of size N at t different time steps.
    Delta_t : `float64`
        Time step.
    D_rr : `float64[:,:]`
        Rotational diffusion tensor.
    time_interaval : `int64`
        Target time step. Note: Currently sampling always starts at time step 0.
    stride : `int64`
        Stride with which the molecule orientations are sampled.
    Simulation : `object`
        Instance of the Simulation class.
    molecule : `string`
        Molecule type
    theory_only : `boolean`
        If True, only the theoretical prediction of P2 is returned. Default = False.
    
    
    Returns
    -------
    tuple(list(float64[:], float64[:], float64[:]), list(float64[:], float64[:], float64[:]), float64[:])
        P2 (Simulation), P2 (Theory), time vector
    
    """
    
    time_steps = np.arange(0,time_interval, stride)

    P2_1_t = []
    P2_2_t = []
    P2_3_t = [] 
    ex = np.array([1.0,0.0,0.0])
    ey = np.array([0.0,1.0,0.0])
    ez = np.array([0.0,0.0,1.0])
    for n in time_steps:
        P2_1_t.append(calc_P_theory(n*Delta_t, ex, D_rr))
        P2_2_t.append(calc_P_theory(n*Delta_t, ey, D_rr))
        P2_3_t.append(calc_P_theory(n*Delta_t, ez, D_rr))
    
    if theory_only == False:
        P2_1, P2_2, P2_3 = calcP2_anisotropic_Population(time_interval, stride, orientation_trace)
    
        
    if theory_only == False:
        return [P2_1, P2_2, P2_3], [P2_1_t, P2_2_t, P2_3_t], time_steps*Delta_t
    else:
        return [P2_1_t, P2_2_t, P2_3_t], time_steps*Delta_t