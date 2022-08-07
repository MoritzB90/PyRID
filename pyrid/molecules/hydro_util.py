# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""


import numpy as np
import numba as nb
# from numba.experimental import jitclass
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from numpy.linalg import inv, norm
from numpy import pi

@nb.njit
def center_of_mass(pos,radii):
    
    """Returns the center of mass of the molecule assuming equal mass density for all paricles with radius a_i.
    
    Parameters
    ----------
    my_molecule : `obj`
        Instance of molecule containing the position data of each bead.
    eta_0 : `float`
        Viscosity
    
    Returns
    -------
    float64[3]
        Center of mass
    
    Notes
    -----
    The center of mass is defined by:
        
    .. math::
        :label: r_CoM
        
        \\boldsymbol{R} = \\frac{1}{M} \\sum_{i=1}^n m_i \\boldsymbol{r_i},
        
    where :math:`M` is the total mass of the molecule.
    
    """
    
    n = len(pos)
    
    R_CoM = np.zeros(3, dtype = np.float64)
    rho = 1.0
    M = 0.0
    for i in range(n):
        r_i = pos[i]
        a_i = radii[i]
        
        R_CoM[:] += 4/3*np.pi*a_i**3*rho*r_i
        
        M += 4/3*np.pi*a_i**3*rho
        
    R_CoM[:]/=M
    
    return R_CoM
    

@nb.njit
def supermatrix_inverse(M1,M2,M3,M4):
    
    """Calculates the inverse of a 2x2 supermatrix.
    
    Parameters
    ----------
    M1 : float64[3N,3N]
        submatrix at (0,0)
    M2 : float64[3N,3N]
        submatrix at (0,1)
    M3 : float64[3N,3N]
        submatrix at (1,0)
    M4 : float64[3N,3N]
        submatrix at (1,1)
        
    Notes
    -----
    A super Matrix :math:`M=[[M1, M2], [M3, M4]]` is invertible, if both the diagonal blocks, :math:`M_1` and :math:`M_4` are invertible
    The inverse of a (2x2) supermatrix can be calculated by :cite:p:`Varadarajan2004`, :cite:p:`Deligne1996`:
    
    .. math::
        :label: supermaztrix_inverse
        
        \\begin{align*}
        & T_1 = (M_1 - M_2 M_4^{-1} M_3)^{-1} \\\\
        & T_2 = -M_1^{-1} M_2 (M_4-M_3 M_1^{-1} M_2)^{-1} \\\\
        & T_3 = -M_4^{-1} M_3 (M_1-M_2 M_4^{-1} M_3)^{-1} \\\\
        & T_4 = (M_4 - M_3 M_1^{-1} M_2)^{-1} \\\\
        \\end{align*}
        
    Returns
    -------
    tuple(float64[3N,3N], float64[3N,3N], float64[3N,3N], float64[3N,3N])
        Some information
    
    """
    
    T1 = inv(M1-np.dot(M2,np.dot(inv(M4),M3)))
    T2 = -np.dot(inv(M1),np.dot(M2,inv(M4-np.dot(M3,np.dot(inv(M1),M2)))))
    T3 = -np.dot(inv(M4),np.dot(M3,inv(M1-np.dot(M2,np.dot(inv(M4),M3)))))
    T4 = inv(M4-np.dot(M3,np.dot(inv(M1),M2)))
    
    return T1,T2,T3,T4

#%%

@nb.njit
def calc_CoD(D_rr, D_tr):
    
    """Calculates the center of diffusion of a rigid bead molecule.
    
    Parameters
    ----------
    D_rr : `float64[3,3]`
        Rotational diffusion tensor.
    D_tr : `float64[3,3]`
        Translation-rotation coupling
        
    
    Returns
    -------
    float64[3]
        Center of diffusion.
        
    
    Notes
    -----
    The diffusion tensor :math:`[[D^{tt}, D^{tr}],[D^{rt},D^{rr}]]` depends on the choice of the bead model origin (body frame). For symmetry reason, one should use the so-called center of Diffusion, which can be calculated from a diffusion tensor referring to an arbitrary origin :cite:p:`Harvey1980`, :cite:p:`Carrasco1999` :
        
    .. math::
        :label: r_CoD
        
        \\begin{align*}
        \\boldsymbol{r}_{OD}
        = &
        \\begin{pmatrix}
        x_{OD} \\\\
        y_{OD}\\\\
        z_{OD}
        \\end{pmatrix} \\\\
        = &
        \\begin{pmatrix}
        D_{rr}^{yy}+D_{rr}^{zz} & -D_{rr}^{xy} & -D_{rr}^{xz}\\\\
        -D_{rr}^{xy} & D_{rr}^{xx}+D_{rr}^{zz} & -D_{rr}^{yz}\\\\
        -D_{rr}^{xz} & -D_{rr}^{yz} & D_{rr}^{yy}+D_{rr}^{xx}
        \\end{pmatrix}^{-1}
        \\begin{pmatrix}
        D_{tr}^{zy}-D_{tr}^{yz}\\\\
        D_{tr}^{xz}-D_{tr}^{zx}\\\\
        D_{tr}^{yx}-D_{tr}^{xy}
        \\end{pmatrix}
        \\end{align*}
        
    After calculating the the center of diffusion, we simply set the rigid body origin to :math:`r_{OD}` and recalculate the diffusion tensor.
    
    """
    
    r_CoD_0 = np.array([[D_rr[1][1]+D_rr[2][2], -D_rr[0][1], -D_rr[0][2]],
                      [-D_rr[0][1], D_rr[0][0]+D_rr[2][2], -D_rr[1][2]],
                      [-D_rr[0][2], -D_rr[1][2], D_rr[1][1]+D_rr[0][0]]])
    
    r_CoD_1 = np.array([[D_tr[2][1]-D_tr[1][2]],
                      [D_tr[0][2]-D_tr[2][0]],
                      [D_tr[1][0]-D_tr[0][1]]])
    
    r_CoD = np.dot(inv(r_CoD_0), r_CoD_1).reshape(3)
    
    return r_CoD


#%%

@nb.njit
def I():
    return np.eye(3)

@nb.njit
def P(rij):
    
    """Calculates the normalized tensor/outer product :math:`\\frac{\\boldsymbol{r}_{ij} \otimes \\boldsymbol{r}_{ij}}{r_{ij}}` for the hydrodynamic interaction tensor (Oseen tensor).
    
    Parameters
    ----------
    rij : `float64[3]`
        Distance between partcile i and j
    
    Returns
    -------
    float[3,3]
        Normalized outer product of r_ij
    
    """
    
    # return np.dot(rij,rij)/norm(rij)**2
    P = np.zeros((3,3))
    for alpha in range(3):
        for beta in range(3):
            P[alpha][beta] = (rij[alpha]*rij[beta])/norm(rij)**2

    return P

@nb.njit
def levi_civita(rij):
    
    """Returns the product of the Levi-Civita tensor with vector rij.
    
    Parameters
    ----------
    rij : `float64[3]`
        Distance between partcile i and j
    
    Notes
    -----
    
    Given the distance vector :math:`r_{ij} = [x_{ij},y_{ij},z_{ij}]` the product :math:`\epsilon \cdot r_{ij}` is returned, where :math:`\epsilon` is the Levi-Civita tensor:
    
    .. math::
        
        \epsilon \cdot r_{ij} = 
        \\begin{pmatrix}
        0 & z_{ij} & -y_{ij}\\\\
        -z_{ij} & 0 & x_{ij}\\\\
        y_{ij} & -x_{ij} & 0
        \\end{pmatrix}.
    
    
    Returns
    -------
    float[3,3]
        Levi-Civita tensor
    
    """
    
    xij = rij[0]
    yij = rij[1]
    zij = rij[2]
    return np.array([[0,zij,-yij],[-zij,0,xij],[yij,-xij,0]])
   
@nb.njit 
def calc_mu(pos,radii, eta_0):
    
    """Calculates the transtalional, rotational and translation-rotation coupling parts of the mobility super matrix :math:`\\mu_{tt}, \\mu_{rr}, \\mu_{tr} = \\mu_{rt}`.
    
    Parameters
    ----------
    my_molecule : `obj`
        Instance of molecule containing the position data of each bead.
    eta_0 : `float`
        Viscosity
    
    Notes
    -----
    
    The mobility matrices are directly related to the modified Oseen tensor. The Oseen tensor has first been introduced by Oseen in 1927 (For reference also see :cite:p:`Dhont1996a`). The Oseen tensor comes up when solving the Stokes equations, which are a linearization of the Navier-Stokes equations. More precisely, it comes up when solving for the flow velocity field in case of a force acting on a point particle (:math:`f(r) = f_0 \\delta(r-r_p)`) which is immersed in a viscous liquid.
    In this case, we can write the solution to the Stokes equation as (due to its linearity, any solution to the Stokes equation has to be a linear transformation):
        
    .. math::
        
        v(r) = T(r-r_p) \cdot F
        
    T is called the hyrodynamic interaction tensor, Oseen tensor or Green's function of the Stoke's equations. The above solution is also called Stokeslet:
        
    .. math::
        
        \\boldsymbol{T}(r) = \\frac{1}{8 \\pi \\eta r} \cdot \\Big(\\boldsymbol{I}+\\frac{\\boldsymbol{r} \otimes \\boldsymbol{r}}{r^2} \\Big),
        
    where :math:`\\eta` is the viscosity and :math:`\otimes` is the outer product. In prosa, :math:`T` gives the fluid flow velocty at some point :math:`r`, given a force acting at another point :math:`r_p`.
    Kirkwood and Riseman calculated the translational mobility tensor of a rigid bead molecule using the Oseen tensor to describe the hydrodynamic interaction between the beads, and by assigning each bead its friction coefficient :math:`\\zeta_i = 6 \pi \\eta_0 a_i`:
        
    .. math::
        
        \\begin{align*}
        \\mu_{ij}^{tt} = & \\delta_{ij}(6 \\pi \\eta_0 a_i)^{-1} \\boldsymbol{I} \\\\
        & + (1-\\delta_ij)(8 \\pi \\eta_0 r_{ij})^{-1} \\\\
        & \\Big(\\boldsymbol{I}+\\frac{\\boldsymbol{r} \otimes \\boldsymbol{r}}{r^2} \\Big)
        \\end{align*}
    
    This solution is fairly intuitive. The first term is just the mobility of a single particle with radius :math:`a_i`. The second term is just the Oseen tensor. Recalling that the mobility :math:`\mu` is defined as the ratio of a particles drift velocity and the applied applied force, the interpretation of the Oseen tensor as representing the interaction part of the bead mobility matrix feels natural (recalling its origin (see above)). However, we also instantly see that something is missing since the Oseen tensor  only considers the distance between the bead centers but neglects their volume/radius :math:`a_i`. Fortunately,  :cite:t:`Torre1977` established a correction to the Oseen tensor for nonidentical spheres (also see :cite:`Torre2007`):
        
    .. math::
        :label: modified_Oseen
        
        \\boldsymbol{T}_{ij} = \\frac{1}{8 \\pi \\eta r} \cdot \\Big(\\boldsymbol{I}+\\frac{\\boldsymbol{r}_{ij} \otimes \\boldsymbol{r}_{ij}}{r_{ij}^2} + \\frac{\\sigma_i + \\sigma_j}{r_{ij}^2} \\Big( \\frac{1}{3} \\boldsymbol{I} - \\frac{\\boldsymbol{r}_{ij} \otimes \\boldsymbol{r}_{ij}}{r_{ij}^2} \\Big) \\Big),
    
    By that, the friction tensor reads:
    
    .. math::
        :label: mu_tt
        
        \\begin{align*}
        \mu^{tt}_{ij} = & \\delta_{ij} (6 \\pi \eta_0 a_i)^{-1} \\boldsymbol{I} + (1-\\delta_{ij})(8 \\pi \\eta_0 r_{ij}^{-1})(\\boldsymbol{I}+\\boldsymbol{P}_{ij}) \\\\
        & + (8 \\pi \\eta_0 r_{ij}^{-3})(a_i^2+a_j^2)(\\boldsymbol{I}-3 \\boldsymbol{P}_{ij}),
        \\end{align*}
    
    where :math:`\\boldsymbol{P}_{ij} = \\Big(\\boldsymbol{I}+\\frac{\\boldsymbol{r} \otimes \\boldsymbol{r}}{r^2} \\Big)`.
    The mobility tensor for rotation, not correcting for the beads volume, reads :cite:p:`Carrasco1999a`.
    
    .. math::
        :label: mu_rr
        
        \\begin{align*}
        \mu^{rr}_{ij} = & \\delta_{ij} (8 \\pi \\eta_0 a_i^3)^{-1} \\boldsymbol{I} \\\\
        & + (1 - \delta_{ij})(16 \\pi \\eta_0 r^3_{ij})^{-1} (3 \\boldsymbol{P}_{ij} - \\boldsymbol{I})
        \\end{align*}    
        
    Here, again, the first term is just the rotational mobility of the single bead and the second term accounts for the hydrodynamic interaction. In this formulation, there is still a correction for the volume missing. This correction consists of adding :math:`6 \eta_0 V_m \\boldsymbol{I}` to the diagonal components of the rotational friction tensor :math:`\\Xi^{rr}_O`, where :math:`V_m` is the volume of the bead model (sum over all bead volumes) :cite:`Torre1983`, :cite:p:`Carrasco1999a`.
    
    And, at last, for rotation-translation coupling, we have :cite:p:`Carrasco1999a`:
    
    .. math::
        :label: mu_rt
        
        \mu^{rt}_{ij} = (1-\\delta_{ij}) (8 \\pi \\eta_0 r_{ij}^2)^{-1} \\boldsymbol{\\epsilon}\\boldsymbol{\\hat{r}}_{ij} 
    
    Returns
    -------
    tuple(float[N,N,3,3], float[N,N,3,3], float[N,N,3,3], float[N,N,3,3])
        mu_tt, mu_rt, mu_tr, mu_rr
    
    """
    
    n = len(pos)
    mu_tt = np.zeros((n,n,3,3))
    mu_rt = np.zeros((n,n,3,3))
    mu_tr = np.zeros((n,n,3,3))
    mu_rr = np.zeros((n,n,3,3))

    for i in range(n):
        r_i = pos[i]
        a_i = radii[i]
        for j in range(n):
            r_j = pos[j]    
            a_j = radii[i]
            rij = r_i-r_j
            
            if i == j:
                mu_tt[i][j] = (6*pi*eta_0*a_i)**-1*I()
                
                mu_rr[i][j] = (8*pi*eta_0*a_i**3)**-1*I()
            else:
                # mu_tt[i][j] = (8*pi*eta_0*norm(rij)**-1)*(I()+P(rij))+(8*pi*eta_0*norm(rij)**-3)*(a_i**2+a_j**2)*(I()+3*P(rij))
                # mu_tt[i][j] = np.dot((8*pi*eta_0*norm(rij)**-1)*(I()+P(rij)),(8*pi*eta_0*norm(rij)**-3)*(a_i**2+a_j**2)*(I()+3*P(rij)))
                
                mu_tt[i][j] = (8*pi*eta_0*norm(rij))**-1*(I()+P(rij)+(a_i**2+a_j**2)/norm(rij)**2*(I()/3-P(rij)))
                
                
                mu_rt[i][j] = -(8*pi*eta_0*norm(rij)**2)**-1*levi_civita(rij)
                mu_tr[j][i] = -(8*pi*eta_0*norm(rij)**2)**-1*levi_civita(rij)
                
                mu_rr[i][j] = (16*pi*eta_0*norm(rij)**3)**-1*(3*P(rij)-I())
         
    return mu_tt, mu_rt, mu_tr, mu_rr

@nb.njit
def calc_zeta(mu_tt, mu_tr, mu_rt, mu_rr):
    
    """Calculates the friction tensors :math:`\\zeta^{tt}, \\zeta^{rr}, \\zeta^{tr}, \\zeta^{rt}` from the inverse of the mobility supermatrix :math:`[[\\mu^{tt}, \\mu^{rr}], [\\mu^{tr}, \\mu^{rt}]]`.
    
    Parameters
    ----------
    mu_tt : `float64[3N,3N]`
        Translational mobility tensor.
    mu_tr : `float64[3N,3N]`
        Translation-rotation coupling.
    mu_rt : `float64[3N,3N]`
        Rotation-translation coupling.
    mu_rr : `float64[3N,3N]`
        Rotational mobility tensor.
    
    Raises
    ------
    NotImplementedError (just an example)
        Brief explanation of why/when this exception is raised
    
    Returns
    -------
    tuple(float64[3N,3N], float64[3N,3N], float64[3N,3N], float64[3N,3N])
        The friction tensors :math:`\\zeta^{tt}, \\zeta^{rr}, \\zeta^{tr}, \\zeta^{rt}`.
    
    """
    
    zeta_tt, zeta_tr, zeta_rt, zeta_rr = supermatrix_inverse(mu_tt, mu_tr, mu_rt, mu_rr)

    return zeta_tt, zeta_tr, zeta_rt, zeta_rr

@nb.njit
def A(r):
    
    """Returns matrix A which turns the cross product rxw into a dot product A.w.
    
    Parameters
    ----------
    r : `float`
        Bead position
    
    
    Notes
    -----
    
    We can turn the cross product :math:`r \\times \\omega` with :math:`r = [x,y,z]` into a dot product :math:`A \\cdot \\omega`, where
    
    .. math::
        
        A = 
        \\begin{pmatrix}
        0 & -z & y\\\\
        z & 0 & -x\\\\
        -y & x & 0
        \\end{pmatrix}.
    
    """
    
    x = r[0]
    y = r[1]
    z = r[2]
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])

@nb.njit
def calc_Xi(zeta_tt, zeta_rt, zeta_tr, zeta_rr, pos,radii):
    
    """Calculates the friction tensors (3,3) of a ridig bead molecule from the friction super matricies (3N,3N) of its individiual beads.
    
    Parameters
    ----------
    zeta_tt : `float64[3N,3N]`
        Translational friction matrix.
    zeta_rt : `float64[3N,3N]`
        Rotation-translation coupling matrix.
    zeta_tr : `float64[3N,3N]`
        Translation-rotation coupling matrix.
    zeta_rr : `float64[3N,3N]`
        Rotational friction matrix.
    my_molecule : `obj`
        Molceule class instance containing position data of each bead.
    
    Raises
    ------
    NotImplementedError (just an example)
        Brief explanation of why/when this exception is raised
    
    Returns
    -------
    tuple(float[3,3], float[3,3], float[3,3], float[3,3])
        Returns the friction tensors Xi_tt, Xi_rt, Xi_tr, Xi_rr of the molecule.
        
    
    Notes
    -----
    
    Here, we closely follow :cite:p:`Carrasco1999a`. To get an expression for the friction tensor of a rigid bead molecule, we start by considering a sytsem of :math:`N` free spherical particles in a fluid with viscosity :math:`\\eta_0`. Each sphere lateraly moves at some velocity :math`u_i` and rotates with some angular velocity :math:`\\omega_i`. The spheres will experience a frictional force and torque :math:`F_i, T_i`. In the noninertial regime (Stokes), the relationship between the force/torque and the velocities are linear:
        
    .. math::
        :label: FrictionForce
        
        \\boldsymbol{F}_i = \\sum_{j=1}^N \\zeta_{ij}^{tt} \\cdot \\boldsymbol{u}_j + \\zeta_{ij}^{tr} \\cdot \\boldsymbol{\\omega}_j
        
    .. math::
        :label: FrictionTorque
        
        \\boldsymbol{T}_i = \\sum_{j=1}^N \\zeta_{ij}^{rt} \\cdot \\boldsymbol{u}_j + \\zeta_{ij}^{rr} \\cdot \\boldsymbol{\\omega}_j .
    
    The :math:`\\zeta_{ij}` are the (3x3) friction matrices, connecting the amount of friction a particle i expiriences due to the presence of particle j moving through the fluid with velocities :math:`u_j, \\omega_j`. We may rewrite this in a matrix representation as:
    
    .. math::
        :label: ForceTorque
        
        \\begin{pmatrix}
        F \\\\
        T \\\\
        \\end{pmatrix}
        =
        \\begin{pmatrix}
        \\zeta^{tt} & \\zeta^{tr} \\\\
        \\zeta^{rt} & \\zeta^{rr} \\\\
        \\end{pmatrix}
        \\begin{pmatrix}
        U \\\\
        W \\\\
        \\end{pmatrix},
    
    where :math:`F = (\\boldsymbol{F}_1, ..., \\boldsymbol{F}_N)^T`, :math:`T = (\\boldsymbol{T}_1, ..., \\boldsymbol{T}_N)^T`
    and :math:`U = (\\boldsymbol{u}_1, ..., \\boldsymbol{u}_N)^T`, :math:`W = (\\boldsymbol{\\omega}_1, ..., \\boldsymbol{\\omega}_N)^T`. Here :math:`\\zeta` are of dimension (3Nx3N), forming the friction supermatrix of dimension (6N,6N). The inverted friction supermatrix is the mobility supermatrix (for inmversion of supermatrices also see supermatrix_inverse()).
    
    Next, we consider not a system of N free beads but a rigid bead model, i.e. the beads are rigidly connected.
    Thereby, all beads move together with some translational velocity :math:`u_{O}`. Let the body's frame of reference lie at the center of diffusion of the bead model :math:`\\boldsymbol{r}_O` and let :math:`\\omega` be the angular velocity of the rigid bead model. Then, in addition to the translational velocity of the molecule's center, each bead experiences a translation velocity due to the rotation :math:`\\boldsymbol{\omega} \\times \\boldsymbol{r}_i`, where :math:`\\boldsymbol{r}_i` is the position vector from the moclules origin :math:`\\boldsymbol{r}_O` (in the body frame of reference). Thereby, the total velocity is:
        
    .. math::
        :label: Velocity
        
        \\boldsymbol{u}_i = \\boldsymbol{u}_O + \\boldsymbol{\\omega} \\times \\boldsymbol{r}_i  
        
    Thereby, the force that a single bead experiences due to the movement of all the other beads is:
    
    .. math::
        :label: FrictionForce_Bead
        
        \\boldsymbol{F}_i = \\sum_{j=1}^N \\zeta_{ij}^{tt} \\cdot (\\boldsymbol{u}_O + \\boldsymbol{\\omega} \\times \\boldsymbol{r}_j) + \\zeta_{ij}^{tr} \\cdot \\boldsymbol{\\omega},
        
    and the torque that single bead experiences due to the movement of all the other beads is:
    
    .. math::
        :label: FrictionTorque_Bead
        
        \\boldsymbol{T}_{P,i} = \\sum_{j=1}^N \\zeta_{ij}^{rt} \\cdot (\\boldsymbol{u}_O + \\boldsymbol{\\omega} \\times \\boldsymbol{r}_j) + \\zeta_{ij}^{rr} \\cdot \\boldsymbol{\\omega} .    
        
    From these expressions we simply get the total force acting at the rigid body origin by summation over all beads:
        
    .. math::
        :label: FrictionForce_Total
        
        \\boldsymbol{F} = \\sum_{i=1}^N \\sum_{j=1}^N \\zeta_{ij}^{tt} \\cdot (\\boldsymbol{u}_O + \\boldsymbol{\\omega} \\times \\boldsymbol{r}_j) + \\zeta_{ij}^{tr} \\cdot \\boldsymbol{\\omega}
        
    For the total torque, however, we get an extra term. :math:`\\boldsymbol{T}_{P,i}` is only the torque acting on bead i relative to it's center, i.e. the center of the sphere. Thereby, this only describes the amount of rotation bead i would experience around its center due to the movement of all the other beads. However, the force :math:`\\boldsymbol{F}_{i}` acting on bead i due to the movement of the other beads also results in a torque with which bead i acts on the rigid bead models center :math:`r_O`:
    
    .. math::
        :label: FrictionTroque_Exk
        
        \\boldsymbol{r}_i \\times \\boldsymbol{F}_i = \\boldsymbol{r}_i \\times \\Big( \\sum_j^N \\zeta_{ij}^{tt} (\\boldsymbol{u}_O + \\boldsymbol{\\omega} \\times \\boldsymbol{r}_j) + \\zeta_{ij}^{tr} \\omega \\Big)
    
    Thereby, the total torque acting on the rigid bead model's origin is:
        
    .. math::
        :label: FrictionTorque_Total
        
        \\boldsymbol{T}_O = \\sum_i^N \\boldsymbol{T}_{P,i} +  \\boldsymbol{r}_i \\times \\boldsymbol{F}_i = \\sum_{i=1}^N \\sum_{j=1}^N \\zeta_{ij}^{rt} \\cdot (\\boldsymbol{u}_O + \\boldsymbol{\\omega} \\times \\boldsymbol{r}_j) + \\zeta_{ij}^{rr} \\cdot \\boldsymbol{\\omega} + \\boldsymbol{r}_i \\times \\Big( \\zeta_{ij}^{tt} (\\boldsymbol{u}_O + \\boldsymbol{\\omega} \\times \\boldsymbol{r}_j) + \\zeta_{ij}^{tr} \\omega \\Big). 
  
        
    In principle, we are done now, however, we would like to transform this into a more 'general' expression that we can write in a simple matrix form. For this, we use a little trick to get rid of the cross product, by turning :math:`\\omega \\times r` into the dot product :math:`- A \cdot \\omega` (note: the sign changed, because of the anticommutativity of the cross product). After some rearranging, we end up with:
        
    .. math::
        :label: FrictionForce_Total_2
        
        \\boldsymbol{F} = \\Big( \\sum_{i=1}^N \\sum_{j=1}^N \\zeta_{ij}^{tt} \\Big) \\cdot \\boldsymbol{u}_O + \\Big( \\sum_{i=1}^N \\sum_{j=1}^N - \\zeta_{ij}^{tt} \\cdot \\boldsymbol{A}_j + \\zeta_{ij}^{tr} \\Big) \cdot \\boldsymbol{\\omega}
        
    .. math::
        :label: FrictionTorque_Total_2
        
        \\boldsymbol{T} = \\Big( \\sum_{i=1}^N \\sum_{j=1}^N \\zeta_{ij}^{rt} + A_i \zeta_{ij}^{tt} \\Big) \\cdot \\boldsymbol{u}_O + \\Big( \\sum_{i=1}^N \\sum_{j=1}^N \\zeta_{ij}^{rt} \\cdot \\boldsymbol{A}_j + \\zeta_{ij}^{rr} - A_i \\zeta_{ij}^{tt} A_j  + A_i \\zeta_{ij}^{tr} \\Big) \\cdot \\boldsymbol{\\omega}.           
        
    If we now want write this in matrix form, similar to the free bead example from above:
        
    .. math::
        :label: ForceTorque_Bead
        
        \\begin{pmatrix}
        \\boldsymbol{F} \\\\
        \\boldsymbol{T}_O \\\\
        \\end{pmatrix}
        =
        \\begin{pmatrix}
        \\Xi^{tt} & \\Xi^{tr} \\\\
        \\Xi^{rt} & \\Xi^{rr} \\\\
        \\end{pmatrix}
        \\begin{pmatrix}
        \\boldsymbol{u}_O \\\\
        \\boldsymbol{\\omega} \\\\
        \\end{pmatrix},    
    
    Where we call :math:`\Xi` the friction tensor of the rigid bead molecule :cite:p:`Carrasco1999a` :
    
    .. math::
        :label: Xi
    
        \\begin{align*}
        &\\Xi^{tt} = \\sum_{i=1}^N \\sum_{j=1}^N \\zeta_{ij}^{tt} \\\\
        &\\Xi_{O}^{tr} = \\sum_{i=1}^N \\sum_{j=1}^N ( -\\zeta_{ij}^{tt} \\cdot \\boldsymbol{A}_j + \\zeta_{ij}^{tr} ) \\\\
        &\\Xi_{O}^{rt} = \\sum_{i=1}^N \\sum_{j=1}^N ( \\boldsymbol{A}_j \\cdot \\zeta_{ij}^{tt} + \\zeta_{ij}^{rt} ) \\\\
        &\\Xi_{O}^{rr} = \\sum_{i=1}^N \\sum_{j=1}^N ( \\zeta_{ij}^{rr} - \\zeta_{ij}^{rt} \\cdot \\boldsymbol{A}_j + \\boldsymbol{A}_i \\cdot \\zeta_{ij}^{tr} - \\boldsymbol{A}_i \\cdot \\zeta_{ij}^{tt} \\boldsymbol{A}_j)
        \\end{align*}
        
    The only thing left to do now, is calculating :math:`\\zeta`, which we get from the inverse of the mobility supermatrix. The mobility supermatrix is calculated in calc_mu(my_molecule, eta_0).
    
    """
    
    n = len(pos)
    
    Xi_tt = np.zeros((3,3))
    Xi_rt = np.zeros((3,3))
    Xi_tr = np.zeros((3,3))
    Xi_rr = np.zeros((3,3))
    
    for i in range(n):
        r_i = pos[i]
        for j in range(n):
            r_j = pos[j]
            
            Xi_tt += zeta_tt[i][j]
            
            Xi_rt += zeta_rt[i][j]+np.dot(A(r_i), zeta_tt[i][j])
            
            Xi_tr += -np.dot(zeta_tt[i][j],A(r_i))+zeta_tr[i][j] # TODO for some reason (prob. numerical errors), Xi_tr is less accurately calculated than Xi_rt!
            
            Xi_rr += zeta_rr[i][j]-np.dot(zeta_rt[i][j], A(r_j))+np.dot(A(r_i), zeta_tr[i][j])-np.dot(A(r_i),np.dot(zeta_tt[i][j], A(r_j)))

            
    Xi_tr = Xi_rt.T
    
    # V = 4/3*np.pi*np.sum(my_molecule.radii_rb[:]**3)
    # Xi_rr += 6*eta_0*V*np.eye(3)
    
    return Xi_tt, Xi_rt, Xi_tr, Xi_rr

@nb.njit    
def calc_D(Xi_tt, Xi_rt, Xi_tr, Xi_rr, kB, Temp):
    
    """Caclulates the diffusion tensor from  the supermatrix inverse of the friction tensor.
    
    Parameters
    ----------
    parameter_1 : dtype
        Some Information
    parameter_2 : dtype
        Some Information
    
    Notes
    -----
   
    The diffusion tensor can be calculated from the rigid body's friction tensor via :cite:p:`Carrasco1999` :
        
    .. math::
        :label: DiffTensor
        
        \\begin{pmatrix}
        \\boldsymbol{D}^{tt} & \\boldsymbol{D}^{tr,T} \\\\
        \\boldsymbol{D}^{rt} & \\boldsymbol{D}^{rr} \\\\
        \\end{pmatrix}
        = k_B T
        \\begin{pmatrix}
        \\boldsymbol{\\Xi}^{tt} & \\boldsymbol{\\Xi}^{tr,T} \\\\
        \\boldsymbol{\\Xi}^{rt} & \\boldsymbol{\\Xi}^{rr} \\\\
        \\end{pmatrix}^{-1}  

    Returns
    -------
    tuple(float64[3,3], float64[3,3], float64[3,3], float64[3,3])
        Translation, rotational and rotation-translation coupling diffusion tensors D_tt, D_tr, D_rt, D_rr.
    
    """
    
    D_tt, D_tr, D_rt, D_rr = supermatrix_inverse(Xi_tt, Xi_tr, Xi_rt, Xi_rr)
    
    return kB*Temp*D_tt, kB*Temp*D_tr, kB*Temp*D_rt, kB*Temp*D_rr

@nb.njit
def transform_supermatrix(mu_tt,mu_tr,mu_rt,mu_rr, pos,radii):
    
    """Transforms a (N,N,3,3) matrix into a (3N,3N) matrix.
    
    Parameters
    ----------
    mu_tt : `float64[N,N,3,3]`
        Translational mobility tensor.
    mu_tr : `float64[N,N,3,3]`
        Translation-rotation coupling.
    mu_rt : `float64[N,N,3,3]`
        Rotation-translation coupling.
    mu_rr : `float64[N,N,3,3]`
        Rotational mobility tensor.
    
    
    Returns
    -------
    tuple(float[3N,3N], float[3N,3N], float[3N,3N], float[3N,3N])
        mu_tt_Tf, mu_tr_Tf, mu_rt_Tf, mu_rr_Tf
    
    """
    
    n = len(pos)
    mu_tt_Tf = np.zeros((3*n, 3*n))
    mu_tr_Tf = np.zeros((3*n, 3*n))
    mu_rt_Tf = np.zeros((3*n, 3*n))
    mu_rr_Tf = np.zeros((3*n, 3*n))
    for i in range(n):
        for j in range(n):
            mu_tt_Tf[i*3:(i+1)*3, j*3:(j+1)*3] = mu_tt[i][j]
            mu_tr_Tf[i*3:(i+1)*3, j*3:(j+1)*3] = mu_tr[i][j]
            mu_rt_Tf[i*3:(i+1)*3, j*3:(j+1)*3] = mu_rt[i][j]
            mu_rr_Tf[i*3:(i+1)*3, j*3:(j+1)*3] = mu_rr[i][j]
            
    return mu_tt_Tf, mu_tr_Tf, mu_rt_Tf, mu_rr_Tf

@nb.njit
def transform_reverse_supermatrix(zeta_tt_Tr,zeta_tr_Tr,zeta_rt_Tr,zeta_rr_Tr, pos,radii):
    
    """Transforms a (3N,3N) matrix into a (N,N,3,3) matrix.
    
    Parameters
    ----------
    zeta_tt_Tr : `float64[3N,3N]`
        Translational friction matrix.
    zeta_tr_Tr : `float64[3N,3N]`
        Translation-rotation coupling.
    zeta_rt_Tr : `float64[3N,3N]`
        Rotation-translation coupling.
    zeta_rr_Tr : `float64[3N,3N]`
        Rotational friction matrix.
    
    
    Returns
    -------
    tuple(float[N,N,3,3], float[N,N,3,3], float[N,N,3,3], float[N,N,3,3])
        zeta_tt, zeta_tr, zeta_rt, zeta_rr
    
    """
    
    n = len(pos)
    
    zeta_tt = np.zeros((n, n, 3, 3))
    zeta_tr = np.zeros((n, n, 3, 3))
    zeta_rt = np.zeros((n, n, 3, 3))
    zeta_rr = np.zeros((n, n, 3, 3))
    for i in range(n):
        for j in range(n):
            zeta_tt[i][j] = zeta_tt_Tr[i*3:(i+1)*3, j*3:(j+1)*3]
            zeta_tr[i][j] = zeta_tr_Tr[i*3:(i+1)*3, j*3:(j+1)*3]
            zeta_rt[i][j] = zeta_rt_Tr[i*3:(i+1)*3, j*3:(j+1)*3]
            zeta_rr[i][j] = zeta_rr_Tr[i*3:(i+1)*3, j*3:(j+1)*3]
            
            
    return zeta_tt, zeta_tr, zeta_rt, zeta_rr

@nb.njit
def diffusion_tensor_off_center(pos,radii, eta_0, Temp, kB):
    
    """Calculates the diffusion tensor (off the center of diffusion) from the moelcule structure, the fluid viscosity and temperature.
    
    Parameters
    ----------
    my_molecule : `obj`
        Instance of molecule containing the position data of each bead.
    eta_0 : `float`
        Viscosity
    Temp : `float`
        Temperature
    kB : `float`
        Boltzmann constant
    
    
    Returns
    -------
    tuple(float64[3,3], float64[3,3], float64[3,3], float64[3,3])
        Translation, rotational and rotation-translation coupling diffusion tensors D_tt, D_tr, D_rt, D_rr.
    
    """
    
    mu_tt, mu_rt, mu_tr, mu_rr = calc_mu(pos,radii, eta_0)
    
    mu_tt_Tf, mu_tr_Tf, mu_rt_Tf, mu_rr_Tf = transform_supermatrix(mu_tt,mu_tr,mu_rt,mu_rr, pos,radii)
    
    zeta_tt_Tr, zeta_tr_Tr, zeta_rt_Tr, zeta_rr_Tr = calc_zeta(mu_tt_Tf, mu_tr_Tf, mu_rt_Tf, mu_rr_Tf)
    
    zeta_tt, zeta_tr, zeta_rt, zeta_rr = transform_reverse_supermatrix(zeta_tt_Tr,zeta_tr_Tr,zeta_rt_Tr,zeta_rr_Tr, pos,radii)
    
    Xi_tt, Xi_rt, Xi_tr, Xi_rr = calc_Xi(zeta_tt, zeta_rt, zeta_tr, zeta_rr, pos,radii)
    
    D_tt, D_tr, D_rt, D_rr = calc_D(Xi_tt, Xi_rt, Xi_tr, Xi_rr, kB, Temp)
    
    return D_tt, D_tr, D_rt, D_rr
   
# @nb.njit
def diffusion_tensor(Simulation, molecule_name, return_CoD = False, return_CoM = False, return_coupling = False):
    
    """Calculates the diffusion tensor, accounting for the center of diffusion of the rigid bead molecule. The origin of the molecule is automatically updated to the center of diffusion :math:`r_{OD}`!
    
    Parameters
    ----------
    my_molecule : `obj`
        Instance of molecule containing the position data of each bead.
    Simulation : `obj`
        Instance of the Simulation class
    
    
    Returns
    -------
    tuple(float64[3,3], float64[3,3], float64[3,3], float64[3,3], float64)
        Translation, rotational and rotation-translation coupling diffusion tensors D_tt, D_rr, D_tr, D_rt and the center of diffusion r_CoD, center of mass r_CoM.
    
    """
    
    print('Calculating mobility tensor for rigid body molecule '+molecule_name+'.')
    
    System = Simulation.System
    
    my_molecule = System.molecule_types[molecule_name]
    unit_prefix = System.length_units_prefix[System.length_unit]
    
    pos = np.copy(my_molecule.pos_rb)/unit_prefix
    radii = np.copy(my_molecule.radii_rb)/unit_prefix
    
    eta_0 = 1.0 # System.eta
    Temp = 1.0 # System.Temp
    kB = 1.0 # System.kB
    
    D_tt, D_tr, D_rt, D_rr = diffusion_tensor_off_center(pos,radii, eta_0, Temp, kB)
    
    r_CoD = calc_CoD(D_rr, D_tr)
    
    n = len(pos)
    for i in range(n):
        pos[i][:] -= r_CoD
        
    D_tt, D_tr, D_rt, D_rr = diffusion_tensor_off_center(pos,radii, eta_0, Temp, kB)
    
    r_CoM = center_of_mass(pos,radii)
    
    print('-------------------------------')
    
    D_tt *= System.kB*System.Temp/System.eta/unit_prefix
    D_rr *= System.kB*System.Temp/System.eta/unit_prefix**3
    D_tr *= System.kB*System.Temp/System.eta/unit_prefix**2
    D_rt *= System.kB*System.Temp/System.eta/unit_prefix**2
    r_CoD *= unit_prefix
    r_CoM *= unit_prefix
    
    n = len(my_molecule.pos_rb)
    for i in range(n):
        my_molecule.pos_rb[i][:] -= r_CoD
        
    n = len(my_molecule.pos)
    for i in range(n):
        my_molecule.pos[i][:] -= r_CoD
    
    my_molecule.h_membrane -= r_CoD
    
    
    return_values = [D_tt, D_rr, D_tr, D_rt, r_CoD, r_CoM]
    return_booleans = [True, True, return_coupling, return_coupling, return_CoD, return_CoM]
    
    return tuple(return_values[i] for i in range(len(return_values)) if return_booleans[i])
    
#%%

# if __name__ == '__main__':




