=====================
Anisotropic diffusion
=====================

The motion of an isolated rigid bead molecule j in solution can be described in terms of the Langevin equation for translational and rotational motion. Note that we are always considering isolated molecules in dispersion and do not account for the hydrodynamic interaction between molecules as this is computationally very expensive (:math:`O(N^2)-O(N^3)`) :cite:p:`Geyer2009, Dlugosz2011`. In the most general case the Langevin equation for translational and rotational motion reads :cite:p:`Ermak1978, Dickinson1985, Jones1991`:

.. math::
    m \frac{d^2\boldsymbol{r}_j(t)}{dt^2} = \boldsymbol{F}_j - \Big(\boldsymbol{\Xi}^{tt} \frac{d\boldsymbol{r}_j}{dt} + \boldsymbol{\Xi}^{tr} \frac{d\boldsymbol{\phi}_j}{dt}\Big) + \boldsymbol{R}^t


.. math::
    \frac{d}{dt} \Big( I \frac{d \boldsymbol{\phi}_j(t)}{dt} \Big) = \boldsymbol{T}_j - \Big(\boldsymbol{\Xi}^{rr} \frac{d\boldsymbol{\phi}_j}{dt} + \boldsymbol{\Xi}^{rt} \frac{d\boldsymbol{r}_j}{dt}\Big) + \boldsymbol{R}^r,


with 

.. math::
    \langle \boldsymbol{R}^a(t)\rangle = 0


.. math::
    \langle \boldsymbol{R}^a(t) \boldsymbol{R}^b(t')\rangle = 2 k_B T \boldsymbol{\Xi}_{ij}^{ab} \delta(t-t'),


where :math:`a,b \in \{t,r\}`. Here, :math:`\boldsymbol{\Xi}^{tt}, \boldsymbol{\Xi}^{rr}, \boldsymbol{\Xi}^{tr}, \boldsymbol{\Xi}^{rt}` are the translational, rotational and translation-rotation coupling friction tensors of the rigid body in the lab frame. Also, :math:`\boldsymbol{\Xi}^{ab} = k_B T (\boldsymbol{D}^{-1})^{ab}` (Einstein relation). Due to the translation-rotation coupling, the equations for rotation and translation are not independent. For low-mass particles, such as molecules, and for long enough time intervals, the acceleration of the molecules can be neglected in the description of the diffusion process. As such it is convenient to describe the motion of molecules by overdamped Langevin dynamics also called Brownian motion where :math:`I \frac{d^2 \phi_j(t)}{dt^2} = m \frac{d^2 x_j(t)}{dt^2} = 0`:

.. math::
    \frac{d\boldsymbol{r}_j(t)}{dt} = \boldsymbol{M}_{j}^{tt} \boldsymbol{F}_j + \boldsymbol{M}_{j}^{tr} \boldsymbol{T}_j + \boldsymbol{S}^t


.. math::
    \frac{d \boldsymbol{\phi}_j(t)}{dt} = \boldsymbol{M}_{j}^{rr} \boldsymbol{T}_j + \boldsymbol{M}_{j}^{rt} \boldsymbol{F}_j + \boldsymbol{S}^r.


with 

.. math::
    \langle \boldsymbol{S}^a(t)\rangle = 0


.. math::
    \langle \boldsymbol{S}^a(t) \boldsymbol{S}^b(t')\rangle = 2 k_B T \boldsymbol{M}_{ij}^{ab} \delta(t-t'),


where :math:`\boldsymbol{M}^{tt}, \boldsymbol{M}^{rr}, \boldsymbol{M}^{tr}, \boldsymbol{M}^{rt}` are the translational, rotational and translation-rotation coupling mobility tensors of the rigid body in the lab frame and :math:`\boldsymbol{M}^{ab} = \frac{\boldsymbol{D}^{ab}}{k_B T}`. Also  :math:`\boldsymbol{M}^{rt} = \boldsymbol{M}^{tr,T}`. In most cases, the effect of the translation-rotation coupling on the molecular dynamics is negligible. However, translation-rotation coupling increases the complexity of the propagation algorithm for the translation and rotation vectors. Therefore, in the following, we will consider translation and rotation as being independent. In this case, the propagator for the Cartesian coordinates as well as the orientation angle can be formulated as

.. math::
    \boldsymbol{r}_j(t) = \boldsymbol{r}_j(t-\Delta t) + \boldsymbol{A}_j \boldsymbol{M}_{j}^{tt,b} \boldsymbol{A}_j^T \boldsymbol{F}_j \Delta t + \boldsymbol{A}_j \sqrt{2 \boldsymbol{M}_{j}^{tt,b} k_B T}\, \boldsymbol{W}^t(\Delta t)


.. math::
    :label: eq:dphidt

    \boldsymbol{\phi}_j(t) = \boldsymbol{\phi}_j(t-\Delta t) + \boldsymbol{A}_j \boldsymbol{M}_{j}^{rr,b} \boldsymbol{A}_j^T \boldsymbol{T}_j \Delta t + \boldsymbol{A}_j \sqrt{2 \boldsymbol{M}_{j}^{rr,b} k_B T}\, \boldsymbol{W}^r(\Delta t).


Here, :math:`\boldsymbol{W}(\Delta t)` is a 3-dimensional Wiener process, i.e. :math:`\boldsymbol{W}(t+\Delta t) - \boldsymbol{W}(t) \sim \mathcal{N}(0, \Delta t)`, which can be argued from the central limit theorem and the assumption that the forces of the solvent molecules act with equal probability from all directions. The superscript :math:`b` indicates that the mobility tensors :math:`\boldsymbol{M}^{ab,b}` are given in terms of the body/local frame of the molecule, which is much more convenient when we talk about the propagation algorithm. In this context, :math:`\boldsymbol{A}_j` is the rotation matrix of molecule j. One problem with the rotational equation of motion is that several issues arise depending on how rotations are represented. Propagating the rotation in terms of Euler angles, e.g., will result in numerical drift and singularities :cite:p:`Baraff2001, Ilie2016`. Therefore, especially in computer graphics, it is standard to represent rotations in unit quaternions, which is much more stable and has fewer issues in general. An algorithm for the rotation propagator based on quaternions can, for example, be found in :cite:p:`Ilie2015`. Here, I follow a path very similar to :cite:p:`Ilie2015`. However, I will introduce a more concise derivation of the algorithm instead of the somewhat lengthy derivation in :cite:p:`Ilie2015` that is based on a reformulation of the problem in terms of generalized coordinates.

Quaternion propagator
---------------------

The goal is to derive a propagator for the rotation quaternion. A well-established connection between the angular velocity and the unit quaternion velocity is :cite:p:`Baraff2001`:

.. math::
    :label: eq:dg(dphi)

    \frac{d\boldsymbol{q}}{dt} = \frac{1}{2} \frac{\boldsymbol{\phi}}{dt} \boldsymbol{q} = \boldsymbol{B} \frac{\boldsymbol{\phi}}{dt}


where

.. math::
    \begin{split}\boldsymbol{B}
    = \frac{1}{2}
    \begin{pmatrix}
        -q_1 & -q_2 & -q_3 \\
        q_0 & q_3 & -q_2 \\
        -q_3 & q_0 & q_1 \\
        q_2 & -q_1 & q_0 \\
    \end{pmatrix}.
    \end{split}


Inserting :eq:`eq:dphidt` into :math:numref:`eq:dg(dphi)`, we get:

.. math::
	:label: eq:dqdt

    \boldsymbol{q}_j(t) = \boldsymbol{q}_j(t-\Delta t) + \boldsymbol{B}_j\boldsymbol{A}_j \boldsymbol{M}_{j}^{rr,b} \boldsymbol{A}_j^T \boldsymbol{T}_j \Delta t + \boldsymbol{B}_j \boldsymbol{A}_j \sqrt{2 \boldsymbol{M}_{j}^{rr,b} k_B T}\, \boldsymbol{W}^r(\Delta t).


The factor :math:`\boldsymbol{B}\boldsymbol{A}` can, however, be simplified to a somewhat surprising degree:

.. math::
    \begin{split}
    \boldsymbol{B}\boldsymbol{A}
    = & \frac{1}{2}
    \begin{pmatrix}
        -q_1 & -q_2 & -q_3 \\
        q_0 & q_3 & -q_2 \\
        -q_3 & q_0 & q_1 \\
        q_2 & -q_1 & q_0 \\
    \end{pmatrix}
    \begin{pmatrix}
        1-2(q_2^2+q_3^2) & 2(q_1 q_2-q_0 q_3) & 2(q_1 q_3+q_0 q_2) \\
        2(q_1 q_2+q_0 q_3) & 1-2(q_1^2+q_3^2) & 2(q_2 q_3-q_0 q_1) \\
    2(q_1 q_3-q_0 q_2) & 2(q_2 q_3+q_0 q_1) & 1-2(q_1^2+q_2^2) \\
    \end{pmatrix}\\
    = & \frac{1}{2}
    \begin{pmatrix}
       -q_1 & -q_2 & -q_3 \\
        q_0 & q_3 (1-2 q^2 ) & q_2 (2 q^2 -1) \\
        q_3 (2 q^2 -1) & q_0 & q1 (1-2 q^2 ) \\
        q_2 (1-2 q^2) & q_1 (2 q^2-1) & q_0 \\
    \end{pmatrix} \\
    = & \frac{1}{2}
    \begin{pmatrix}
       -q_1 & -q_2 & -q_3 \\
        q_0 & -q_3 & q_2 \\
        q_3 & q_0 & -q1 \\
        -q_2 & q_1 & q_0 \\
    \end{pmatrix},
    \end{split}


where :math:`q^2 = q_0^2+q_1^2+q_2^2+q_3^2 = 1`.
For the quaternion to accurately represent the rotation, we need to ensure that it keeps its unit length. However, due to the finite time step in simulations, the quaternion will diverge from unit length over time. Thus, it is necessary to frequently re-normalize the quaternion. :cite:t:`Ilie2015` point out that re-normalization will introduce a bias by changing the sampled phase space distribution. Thereby, it is more appropriate to introduce a constraint force using the method of undetermined Lagrange multipliers as is used in molecular dynamics algorithms such as SHAKE. However, for integration time steps used in practice, I found the error introduced by re-normalization to be negligible.
Fig. :numref:`fig:Diff` shows that the rotation and translation propagators result in the correct mean squared distribution and rotational time correlation. The translational and rotational diffusion tensors used are:

.. math::
    D_{tt} = 
    \begin{pmatrix}
       0.5 & 0 & 0 \\
        0 & 0.4 & 0 \\
        0 & 0 & 0.1 \\
    \end{pmatrix} \frac{nm^2}{ns},


.. math::
    D_{rr} = 
    \begin{pmatrix}
       0.005 & 0 & 0 \\
        0 & 0.04 & 0 \\
        0 & 0 & 0.1 \\
    \end{pmatrix} \frac{rad}{ns},


The mean squared displacement (MSD) is given by 

.. math::
    MSD = \langle |\boldsymbol{x}(t+\Delta t)-\boldsymbol{x}(t)|^2 \rangle


The rotational time correlation function is given by

.. math::
    MSD = \frac{3}{2} \langle (\hat{\boldsymbol{n}}(t+\Delta t)\hat{\boldsymbol{n}}(t))^2 \rangle - \frac{1}{2},


where :math:`\hat{\boldsymbol{n}}(t)` is the normal vector of the molecule, representing it's orientation at time point :math:`t`. Fig. :numref:`fig:Diff` compares the simulation results to the theoretical prediction, which, for the rotational time correlation function, is given by a multi-exponential decay function :cite:p:`Torre1999`:

.. math:: 
	:label: eq:P2

    P_{2,l}(t) = \sum_{i=1}^5 a_{i,l} exp(-t/\tau_i),


where :math:`l \in {1,2,3}`. The relaxation times are given by

.. math::
    \begin{split}
    \tau_1 = (6D - 2 \Delta)^{-1} \\
    \tau_2 = (3D - D^{rr,b}_1)^{-1} \\
    \tau_3 = (3D - D^{rr,b}_2)^{-1} \\
    \tau_4 = (3D - D^{rr,b}_3)^{-1} \\
    \tau_5 = (6D - 2 \Delta)^{-1}.
    \end{split}


`D^{rr,b}_1, D^{rr,b}_2, D^{rr,b}_3` are the eigenvalues of the rotational diffusion tensor :math:`\boldsymbol{D}^{rr,b}` in the molecule frame and D is the scalar rotational diffusion coefficient given by :math:`D = \frac{Tr(\boldsymbol{D}^{rr,b})}{3}`.
Parameter :math:`\Delta` is given by

.. math::
    \Delta = \sqrt{((D^{rr,b}_1)^2+(D^{rr,b}_2)^2+(D^{rr,b}_3)^2-D^{rr,b}_1 D^{rr,b}_2-D^{rr,b}_1 D^{rr,b}_3-D^{rr,b}_2D^{rr,b}_3)}


The amplitudes of the individual exponential decays are given by

.. math:: 
	:label:eq:amplitude_rotrelax

    \begin{split}
    a_{1,l} = \frac{3}{4}(F+G) \\
    a_{2,l} = 3 \hat{n}_{l,2}^2 \hat{n}_{l,3}^2 \\
    a_{3,l} = 3 \hat{n}_{l,1}^2 \hat{n}_{l,3}^2 \\
    a_{4,l} = 3 \hat{n}_{l,1}^2 \hat{n}_{l,2}^2 \\
    a_{5,l} = \frac{3}{4}(F-G),
    \end{split}


with :math:`F = - \frac{1}{3} + \sum_{k=1}^3 \hat{n}_k^4` and :math:`G=\frac{1}{\Delta}\Big( -D + \sum_{k=1}^3 D^{rr,b}_k \Big[ \hat{n}_k^4 + 2 \hat{n}_m^2 \hat{n}_n^2 \Big] \Big)`, where :math:`m, n \in \{1,2,3\}-\{k\}`.

If we choose the normal vectors of each axis :math:`\hat{\boldsymbol{n}}_l` such that these are identical with the basis vectors of the local frame, i.e. :math:`\hat{\boldsymbol{u}}_1 = \boldsymbol{e}_x = [1,0,0]`, :math:`\hat{\boldsymbol{u}}_2 = \boldsymbol{e}_y = [0,1,0]`, :math:`\hat{\boldsymbol{u}}_3 = \boldsymbol{e}_z = [0,0,1]`, :math:`a_2-a_3` vanish such that we end up with a double exponential decay (Fig. :numref:`fig:Diff` B).

.. figure:: Figures/Diffusion.png
    :width: 50%
    :name: fig:Diff
    
    **MSD and rotational relaxation times of a rigid bead molecule matches the theoretical prediction.** **(A)** Mean squared displacement (MSD) of the rigid bead molecule computed with PyRID. The displacement in each dimension (colored markers) is in very good agreement with the theory (black line). **(B)** The rotational relaxation of the rigid bead molecule is also in close agreement with the theory (gray lines) for each of the the rotation axes (colored markers).
