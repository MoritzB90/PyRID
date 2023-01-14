.. _`ch:appendix`:

Appendix
========

.. _`sec:appendix_a`:

Appendix A
----------

Since the method goes beyond what is found in most textbooks, I will
give an introduction to the method in the following. However, I will not
derive the methods in detail since this has been done in various
publications
:cite:p:`Torre1977, Torre1977a, Carrasco1999, Carrasco1999a, Torre2007`.

The Oseen tensor and hydrodynamic interaction between beads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Oseen tensor has first been introduced by Oseen in 1927 (for
reference also see :cite:p:`Dhont1996a`). The Oseen tensor
emerges from the solution of the Stokes equations (linearization of the
Navier-Stokes equations) for the flow velocity field in case of a force
acting on a point-like particle
(:math:`\boldsymbol{F}(\boldsymbol{r}) = \boldsymbol{F}_0 \delta(\boldsymbol{r}-\boldsymbol{r}_p)`)
which is immersed in a viscous liquid. In this case, the solution to the
Stokes equation can be written as a linear transformation (due to its
linearity, any solution to the Stokes equation has to be a linear
transformation):

.. math:: \boldsymbol{v}(\boldsymbol{r}) = \boldsymbol{T}(\boldsymbol{r}-\boldsymbol{r}_p) \cdot \boldsymbol{F},

where :math:`\boldsymbol{r}_p` is the Cartesian coordinate vector of the
point-like particle. :math:`\boldsymbol{T}` is called the hydrodynamic
interaction tensor, Oseen tensor or Green’s function of the Stoke’s
equations. The above solution is also called Stokeslet
:cite:p:`Oseen1927`:

.. math:: \boldsymbol{T}(\boldsymbol{r}) = \frac{1}{8 \pi \eta r} \cdot \Big(\boldsymbol{I}+\frac{\boldsymbol{r} \otimes \boldsymbol{r}}{r^2} \Big),

where :math:`\eta` is the fluid viscosity and :math:`\otimes` is the
outer product and :math:`\boldsymbol{I}` is the identity matrix.
Thereby, :math:`\boldsymbol{T}` relates the fluid flow velocity at some
point :math:`\boldsymbol{r}` to a force acting at another point
:math:`\boldsymbol{r}_p` in the fluid. As mentioned above, the mobility
matrix of a system of dispersed subunits/beads can be related to the
Oseen tensor. The mobility :math:`\boldsymbol{\mu}` is defined as the
ratio of a particle‘s drift velocity and the applied force; thereby, the
Oseen tensor represents an approximation for the hydrodynamic
interaction part of the mobility matrix.
:cite:p:`Bloomfield1967` first introduced a formulation of
the translational mobility tensor for a system of multiple dispersed
beads using the Oseen tensor to describe the hydrodynamic interaction
between the beads, and by assigning each bead its friction coefficient
:math:`\xi_i = 6 \pi \eta_0 \sigma_i` :cite:p:`Carrasco1999`:

.. math::

   \begin{split}
       \boldsymbol{\mu}_{ij}^{tt} = & \delta_{ij}(6 \pi \eta_0 \sigma_i)^{-1} \boldsymbol{I} \\
       & + (1-\delta_ij)(8 \pi \eta_0 r_{ij})^{-1} \\
       & \Big(\boldsymbol{I}+\frac{\boldsymbol{r} \otimes \boldsymbol{r}}{r^2} \Big)
       \end{split}

Here, the first term is just the mobility coefficient of a single
particle with radius :math:`\sigma_i` in the absence of any other beads.
The second term is the Oseen tensor. However, since the Oseen tensor
only considers the distance between the bead centers but neglects their
finite radius :math:`\sigma_i` :cite:p:`Torre1977`
established a correction to the Oseen tensor for nonidentical spheres
(also see :cite:p:`Torre2007`):

.. math::
   :label: modified_Oseen_appdx

   \label{modified_Oseen_appdx}
       \boldsymbol{T}_{ij} = \frac{1}{8 \pi \eta r} \cdot \Big(\boldsymbol{I}+\frac{\boldsymbol{r}_{ij} \otimes \boldsymbol{r}_{ij}}{r_{ij}^2} + \frac{\sigma_i + \sigma_j}{r_{ij}^2} \Big( \frac{1}{3} \boldsymbol{I} - \frac{\boldsymbol{r}_{ij} \otimes \boldsymbol{r}_{ij}}{r_{ij}^2} \Big) \Big),

The corrected friction tensor then reads
:cite:p:`Carrasco1999a`:

.. math::
   :label: mu_tt_appdx

   \label{mu_tt_appdx}
       \begin{split}
       \boldsymbol{\mu}^{tt}_{ij} = & \delta_{ij} (6 \pi \eta_0 \sigma_i)^{-1} \boldsymbol{I} + (1-\delta_{ij})(8 \pi \eta_0 r_{ij}^{-1})(\boldsymbol{I}+\boldsymbol{P}_{ij}) \\
       & + (8 \pi \eta_0 r_{ij}^{-3})(\sigma_i^2+\sigma_j^2)(\boldsymbol{I}-3 \boldsymbol{P}_{ij}),
       \end{split}

where
:math:`\boldsymbol{P}_{ij} = \Big(\boldsymbol{I}+\frac{\boldsymbol{r} \otimes \boldsymbol{r}}{r^2} \Big)`.
The mobility tensor for rotation, however, not correcting for the bead
radii, is :cite:p:`Carrasco1999a`:

.. math::
   :label: mu_rr_appdx

   \label{mu_rr_appdx}
       \begin{split}
       \boldsymbol{\mu}^{rr}_{ij} = & \delta_{ij} (8 \pi \eta_0 \sigma_i^3)^{-1} \boldsymbol{I} \\
       & + (1 - \delta_{ij})(16 \pi \eta_0 r^3_{ij})^{-1} (3 \boldsymbol{P}_{ij} - \boldsymbol{I}).
       \end{split}

Here, again, the first term is just the rotational mobility of the
single bead and the second term accounts for the hydrodynamic
interactions. In this formulation, there is still a correction for the
bead radii missing. This correction consists of adding
:math:`6 \eta_0 V_m \boldsymbol{I}` to the diagonal components of the
rotational friction tensor :math:`\boldsymbol{\Xi}^{rr}_O`, where
:math:`V_m` is the total volume of the rigid bead molecule
:cite:p:`Torre1983, Carrasco1999a`.

The rotation-translation coupling is given by
:cite:p:`Carrasco1999a`:

.. math::
   :label: mu_rt_appdx

   \label{mu_rt_appdx}
       \boldsymbol{\mu}^{rt}_{ij} = (1-\delta_{ij}) (8 \pi \eta_0 r_{ij}^2)^{-1} \boldsymbol{\epsilon}\boldsymbol{\hat{r}}_{ij},

where :math:`\boldsymbol{\epsilon}` is the Levi-Civita tensor.
:math:`\boldsymbol{\mu}^{tt}, \boldsymbol{\mu}^{rr}, \boldsymbol{\mu}^{rt}`
describe the mobility of a multi-sphere system with hydrodynamic
interactions. The above can be extended to account for rigid bead
molecules :cite:p:`Carrasco1999a` as outlined in the next
section.

The friction tensor for rigid bead molecules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, we closely follow :cite:p:`Carrasco1999a`. To get an
expression for the friction tensor of a rigid bead molecule, we start by
considering a system of :math:`N` free spherical beads in a fluid with
viscosity :math:`\eta_0`. Each sphere laterally moves at some velocity
:math:`\boldsymbol{u}_i` and rotates with some angular velocity
:math:`\boldsymbol{\omega}_i`. The spheres will experience a frictional
force and torque :math:`\boldsymbol{F}_i,  \boldsymbol{T}_i`. In the
non-inertial regime (Stokes regime), the relationship between the
force/torque and the velocities is linear:

.. math::
   :label: eq:FrictionForce_appdx

   \label{eq:FrictionForce_appdx}
       \boldsymbol{F}_i = \sum_{j=1}^N  \boldsymbol{\xi}_{ij}^{tt} \cdot \boldsymbol{u}_j +  \boldsymbol{\xi}_{ij}^{tr} \cdot \boldsymbol{\omega}_j

.. math::
   :label: eq:FrictionTorque_appdx

   \label{eq:FrictionTorque_appdx}
       \boldsymbol{T}_i = \sum_{j=1}^N  \boldsymbol{\xi}_{ij}^{rt} \cdot \boldsymbol{u}_j +  \boldsymbol{\xi}_{ij}^{rr} \cdot \boldsymbol{\omega}_j .

The :math:`\boldsymbol{\xi}_{ij}^{ab}, a,b \in \{t,r\}` are the (3x3)
friction matrices, connecting the amount of friction a particle i
experiences due to the presence of particle j moving through the fluid
at velocities :math:`\boldsymbol{u}_j, \boldsymbol{\omega}_j`. We may
rewrite Eqs. :math:numref:`eq:FrictionForce_appdx`,
:math:numref:`eq:FrictionTorque_appdx` in matrix form as:

.. math::
   :label: ForceTorque_appdx

   \label{ForceTorque_appdx}
       \begin{pmatrix}
       \boldsymbol{F} \\
       \boldsymbol{T} \\
       \end{pmatrix}
       =
       \begin{pmatrix}
       \boldsymbol{\xi}^{tt} & \boldsymbol{\xi}^{tr} \\
       \boldsymbol{\xi}^{rt} & \boldsymbol{\xi}^{rr} \\
       \end{pmatrix}
       \begin{pmatrix}
       \boldsymbol{U} \\
       \boldsymbol{W} \\
       \end{pmatrix},

where
:math:`\boldsymbol{F} = (\boldsymbol{F}_1, ..., \boldsymbol{F}_N)^T`,
:math:`T = (\boldsymbol{T}_1, ..., \boldsymbol{T}_N)^T` and
:math:`\boldsymbol{U} = (\boldsymbol{u}_1, ..., \boldsymbol{u}_N)^T`,
:math:`W = (\boldsymbol{\omega}_1, ..., \boldsymbol{\omega}_N)^T`. Here
:math:`\boldsymbol{\xi}^{ab}, a,b \in \{t,r\}` are of dimension (3Nx3N),
forming the friction supermatrix of dimension (6N,6N). The inverted
friction supermatrix is the mobility supermatrix.

.. math::
   :label: eq:mobility_supermatrix_appdx

   \label{eq:mobility_supermatrix_appdx}
       \begin{pmatrix}
       \boldsymbol{\mu}^{tt} & \boldsymbol{\mu}^{tr} \\
       \boldsymbol{\mu}^{rt} & \boldsymbol{\mu}^{rr} \\
       \end{pmatrix}
       =
       \begin{pmatrix}
       \boldsymbol{\xi}^{tt} & \boldsymbol{\xi}^{tr} \\
       \boldsymbol{\xi}^{rt} & \boldsymbol{\xi}^{rr} \\
       \end{pmatrix}^{-1}

Next, we consider not a system of N free beads, but a rigid bead model,
i.e., the beads are rigidly connected. Thereby, all beads move together
with some translational velocity :math:`\boldsymbol{u}_{O}`. Let the
body’s frame of reference lie at the center of diffusion of the bead
model :math:`\boldsymbol{r}_O` and let :math:`\boldsymbol{\omega}` be
the angular velocity of the rigid bead model. Then, in addition to the
translational velocity of the molecule’s center, each bead experiences a
translation velocity due to the rotation
:math:`\boldsymbol{\omega} \times \boldsymbol{r}_i`, where
:math:`\boldsymbol{r}_i` is the position vector from the molecules
origin :math:`\boldsymbol{r}_O` (in the body frame of reference).
Thereby, the total velocity is:

.. math::
   :label: Velocity_appdx

   \label{Velocity_appdx}
       \boldsymbol{u}_i = \boldsymbol{u}_O + \boldsymbol{\omega} \times \boldsymbol{r}_i

The force that a single bead experiences due to the movement of all the
other beads is:

.. math::
   :label: FrictionForce_Bead_appdx

   \label{FrictionForce_Bead_appdx}
       \boldsymbol{F}_i = \sum_{j=1}^N \boldsymbol{\xi}_{ij}^{tt} \cdot (\boldsymbol{u}_O + \boldsymbol{\omega} \times \boldsymbol{r}_j) + \boldsymbol{\xi}_{ij}^{tr} \cdot \boldsymbol{\omega},

and the torque that a single bead experiences due to the movement of all
the other beads is:

.. math::
   :label: FrictionTorque_Bead_appdx

   \label{FrictionTorque_Bead_appdx}
       \boldsymbol{T}_{P,i} = \sum_{j=1}^N \boldsymbol{\xi}_{ij}^{rt} \cdot (\boldsymbol{u}_O + \boldsymbol{\omega} \times \boldsymbol{r}_j) + \boldsymbol{\xi}_{ij}^{rr} \cdot \boldsymbol{\omega} .

From these expressions, we get the total force acting at the rigid body
origin by summation over all beads:

.. math::
   :label: FrictionForce_Total_appdx
   
   \label{FrictionForce_Total_appdx}
       \boldsymbol{F} = \sum_{i=1}^N \sum_{j=1}^N \boldsymbol{\xi}_{ij}^{tt} \cdot (\boldsymbol{u}_O + \boldsymbol{\omega} \times \boldsymbol{r}_j) + \boldsymbol{\xi}_{ij}^{tr} \cdot \boldsymbol{\omega}

For the total torque, however, we get an extra term.
:math:`\boldsymbol{T}_{P,i}` is only the torque acting on bead i
relative to it’s center, i.e., the center of the sphere. Thereby, this
only describes the amount of rotation bead i would experience around its
center due to the movement of all the other beads. However, the force
:math:`\boldsymbol{F}_{i}` acting on bead i due to the movement of the
other beads also results in a torque with which bead i acts on the rigid
bead models center :math:`\boldsymbol{r}_O`:

.. math::
   :label: FrictionTorque_Exk_appdx
   
   \label{FrictionTorque_Exk_appdx}
       \boldsymbol{r}_i \times \boldsymbol{F}_i = \boldsymbol{r}_i \times \Big( \sum_j^N \boldsymbol{\xi}_{ij}^{tt} (\boldsymbol{u}_O + \boldsymbol{\omega} \times \boldsymbol{r}_j) + \boldsymbol{\xi}_{ij}^{tr} \boldsymbol{\omega} \Big)

Thereby, the total torque acting on the rigid bead model’s origin is:

.. math::
   :label: FrictionTorque_Total_appdx
   
   \label{FrictionTorque_Total_appdx}
       \boldsymbol{T}_O = \sum_i^N \boldsymbol{T}_{P,i} +  \boldsymbol{r}_i \times \boldsymbol{F}_i = \sum_{i=1}^N \sum_{j=1}^N \boldsymbol{\xi}_{ij}^{rt} \cdot (\boldsymbol{u}_O + \boldsymbol{\omega} \times \boldsymbol{r}_j) + \boldsymbol{\xi}_{ij}^{rr} \cdot \boldsymbol{\omega} + \boldsymbol{r}_i \times \Big( \boldsymbol{\xi_{ij}}^{tt} (\boldsymbol{u}_O + \boldsymbol{\omega} \times \boldsymbol{r}_j) + \boldsymbol{\xi}_{ij}^{tr} \omega \Big).

The above can be transformed into a general expression in simpler matrix
form. For this, a little trick can be used to get rid of the cross
product by turning :math:`\boldsymbol{\omega} \times \boldsymbol{r}`
into the dot product :math:`- \boldsymbol{A} \cdot \boldsymbol{\omega}`
(note: the sign changed, because of the anticommutativity of the cross
product). After some rearranging, we end up with:

.. math::
   :label: FrictionForce_Total_2_appdx
   
   \label{FrictionForce_Total_2_appdx}
       \boldsymbol{F} = \Big( \sum_{i=1}^N \sum_{j=1}^N \boldsymbol{\xi}_{ij}^{tt} \Big) \cdot \boldsymbol{u}_O + \Big( \sum_{i=1}^N \sum_{j=1}^N - \boldsymbol{\xi}_{ij}^{tt} \cdot \boldsymbol{A}_j + \boldsymbol{\xi}_{ij}^{tr} \Big) \cdot \boldsymbol{\omega}

.. math::
   :label: FrictionTorque_Total_2_appdx
   
   \label{FrictionTorque_Total_2_appdx}
       \boldsymbol{T} = \Big( \sum_{i=1}^N \sum_{j=1}^N \boldsymbol{\xi}_{ij}^{rt} + A_i \boldsymbol{\xi}_{ij}^{tt} \Big) \cdot \boldsymbol{u}_O + \Big( \sum_{i=1}^N \sum_{j=1}^N \boldsymbol{\xi}_{ij}^{rt} \cdot \boldsymbol{A}_j + \boldsymbol{\xi}_{ij}^{rr} - A_i \boldsymbol{\xi}_{ij}^{tt} A_j  + A_i \boldsymbol{\xi}_{ij}^{tr} \Big) \cdot \boldsymbol{\omega}.

If we now write this in matrix form, similar to the free bead example
from above, we get:

.. math::
   :label: ForceTorque_Bead_appdx
   
   \label{ForceTorque_Bead_appdx}
       \begin{pmatrix}
       \boldsymbol{F} \\
       \boldsymbol{T}_O \\
       \end{pmatrix}
       =
       \begin{pmatrix}
       \boldsymbol{\Xi}^{tt} & \boldsymbol{\Xi}^{tr} \\
       \boldsymbol{\Xi}^{rt} & \boldsymbol{\Xi}^{rr} \\
       \end{pmatrix}
       \begin{pmatrix}
       \boldsymbol{u}_O \\
       \boldsymbol{\omega} \\
       \end{pmatrix},

Where we call :math:`\boldsymbol{\Xi}` the friction tensor of the rigid
bead molecule :cite:p:`Carrasco1999a` :

.. math::
   :label: Xi_appdx
   
   \label{Xi_appdx}
       \begin{split}
       &\boldsymbol{\Xi}^{tt} = \sum_{i=1}^N \sum_{j=1}^N \boldsymbol{\xi}_{ij}^{tt} \\
       &\boldsymbol{\Xi}_{O}^{tr} = \sum_{i=1}^N \sum_{j=1}^N ( -\boldsymbol{\xi}_{ij}^{tt} \cdot \boldsymbol{A}_j + \boldsymbol{\xi}_{ij}^{tr} ) \\
       &\boldsymbol{\Xi}_{O}^{rt} = \sum_{i=1}^N \sum_{j=1}^N ( \boldsymbol{A}_j \cdot \boldsymbol{\xi}_{ij}^{tt} + \boldsymbol{\xi}_{ij}^{rt} ) \\
       &\boldsymbol{\Xi}_{O}^{rr} = \sum_{i=1}^N \sum_{j=1}^N ( \boldsymbol{\xi}_{ij}^{rr} - \boldsymbol{\xi}_{ij}^{rt} \cdot \boldsymbol{A}_j + \boldsymbol{A}_i \cdot \boldsymbol{\xi}_{ij}^{tr} - \boldsymbol{A}_i \cdot \boldsymbol{\xi}_{ij}^{tt} \boldsymbol{A}_j)
       \end{split}

The :math:`\boldsymbol{\xi}`, are calculated from the inverse of the
mobility supermatrix (Eq.
:math:numref:`eq:mobility_supermatrix_appdx`).

A super Matrix
:math:`\boldsymbol{M}=[[\boldsymbol{M}_1, \boldsymbol{M}_2], [\boldsymbol{M}_3, \boldsymbol{M}_4]]`
is invertible, if both the diagonal blocks, :math:`\boldsymbol{M}_1` and
:math:`\boldsymbol{M}_4` are invertible The inverse of a (2x2)
supermatrix can be calculated by :cite:p:`Varadarajan2004`,
:cite:p:`Deligne1996`:

.. math::
   :label: supermatrix_inverse_appdx
   
   \label{supermatrix_inverse_appdx}
       \begin{split}
       & \boldsymbol{T}_1 = (\boldsymbol{M}_1 - \boldsymbol{M}_2 \boldsymbol{M}_4^{-1} \boldsymbol{M}_3)^{-1} \\
       & \boldsymbol{T}_2 = -\boldsymbol{M}_1^{-1} \boldsymbol{M}_2 (\boldsymbol{M}_4-\boldsymbol{M}_3 \boldsymbol{M}_1^{-1} \boldsymbol{M}_2)^{-1} \\
       & \boldsymbol{T}_3 = -\boldsymbol{M}_4^{-1} \boldsymbol{M}_3 (\boldsymbol{M}_1-\boldsymbol{M}_2 \boldsymbol{M}_4^{-1} \boldsymbol{M}_3)^{-1} \\
       & \boldsymbol{T}_4 = (\boldsymbol{M}_4 - \boldsymbol{M}_3 \boldsymbol{M}_1^{-1} \boldsymbol{M}_2)^{-1} \\
       \end{split}
