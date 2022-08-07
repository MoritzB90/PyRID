====================
Rigid bead molecules
====================

Proteins and other molecules are not point like particles. Especially the interactions between proteins are not accurately described by isotropic energy potentials. Instead, the physical properties of bio-molecular systems emerge from an-isotropic multivalent interactions :cite:p:`Dignon2018, Espinosa2019`. Protein-protein interaction can be accurately simulated in atomistic molecular dynamics simulations. However, even modern computers and algorithms are not efficient enough to simulate systems with more than a few molecules. Therefore, coarse graining methods are needed :cite:p:`Tozzini2005`. Rigid bead models are a method of minimal coarse graining that have some important benefits. Strong and short ranged interactions between atoms are replaced by a rigid bead topology. This allows for integration time steps several orders larger than in atomistic simulations. Usually, the beads of a rigid bead model do not represent single atoms but pattern the geometry of the molecule of interest :cite:p:`Torre2001`, significantly reducing the overall number of particles that need to be simulated. In addition, experimentally or theoretically estimated diffusion tensors can be used to accurately describe the diffusive motion of molecules. Importantly, multivalent protein-protein interactions can be described by patches located on the bead model surface. On the downside, the properties of coarse grained model systems strongly depend on the choice of interaction potentials and other model parameters. The estimation of these model parameters is fairly involved and is out of the scope of this work.

The position of each bead i of molecule j can be characterized by

.. math::
    \boldsymbol{R}_i(t) = \boldsymbol{R}_{i}^{local}(t) + \boldsymbol{R}_{j}^{O}(t)

where 

.. math::
    \boldsymbol{R}_{i}^{local}(t) = \boldsymbol{A}_j(t) \cdot \boldsymbol{X}_{i}^{local} .

Here :math:`\boldsymbol{X}_{i}^{local}` are the coordinates of bead i in the local reference frame, and :math:` \boldsymbol{A}(t)` and :math:`\boldsymbol{R}_{j}^{O}(t)` are the rotation matrix and center of diffusion of molecule j in the lab reference frame respectively.
The center of diffusion propagates in response to external forces :math:`\boldsymbol{F}(t)` exerted, e.g., by particle-particle interactions or an external force field, and due to hydrodynamic interactions and collisions of the beads with solvent molecules (Brownian motion).
Thereby, the total force :math:`\boldsymbol{F}(t)` acting on the molecules' center of diffusion is the sum of all forces :math:`\boldsymbol{f}_i(t)` acting on the individual beads:

.. math::
	\boldsymbol{F}(t) = \sum_{i = 1}^{N_{beads}}. \boldsymbol{f}_i(t)


The orientation/rotation of the molecule is best described by a unit quaternion :math:`\boldsymbol{q}(t)`. The rotation quaternion propagates in response to the torque :math:`\boldsymbol{T}_i(t) = \boldsymbol{F}_i(t) \times \boldsymbol{r}_{ij}` exerted by the external forces, where :math:`\boldsymbol{r}_{ij}` is the distance vector between bead i and the center of diffusion of molecule j. The rotation matrix can be represented in terms of rotation quaternion by (:func:`pyrid.molecules.rigidbody_util.RBs.calc_orientation_quat`)

.. math::
    \begin{split}
    \boldsymbol{A}
    = 
    \begin{pmatrix}
        1-2(q_2^2+q_3^2) & 2(q_1 q_2-q_0 q_3) & 2(q_1 q_3+q_0 q_2) \\
        2(q_1 q_2+q_0 q_3) & 1-2(q_1^2+q_3^2) & 2(q_2 q_3-q_0 q_1) \\
    2(q_1 q_3-q_0 q_2) & 2(q_2 q_3+q_0 q_1) & 1-2(q_1^2+q_2^2) \\
    \end{pmatrix}.
    \end{split}