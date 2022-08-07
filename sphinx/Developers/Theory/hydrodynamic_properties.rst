=======================
Hydrodynamic properties
=======================

The well known Einstein equation :math:`D = \mu k_B T = \frac{k_B T}{f}`, where :math:`\mu` is the mobility (defined as the ratio of drift velocity and applied force) and :math:`f` the friction coefficient, often better known in the form of the `Stokes-Einstein equation` :math:`D = \frac{k_B T}{6 \pi \eta r}`, reads in its generalized form:

.. math::
   :label: Eisntein_gen

	\begin{pmatrix}
	D^{tt} & D^{rt}\\
	D^{tr} & D^{rr}
	\end{pmatrix}
	= k_B T
	\begin{pmatrix}
	\Xi^{tt} & \Xi^{rt}\\
	\Xi^{tr} & \Xi^{rr}
	\end{pmatrix}^{-1}.

Here, :math:`D^{tt}` and :math:`D^{rr}` are the translational and rotational diffusion tensors respectively and :math:`D^{rt} = D^{tr,T}` is the translational rotational coupling. :math:`\Xi^{tt}`, :math:`\Xi^{rr}` and :math:`\Xi^{rt} = \Xi^{tr,T}` are the translation, rotation, and translation rotation coupling friction tensors.
The Kirkwood-Riseman treatment of bead models assigns each bead a translational friction coefficient :math:`f_i = 6 \pi \eta_0 r_i`, where :math:`r_i` is the hydrodynamic radius or Stoke's radius.
The hydrodynamic interaction between the beads is represented by the Oseen tensor.

The friction tensors are related to the friction matrices of a bead :math:`\zeta^{tt}, \zeta^{r}, \zeta^{rt}=\zeta^{tr}` by

.. math::
	:label: Friction

	\begin{align*}
	&\Xi^{tt} = \sum_{i=1}^N \sum_{j=1}^N \zeta_{ij}^{tt} \\
	&\Xi_{O}^{tr} = \sum_{i=1}^N \sum_{j=1}^N ( -\zeta_{ij}^{tt} \cdot \boldsymbol{A}_j + \zeta_{ij}^{tr} ) \\
	&\Xi_{O}^{rt} = \sum_{i=1}^N \sum_{j=1}^N ( \boldsymbol{A}_j \cdot \zeta_{ij}^{tt} + \zeta_{ij}^{rt} ) \\
	&\Xi_{O}^{rr} = \sum_{i=1}^N \sum_{j=1}^N ( \zeta_{ij}^{rr} - \zeta_{ij}^{rt} \cdot \boldsymbol{A}_j + \boldsymbol{A}_i \cdot \zeta_{ij}^{tr} - \boldsymbol{A}_i \cdot \zeta_{ij}^{tt} \boldsymbol{A}_j)
	\end{align*}

The friction matrices can be calculated from the inverse of the grand mobility matrix (simplified super matrix inverse):

.. math::
	:label: Friction2

	\begin{align*}
	&\zeta^{tt} = (\mu^{tt}-\mu^{tr} \cdot \mu^{rr} \cdot \mu^{rt})^{-1} \\
	&\zeta^{rr} = (\mu^{rr}-\mu^{rt} \cdot \mu^{tt} \cdot \mu^{tr})^{-1} \\
	&\zeta^{tr} = -(\mu^{rr})^{-1} \cdot \mu^{tr} \zeta^{tt} = -\zeta^{rr} \cdot \mu^{tr} \cdot (\mu^{tt})^{-1} \\
	\end{align*}