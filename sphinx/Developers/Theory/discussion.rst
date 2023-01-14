.. _`sec:discussion_pyrid`:

Discussion
==========

PyRID is a fast and flexible tool for particle-based reaction diffusion
simulations with pair-interactions. However, a challenge remains in that
PyRID is not able simulate processes that take several seconds or even
minutes. However, in cell biology we find many processes that act on
such time scales. This is true for signaling processes, for
self-assembly processes, e.g., of clathrin, and for protein trafficking.
Other tools that follow a similar approach such as ReaDDy
:cite:p:`Hoffmann2019, Schoeneberg2013` also do not provide
methods that would enable simulations on such long time scales. Tools
such a MCell :cite:p:`Kerr2008` and Smoldyn
:cite:p:`Andrews2016` enable simulations on larger time
scales but do not resolve molecular structure and are not able to
simulate protein binding and assembly. However, alternative
reaction-rate based approaches have been developed that account for
molecule structure, binding, and diffusion. A prominent example is
NERDSS :cite:p:`Varga2020` that is able to resolve fast
binding reactions as well as processes on large time and- spatial
scales. In NERDSS, molecules are represented by rigid bodies, similar to
PyRID. Also, excluded volume is accounted for by rejection sampling.
However, NERDSS avoids any energy interaction functions but instead
molecules "snap" into place in a predefined way when a binding reaction
is executed. Thereby, assembly processes can be simulated very
efficiently. Still, since the resulting assembly has a predefined form
NERDSS is not able make predictions about the structure of protein
assemblies. Also, binding reactions are not orientation dependent which
can result in unrealistic binding events, whereas in PyRID orientation
dependence is accounted for by construction. Also, with NERDSS, one
relies on reaction rates that have been measured either in experiment or
by molecular dynamics simulations to describe assembly and disassembly
processes. However, also with PyRID such processes can not accurately be
modeled without exactly specifying the energy functions of the binding
interaction which can even be harder than estimating rates. At last, due
to the lack of interaction forces in NERDSS, any physical properties
that are derived from the interaction forces or the energy functions can
not be computed and flexible chains of beads or molecules are not
supported. Still, such solely rate based approaches are very promising
if one is interested in the kinetics of complex assembly processes and
could also be a valuable addition to PyRID. In principle the PyRID
framework would allow rigid body assembly growth so this could be a
useful future extension. However, as indicated above, there exist many
settings in which we need to include energy functions and where, as a
result, there is a an upper limit for the integration time step that we
can choose. We can slightly shift this threshold towards larger time
steps by using different approximations to the inter-molecular
interaction energy functions but even then we are restricted to
integration time steps :math:`\leq 1 ns`. As such we need to speed up
computation, e.g. by parallelization. For very large system simulations
many molecular dynamics tools such as LAMMPS support parallel
implementations of their algorithms for the message passing interface
standard (MPI). However, here, we only gain a benefit in speed for large
systems as message passing otherwise becomes a bottle neck. For
intermediate sized system such as those that we want to simulate with
PyRID containing 10.000-100.000 particles, algorithms that run on the
GPU are much more promising. A good example is the molecular dynamics
tool HooMD :cite:p:`Anderson2020` that is optimized for the
GPU and that can reach speed ups of up to two order of magnitude
compared to a single CPU and more than one order of magnitude compared
to a modern multi-core CPU :cite:p:`Anderson2020`. As such,
bringing particle-based reaction diffusion simulations to the GPU could
be the key for simulations on time scales of even minutes. The question
remains to what degree the required algorithms and data structures can
be efficiently ported to the GPU. However, this is beyond the scope of
this work. At last, machine learning long found its way into MD
simulations and is used in coarse graining, molecular kinetics and more
:cite:p:`Noe2020`.

On hydrodynamic interactions
----------------------------

As mentioned above, PyRID does not account for hydrodynamic interactions
between molecules because, in this case, the kind of simulations for
which PyRID has been developed would become unfeasible. Here, the 6Nx6N
diffusion tensor of the entire system is needed to propagate the
molecule positions. As the molecule positions change each time step,
this diffusion tensor needs to be recalculated each iteration. A
discussion on this topic in terms of many particle simulations can also
be found in :cite:p:`Geyer2011`. A new algorithm that scales
:math:`O(N^2)` has been introduced by :cite:p:`Geyer2009`
making larger simulations with hydrodynamic interactions more feasible.

Limitations of the Brownian dynamics approach
---------------------------------------------

Brownian dynamics simulations come with some limitations that one should
consider :cite:p:`Snook2007`: In BD, only time steps are
considered that are much longer than the velocity relaxation time
:math:`\tau_{rel} =\frac{m}{\gamma} = \frac{2 \rho r^2}{9 \eta}`. Due to
the strong damping forces in a viscous fluid the kinetic energy of large
molecules rapidly dissipates. Thereby, the erratic movement of the
molecules in between two time steps is memory less and can be described
as a Markov process. However, when accounting for interactions between
molecules, the integration time step must also not be too small such
that forces stay approximately constant in one time step. This becomes a
problem for small interacting molecules or atoms, where the time step
needs to be chosen small enough to resolve the interactions but large
enough for the the approximation of over-damped kinetics. Thereby, if
the molecules are of similar size as the solvent molecules, BD may not
correctly describe the dynamics. :cite:p:`Winter2009`
introduced a Langevin integration scheme that enables the accurate
simulation of small molecules. Here, we will however use the well
established BD scheme introduced by :cite:p:`Ermak1978`.
Another thing to keep in mind is that BD is only applicable for
Newtonian fluids. Also, since the details of the interaction between
solvent particles and bead particles are neglected, simulations of
molecule aggregation may not be correctly described. However, in the
case where aggregation is dominated by the interaction between the
proteins, the latter may be negligible.
