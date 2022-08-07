system_util
===========

Short description

Compartment
-----------
.. autoclass:: pyrid.system.system_util.Compartment
   :members: calc_centroid_radius_AABB, add_border_2d, add_border_3d, add_group

MoleculeType
-------------
.. autoclass:: pyrid.system.system_util.MoleculeType
   :members: update_um_reaction

System
------
.. autoclass:: pyrid.system.system_util.System
   :members: add_box_compartment, add_system_grid, add_barostat_berendsen, register_molecule_type, set_diffusion_tensor, fixed_concentration_at_boundary, register_particle_type, set_compartments, add_border_3d, add_mesh, add_edges, add_neighbours, get_AABB_centers, create_cell_list, add_up_reaction, add_um_reaction, add_bp_reaction, add_interaction, add_bm_reaction