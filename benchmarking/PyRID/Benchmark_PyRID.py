#!/usr/bin/env python
# coding: utf-8

import pyrid as prd
import time
import sys
import numpy as np
import psutil
import os
import csv


def Main_Code(ia_id):
    print(f"Running simulation with {ia_id} particles")
    


    # Initialize memory measurement
    process = psutil.Process(os.getpid())  # Get current process


    # Define file paths
    file_path = os.path.expanduser("~/Expansion/BenchmarkPyRID/Files/") 
#    file_path = ''
    file_path = os.path.expanduser("~/Expansion/BenchmarkPyRID/Figures/") 
#    fig_path = '/Expansion/BenchmarkPyRID/Figures/'
    fig_path = 'Benchmark_PyRID' + str(ia_id)
    os.makedirs(file_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)
    # Basic simulation parameters
    nsteps = 10000
    dt = 1e-3  # 1e-3 time step in ns
    L0 = 20.0  # Initial box size in nm
    N0 = 100   # Initial number of particles
    L_new = L0 * (ia_id/ N0) ** (1/3)  # Adjust box size
    box_lengths = [L_new, L_new, L_new]  # New box dimensions
    Temp = 300.0  # Temperature in K
    eta = 1e-21  # Viscosity in ns·nm^2
    seed = 1000  # Random seed for reproducibility

    # Measure memory before simulation
    mem_before = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    print(f"Memory before simulation: {mem_before:.2f} MB")

    # Initialize PyRID simulation (disable file writing to avoid I/O bottleneck)
    Simulation = prd.Simulation(
        box_lengths=box_lengths,
        dt=dt,
        Temp=Temp,
        eta=eta,
        stride=1,
        write_trajectory=False,  # Disable writing trajectory files
        boundary_condition='periodic',
        nsteps=nsteps,
        seed=seed,
        wall_force=100.0,
        length_unit='nanometer',
        time_unit='ns'
    )

    # Register particle and molecule types
    Simulation.register_particle_type('AA', 1.5)
    Simulation.register_particle_type('BB', 1.5)
    Simulation.register_particle_type('CC', 1.5)
    Simulation.register_molecule_type('A', [[0.0, 0.0, 0.0]], ['AA'])
    Simulation.register_molecule_type('B', [[0.0, 0.0, 0.0]], ['BB'])
    Simulation.register_molecule_type('C', [[0.0, 0.0, 0.0]], ['CC'])

    # Set diffusion tensors (assuming they are computed by PyRID)
    for molecule in ['A', 'B', 'C']:
        D_rr = [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
        D_tt = [[4.0,0.0,0.0],[0.0,4.0,0.0],[0.0,0.0,4.0]]
        Simulation.set_diffusion_tensor(molecule, D_tt, D_rr)

    # Add reactions (optional, depending on the simulation)
    Simulation.add_bm_reaction('fusion', ['A', 'B'], ['C'], [['AA', 'BB']], [100.0], [3.0])

    # Distribute molecules randomly (again, no I/O)
    pos, mol_type_idx, quaternion = Simulation.distribute(
        'MC', 'Volume', 0, ['A', 'B'], [ia_id, ia_id], multiplier=0
    )
    Simulation.add_molecules('Volume', 0, pos, quaternion, mol_type_idx)

    # Start timer for benchmarking
    start_time = time.time()

    # Run the simulation (no I/O, just simulation)
    Simulation.run(progress_stride=1000, out_linebreak=False)



    # End timer for benchmarking
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total simulation time: {runtime:.4f} seconds")

    # Measure memory after simulation
    mem_after = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    mem_used = mem_after - mem_before
    print(f"Memory after simulation: {mem_after:.2f} MB")
    print(f"Total memory used: {mem_used:.2f} MB")


    # File paths for saving results
    file_path = 'Files/'
    file_name = file_path + "simulation_performance_results.txt"
    csv_file = file_path + "simulation_performance_results.csv"

    # Save performance results to a text file
    with open(file_name, "w") as f:
        f.write(f"Total simulation time: {runtime:.4f} seconds\n")
        f.write(f"Memory before simulation: {mem_before:.2f} MB\n")
        f.write(f"Memory after simulation: {mem_after:.2f} MB\n")
        f.write(f"Total memory used: {mem_used:.2f} MB\n")


    # Save results to a CSV file
    csv_header = [
        "Num_Particles", "Runtime (s)", "Memory Before (MB)", 
        "Memory After (MB)", "Total Memory Used (MB)", 
        "CPU Time/Step/Particle (s)"
    ]
    data_row = [
        ia_id, runtime, mem_before, 
        mem_after, mem_used
    ]

    # Append data to CSV file
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(csv_header)  # Write header only if file doesn't exist
        writer.writerow(data_row)

    print(f"Benchmark performance results saved to {file_name}")
    print(f"Benchmark performance results saved to {csv_file}")



















































