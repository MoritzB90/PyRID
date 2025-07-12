#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import readdy
import time
import psutil
import csv

def Main_Code(ia_id):
    print(f"Running simulation with {ia_id} particles")

    # Create output directory
    file_path = 'Files/'
    os.makedirs(file_path, exist_ok=True)

    # Scale simulation box based on particle count
    L0 = 20.0
    N0 = 100
    L_new = L0 * (ia_id / N0) ** (1 / 3)
    box_lengths = [L_new, L_new, L_new]

    # Define system
    system = readdy.ReactionDiffusionSystem(box_lengths, temperature=300. * readdy.units.kelvin)
    system.add_species("A", diffusion_constant=4.0)
    system.add_species("B", diffusion_constant=4.0)
    system.add_species("C", diffusion_constant=4.0)

    # Define bimolecular fusion reaction: A + B -> C
    lambda_on = 1.0
    system.reactions.add("myfusion: A +(3) B -> C", rate=lambda_on / readdy.units.nanosecond)

    # Create simulation object (no output file, no trajectory)
    simulation = system.simulation(kernel="CPU")
    simulation.reaction_handler = "Gillespie"

    # Add initial particles
    n_particles = ia_id
    initial_positions_a = np.random.random(size=(n_particles, 3)) * 20. - 10.
    initial_positions_b = np.random.random(size=(n_particles, 3)) * 20. - 10.
    simulation.add_particles("A", initial_positions_a)
    simulation.add_particles("B", initial_positions_b)

    # Track memory and time
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    start_time = time.time()

    # Run simulation
    simulation.run(n_steps=10000, timestep=1e-3 * readdy.units.nanosecond)

    end_time = time.time()
    mem_after = process.memory_info().rss / (1024 * 1024)
    runtime = end_time - start_time
    mem_used = mem_after - mem_before

    print(f"Memory before: {mem_before:.2f} MB | after: {mem_after:.2f} MB | used: {mem_used:.2f} MB")
    print(f"Runtime: {runtime:.2f} seconds")

    # Save performance metrics to CSV
    csv_file = f"{file_path}readdy_performance.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Num_Particles", "Memory_Before_MB", "Memory_After_MB", "Memory_Used_MB", "Runtime_s"])
        writer.writerow([ia_id, mem_before, mem_after, mem_used, runtime])

# Example usage:
# Main_Code(100)

