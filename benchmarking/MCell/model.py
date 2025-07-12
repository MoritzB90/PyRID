import sys
import os
import time
import psutil
import csv
import numpy as np

# Set up MCELL_PATH and import mcell
MCELL_PATH = os.environ.get('MCELL_PATH', '')
if MCELL_PATH:
    sys.path.append(os.path.join(MCELL_PATH, 'lib'))
else:
    print("Error: MCELL_PATH not set.")
    sys.exit(1)

import mcell as m
def run_simulation(num_particles: int) -> tuple[float, float]:
    # Parameters for compartment size scaling
    L0 = 20.0  # Initial size of the cube (micrometers or nanometers, depending on system)
    N0 = 100    # Normalization constant

    # Calculate the new size of the cube based on the number of initial particles
    L_new = L0 * (num_particles / N0) ** (1 / 3)
    box_lengths = [L_new, L_new, L_new]  # Set the new dimensions for the cube

    # Override BNGL parameter values
    parameter_overrides = {
        'num_A_to_release': num_particles,
        'num_B_to_release': num_particles,
        'rate_A_B_to_C': 1e11  # Set rate to 1e11 (1/s) for the reaction rate
    }

    # Track memory and time before simulation
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
    start_time = time.time()

    # Create geometry with the dynamic box length
    cube = m.geometry_utils.create_box(name='CellCube', edge_dimension=L_new)  # Use L_new

    # Build model
    model = m.Model()
    model.add_geometry_object(cube)
    cube.is_bngl_compartment = True

    # Load BNGL with dummy observables output and parameter overrides
    MODEL_PATH = os.path.dirname(os.path.abspath(__file__))
    bngl_filename = os.path.join(MODEL_PATH, 'model.bngl')

    model.load_bngl(
        file_name=bngl_filename,
        observables_path_or_file="unused/",  # Prevent actual observable output
        parameter_overrides=parameter_overrides
    )

    # Run the simulation
    model.initialize()
    model.run_iterations(10000)  # Number of iterations to simulate
    model.end_simulation()

    # Track memory and time after simulation
    end_time = time.time()
    mem_after = process.memory_info().rss / (1024 * 1024)  # MB
    runtime = end_time - start_time
    mem_used = mem_after - mem_before

    return runtime, mem_used

def main():
    output_csv = "performance_metrics.csv"
    particle_counts = np.linspace(100, 10100, 11, dtype=int)

    # Write CSV header
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Particles', 'Runtime_s', 'Memory_Used_MB'])

    for count in particle_counts:
        print(f"--- Running simulation with {count} particles ---")
        runtime, mem_used = run_simulation(count)

        with open(output_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([count, round(runtime, 4), round(mem_used, 4)])

        print(f"Done: {count} particles → Time: {runtime:.2f}s, Mem: {mem_used:.2f}MB")


if __name__ == '__main__':
    main()

