#!/bin/bash

# Define particle numbers
particle_counts=($(seq 100 1000 10100))

# Initial box size and reference particle number
L0=20.0  # Box size for 100 particles
N0=100   # Reference number of particles

# Output header
echo "Particles  BoxSize(nm)  Runtime(s)  MemoryUsage(MB)" > runtime.txt

for num in "${particle_counts[@]}"; do
    # Compute new box size
    L_new=$(echo "scale=6; $L0 * e(l($num / $N0) / 3)" | bc -l)
    min_bound=$(echo "scale=6; -$L_new / 2" | bc -l)
    max_bound=$(echo "scale=6; $L_new / 2" | bc -l)

    # Generate Smoldyn config file
    config_file="sim_${num}particles_${L_new}nm.txt"
    cat > "$config_file" <<EOF
# Smoldyn config for A + B -> C

graphics none

dim 3
max_species 10
max_mol 100000

species A
species B
species C

# Diffusion constant = 4.0 nm²/ns = 4000 µm²/s
difc A 4000
difc B 4000
difc C 4000

# Time settings for 10,000 steps of 1e-9 s
time_start 0
time_stop 1e-5
time_step 1e-9

# Periodic box boundaries
boundaries 0 $min_bound $max_bound p
boundaries 1 $min_bound $max_bound p
boundaries 2 $min_bound $max_bound p

# Molecule placement
mol $num A u u u
mol $num B u u u

# Reaction: A + B -> C with rate 35.0 (placeholder, adjust if needed)
reaction myfusion A + B -> C 450

# Output
output_files results_${num}particles_${L_new}nm.txt
cmd e molcount results_${num}particles_${L_new}nm.txt

EOF

    echo "Running Smoldyn with $num particles in box size $L_new nm..."

    # Measure runtime and memory
    start_time=$(date +%s%N)

    (/usr/bin/time -f "%M" -o mem_usage.txt smoldyn "$config_file") 2> smoldyn_log.txt

    end_time=$(date +%s%N)
    runtime_ns=$((end_time - start_time))
    runtime=$(echo "scale=6; $runtime_ns / 1000000000" | bc)

    # Read memory in KB and convert to MB
    mem_used_kb=$(cat mem_usage.txt)
    mem_used_mb=$(echo "scale=2; $mem_used_kb / 1024" | bc)

    echo "$num  $L_new  $runtime  $mem_used_mb" >> runtime.txt
    echo "Done: $num particles → Time: $runtime s, Memory: $mem_used_mb MB"
done

