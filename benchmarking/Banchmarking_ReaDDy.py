# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""


import pyrid as prd
import numpy as np

#%%

for volfrac in [0.05,0.1,0.2,0.3,0.4]:
        
    file_path='Files/'
    fig_path = 'Figures/'
    file_name='Benchmarking_2'
        
    nsteps = 1e4
    stride = 1e4
    obs_stride = 1e4
    box_lengths = np.array([100.0,100.0,100.0])
    Temp=293.15
    eta=1e-21
    dt = 0.1
    

    Simulation = prd.Simulation(box_lengths = box_lengths, 
                                dt = dt, 
                                Temp = Temp, 
                                eta = eta, 
                                stride = stride, 
                                write_trajectory = True, 
                                file_path = file_path, 
                                file_name = file_name, 
                                fig_path = fig_path, 
                                boundary_condition = 'periodic', 
                                nsteps = nsteps, 
                                seed = 0, 
                                length_unit = 'nanometer', 
                                time_unit = 'ns')
    

    Simulation.register_particle_type('Core_1', 2.5) # (Name, Radius)

    k=100.0 #kJ/(avogadro*nm^2) 
    Simulation.add_interaction('harmonic_repulsion', 'Core_1', 'Core_1', {'k':k}, bond = False)


    A_pos = [[0.0,0.0,0.0]]
    A_types = ['Core_1']

    Simulation.register_molecule_type('A', A_pos, A_types)
    D_tt, D_rr = prd.diffusion_tensor(Simulation, 'A')
    Simulation.set_diffusion_tensor('A', D_tt, D_rr)


    V = box_lengths.prod()
    VA = 4/3*np.pi*2.5**3
    n = int(V/(VA)*volfrac)
    print('n = ', n)
    print(volfrac, VA*n/V)
    
    points, points_types, quaternion = Simulation.distribute('MC', 'Volume', 0, ['A'], [n])
    
    Simulation.add_molecules('Volume',0, points, quaternion, points_types)

        
    Simulation.run(progress_stride = 1000, out_linebreak = False)
    Simulation.print_timer()
    
#%%

for n in [1000,2000,5000,10000,20000]:
        
    file_path='Files/'
    fig_path = 'Figures/'
    file_name='Benchmarking_3'
        
    nsteps = 1e4
    stride = 1e4
    obs_stride = 1e4
    
    volfrac = 0.3
    VA = 4/3*np.pi*2.5**3
    V = n*VA/volfrac
    L = V**(1/3)
    print(L)
    
    box_lengths = np.array([L,L,L])
    Temp=293.15
    eta=1e-21
    dt = 0.1

    Simulation = prd.Simulation(box_lengths = box_lengths, 
                                dt = dt, 
                                Temp = Temp, 
                                eta = eta, 
                                stride = stride, 
                                write_trajectory = True, 
                                file_path = file_path, 
                                file_name = file_name, 
                                fig_path = fig_path, 
                                boundary_condition = 'periodic', 
                                nsteps = nsteps, 
                                seed = 0, 
                                length_unit = 'nanometer', 
                                time_unit = 'ns')
    

    Simulation.register_particle_type('Core_1', 2.5) # (Name, Radius)

    k=100.0 #kJ/(avogadro*nm^2) 
    Simulation.add_interaction('harmonic_repulsion', 'Core_1', 'Core_1', {'k':k}, bond = False)


    A_pos = [[0.0,0.0,0.0]]
    A_types = ['Core_1']

    Simulation.register_molecule_type('A', A_pos, A_types)
    D_tt, D_rr = prd.diffusion_tensor(Simulation, 'A')
    Simulation.set_diffusion_tensor('A', D_tt, D_rr)
    
    
    points, points_types, quaternion = Simulation.distribute('MC', 'Volume', 0, ['A'], [n])
    
    Simulation.add_molecules('Volume',0, points, quaternion, points_types)

        
    Simulation.run(progress_stride = 1000, out_linebreak = False)
    Simulation.print_timer()


#%%

import pyrid as prd
import numpy as np

pus = [[],[],[],[],[]]
numbers = [1000,2000,5000,10000,20000]

for trials in range(3):
    for i,n in enumerate(numbers):
            
        file_path='Files/'
        fig_path = 'Figures/'
        file_name='Benchmarking_3'
            
        nsteps = 3000
        stride = 3001
        obs_stride = 3001
        
        # volfrac = 0.3
        # VA = 4/3*np.pi*2.5**3
        # V = n*VA/volfrac
        # L = V**(1/3)
        rho_tot = 0.003141
        V = n/rho_tot
        L = V**(1/3)
        print(L)
        
        box_lengths = np.array([L,L,L])
        Temp=293.0
        eta=1e-21
        dt = 0.1
    
        Simulation = prd.Simulation(box_lengths = box_lengths, 
                                    dt = dt, 
                                    Temp = Temp, 
                                    eta = eta, 
                                    stride = stride, 
                                    write_trajectory = True, 
                                    file_path = file_path, 
                                    file_name = file_name, 
                                    fig_path = fig_path, 
                                    boundary_condition = 'periodic', 
                                    nsteps = nsteps, 
                                    seed = 0, 
                                    length_unit = 'nanometer', 
                                    time_unit = 'ns', max_cells_per_dim = 100)
        
    
        Simulation.register_particle_type('Core_1', 1.5) # (Name, Radius)
        Simulation.register_particle_type('Core_2', 3.0) # (Name, Radius)
        Simulation.register_particle_type('Core_3', 3.12) # (Name, Radius)
        
        k=10.0 #kJ/(avogadro*nm^2) 
        Simulation.add_interaction('harmonic_repulsion', 'Core_1', 'Core_1', {'k':k}, bond = False)
        Simulation.add_interaction('harmonic_repulsion', 'Core_1', 'Core_2', {'k':k}, bond = False)
        Simulation.add_interaction('harmonic_repulsion', 'Core_1', 'Core_3', {'k':k}, bond = False)
        
        Simulation.add_interaction('harmonic_repulsion', 'Core_2', 'Core_2', {'k':k}, bond = False)
        Simulation.add_interaction('harmonic_repulsion', 'Core_2', 'Core_3', {'k':k}, bond = False)
        
        Simulation.add_interaction('harmonic_repulsion', 'Core_3', 'Core_3', {'k':k}, bond = False)
    
        A_pos = [[0.0,0.0,0.0]]
        A_types = ['Core_1']
        Simulation.register_molecule_type('A', A_pos, A_types)
        D_tt, D_rr = prd.diffusion_tensor(Simulation, 'A')
        Simulation.set_diffusion_tensor('A', D_tt, D_rr)
        
        B_pos = [[0.0,0.0,0.0]]
        B_types = ['Core_2']
        Simulation.register_molecule_type('B', B_pos, B_types)
        D_tt, D_rr = prd.diffusion_tensor(Simulation, 'B')
        Simulation.set_diffusion_tensor('B', D_tt, D_rr)
        
        C_pos = [[0.0,0.0,0.0]]
        C_types = ['Core_3']
        Simulation.register_molecule_type('C', C_pos, C_types)
        D_tt, D_rr = prd.diffusion_tensor(Simulation, 'C')
        Simulation.set_diffusion_tensor('C', D_tt, D_rr)
        
        
        Simulation.add_um_reaction('fission', 'C', 5e-5, ['A']+['B'], [0]+[0], [1]+[1], 4.5)
        Simulation.add_bm_reaction('fusion', ['A', 'B'], ['C'], [['Core_1', 'Core_2']], [1e-3], [4.5])
        
        
        points, points_types, quaternion = Simulation.distribute('MC', 'Volume', 0, ['A', 'B', 'C'], [int(n/4),int(n/4),int(n/2)])
        
        Simulation.add_molecules('Volume',0, points, quaternion, points_types)
    
            
        Simulation.run(progress_stride = 100, out_linebreak = False)
        Simulation.print_timer()
        
        pus[i].append(Simulation.Timer['pu/s']/(Simulation.current_step-1))
    
#%%

import numpy as np
import pandas as pd
from read_roi import read_roi_zip

nsteps = 3000
numbers = [1000,2000,5000,10000,20000]

pus_PyRID = [[746607.3566427813, 761691.6824159948, 791678.5012554127],
 [776981.7085875442, 855600.7148515738, 637153.6546318715],
 [816648.6313926873, 828947.4364052081, 691325.676713187],
 [721581.8907992956, 796842.6131665349, 782437.9984112013],
 [668443.8410836522, 679604.6540441527, 663875.71594916]]

_times_readdy = [[4.4222217189999355, 4.157257560999824, 4.147727719999239],
 [9.349907376999909, 8.296020840999518, 8.292464236999876],
 [23.972112217999893, 23.47658878400034, 23.115324891000455],
 [56.91762770100013, 57.30933346399979, 57.21489187099996],
 [167.44886802099973, 166.4403209840002, 164.73981485000058]]

pus_readdy = nsteps/np.mean(_times_readdy, axis = 1)*np.array(numbers)

roi = read_roi_zip('D:\\Repositories\\PyRID\\validation\\RoiSet_ReaDDy_benchmark.zip')


X_log = (np.array(roi['dataPoints']['x'])-roi['xAxis']['x1'])
b = X_log[-1]/(np.log10(1e5)-np.log10(200))
a = np.log10(200)*b
pus_readdy_x = 10**((a+X_log)/b)

yAxis=0.8e-5/(roi['yAxis']['y2']-roi['yAxis']['y1'])
pus_readdy_y=((np.array(roi['dataPoints']['y'])-roi['yAxis']['y1'])*yAxis)


#%%

benchmark_hgrid = {}

benchmark_hgrid['n'] = np.array(numbers)
benchmark_hgrid['pu/s']= np.mean(pus_PyRID, axis = 1)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks")

plt.figure(figsize = (5,3), dpi = 150)
plt.semilogx(benchmark_hgrid['n'], 1/benchmark_hgrid['pu/s']*1e6, label = 'PyRID')
plt.semilogx(benchmark_hgrid['n'], 1/pus_readdy*1e6, label = 'ReaDDy')
plt.semilogx(pus_readdy_x[3:6],pus_readdy_y[3:6]*1e6, label = 'ReaDDy (Hoffmann 2019 et al.)')
# plt.axhline(0.12e-5)
# plt.xlim(1000,20000)
plt.ylim(0)
plt.xlabel('Number of particles')
plt.ylabel('Time per particle in $\mu s$')
plt.legend()
plt.savefig('Figures/Benachmark_comparison_ReaDDy', dpi = 300, bbox_inches="tight")

#%%

Vol = Simulation.System.volume
Vmol_A = 4/3*np.pi*2.5**3
Vmol_B = 4/3*np.pi*10.0**3
N_A = Simulation.System.Nmol[0]
N_B = Simulation.System.Nmol[1]

packing = (Vmol_A*N_A+Vmol_B*N_B)/Vol

#%%

import numpy as np
import pandas as pd

benchmark_hgrid = {}

benchmark_hgrid['n'] = np.array([763, 1527, 3055, 4583, 6111])
benchmark_hgrid['eta'] = np.array([763, 1527, 3055, 4583, 6111])*4/3*np.pi*2.5**3/box_lengths.prod()
benchmark_hgrid['it/s'] = np.array([929.769, 423.724, 183.558, 122.798, 83.422])
benchmark_hgrid['pu/s']= benchmark_hgrid['it/s']*benchmark_hgrid['n']


df = pd.DataFrame(benchmark_hgrid['it/s'])

dfm_its = pd.melt(df.reset_index(), id_vars='index', value_vars=['cell list',  'hierarchical grid', 'hierarchical grid \n single component'], var_name = 'method', value_name = 'it/s')

df = pd.DataFrame(benchmark_hgrid['pu/s'])

dfm_pus = pd.melt(df.reset_index(), id_vars='index', value_vars=['cell list',  'hierarchical grid', 'hierarchical grid \n single component'], var_name = 'method', value_name = 'pu/s')


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

plt.figure()
plt.plot(benchmark_hgrid['n'], benchmark_hgrid['pu/s'])
plt.ylim(0)

