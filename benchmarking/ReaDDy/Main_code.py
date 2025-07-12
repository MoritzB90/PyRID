#!/usr/bin/env python
# coding: utf-8



from math import sqrt
from joblib import Parallel, delayed
from Benchmark_ReaDDy import Main_Code
import os
import sys
import numpy as np


Int_Str= np.linspace(100,10100,11,dtype=int)
num_jobs=len(Int_Str)
processed_data = Parallel(n_jobs=num_jobs)(delayed(Main_Code)(interaction_strength) for interaction_strength in Int_Str)                   




