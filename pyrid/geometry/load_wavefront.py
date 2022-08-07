# -*- coding: utf-8 -*-
"""
Created on Mon May 23 17:00:59 2022

@author: Moritz
"""

import numpy as np

    
#%%

def load_compartments(path):
    
    """Reads an .obj file.
    
    Parameters
    ----------
    path : `string`
        Directory path to the obj file.
    
    Raises
    ------
    NotImplementedError (just an example)
        Brief explanation of why/when this exception is raised
    
    Returns
    -------
    `tuple(float64[:,3], int64[:,3], dict())`
        vertices, triangles, compartments
    
    """
    
    with open(path) as f:
        lines = f.readlines()
    
    vertices = []
    triangles = []
    triangle_id = 0
    compartments = {}
    current_group = '(null)'
    for l in lines:
        if l[0]=='o':
            compartments[l[2:-1]] = {'triangle_ids':[], 'face_groups':{}}
            current_comp = l[2:-1]
            current_group = '(null)'
            
        if l[0]=='g':
            current_group = l[2:-1]
            if current_group not in compartments[current_comp]['face_groups'] and current_group != '(null)':
                compartments[current_comp]['face_groups'][current_group] = []
            
        if l[0]=='v':
            if current_comp == 'Box':
                vertices.append([float(x)*0.99999 for x in l[2:-1].split()])
            else:
                vertices.append([float(x) for x in l[2:-1].split()])
            
        if l[0]=='f':
            triangles.append([int(x)-1 for x in l[2:-1].split()])
            compartments[current_comp]['triangle_ids'].append(triangle_id)
            
            if current_group != '(null)':
                compartments[current_comp]['face_groups'][current_group].append(triangle_id)
                
            triangle_id += 1
                
    for key in compartments:
        compartments[key]['triangle_ids'] = np.array(compartments[key]['triangle_ids'], dtype = np.int64)
        
        for group in compartments[key]['face_groups']:
            compartments[key]['face_groups'][group] = np.array(compartments[key]['face_groups'][group], dtype = np.int64)
            
    vertices = np.array(vertices, dtype = np.float64)
    triangles = np.array(triangles, dtype = np.int64)
    
    return vertices, triangles, compartments
    



#%%

# if __name__ == '__main__':
    
    
    