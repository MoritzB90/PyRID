# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product, combinations
import seaborn as sns
import h5py
import pandas as pd

col=sns.color_palette("colorblind", 10)
from matplotlib.font_manager import FontProperties
fontLgd = FontProperties()
fontLgd.set_size('x-small')

import plotly.graph_objects as go
import plotly.io as pio
#pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

import seaborn as sns
sns.set(style='ticks')

#%%

core_colors = np.array([[0.12156863, 0.46666667, 0.70588235, 1.        ],
   [1.        , 0.49803922, 0.05490196, 1.        ],
   [0.17254902, 0.62745098, 0.17254902, 1.        ],
   [0.83921569, 0.15294118, 0.15686275, 1.        ],
   [0.58039216, 0.40392157, 0.74117647, 1.        ],
   [0.54901961, 0.3372549 , 0.29411765, 1.        ],
   [0.89019608, 0.46666667, 0.76078431, 1.        ],
   [0.49803922, 0.49803922, 0.49803922, 1.        ],
   [0.7372549 , 0.74117647, 0.13333333, 1.        ],
   [0.09019608, 0.74509804, 0.81176471, 1.        ],
   [0.4       , 0.76078431, 0.64705882, 1.        ],
   [0.98823529, 0.55294118, 0.38431373, 1.        ],
   [0.55294118, 0.62745098, 0.79607843, 1.        ],
   [0.90588235, 0.54117647, 0.76470588, 1.        ],
   [0.65098039, 0.84705882, 0.32941176, 1.        ],
   [1.        , 0.85098039, 0.18431373, 1.        ],
   [0.89803922, 0.76862745, 0.58039216, 1.        ],
   [0.70196078, 0.70196078, 0.70196078, 1.        ],
   [0.86      , 0.3712    , 0.34      , 1.        ],
   [0.8288    , 0.86      , 0.34      , 1.        ],
   [0.34      , 0.86      , 0.3712    , 1.        ],
   [0.34      , 0.8288    , 0.86      , 1.        ],
   [0.3712    , 0.34      , 0.86      , 1.        ],
   [0.86      , 0.34      , 0.8288    , 1.        ]])
 

core_colors_max_bright = 1/np.max(core_colors[:,0:3], axis = 1)
core_colors_min_bright = np.ones(len(core_colors))*0.5

# tab10 + Set2 + hls reversed
patch_colors = np.array([[0.86      , 0.34      , 0.8288    , 1.        ],
   [0.3712    , 0.34      , 0.86      , 1.        ],
   [0.34      , 0.8288    , 0.86      , 1.        ],
   [0.34      , 0.86      , 0.3712    , 1.        ],
   [0.8288    , 0.86      , 0.34      , 1.        ],
   [0.86      , 0.3712    , 0.34      , 1.        ],
   [0.70196078, 0.70196078, 0.70196078, 1.        ],
   [0.89803922, 0.76862745, 0.58039216, 1.        ],
   [1.        , 0.85098039, 0.18431373, 1.        ],
   [0.65098039, 0.84705882, 0.32941176, 1.        ],
   [0.90588235, 0.54117647, 0.76470588, 1.        ],
   [0.55294118, 0.62745098, 0.79607843, 1.        ],
   [0.98823529, 0.55294118, 0.38431373, 1.        ],
   [0.4       , 0.76078431, 0.64705882, 1.        ],
   [0.09019608, 0.74509804, 0.81176471, 1.        ],
   [0.7372549 , 0.74117647, 0.13333333, 1.        ],
   [0.49803922, 0.49803922, 0.49803922, 1.        ],
   [0.89019608, 0.46666667, 0.76078431, 1.        ],
   [0.54901961, 0.3372549 , 0.29411765, 1.        ],
   [0.58039216, 0.40392157, 0.74117647, 1.        ],
   [0.83921569, 0.15294118, 0.15686275, 1.        ],
   [0.17254902, 0.62745098, 0.17254902, 1.        ],
   [1.        , 0.49803922, 0.05490196, 1.        ],
   [0.12156863, 0.46666667, 0.70588235, 1.        ]])

patch_colors_max_bright = 1/np.max(patch_colors[:,0:3], axis = 1)
patch_colors_min_bright = np.ones(len(patch_colors))*0.5

#%%

def plot_compartments(Simulation, save_fig = False, fig_name = None, fig_path = None, comp_name = None, face_groups = True, show_normals = True, mol_traj = None, indices = None, projection = 'orthographic', plot_cube = True, plane = '-xz', alpha = None, show = True):
    
    """Plots the compartment meshes and highlights any face groups and borders/edges using the plotly library. Also plots the triangle normal vectors.
    
    Parameters
    ----------
    Simulation : `object`
        Instance of the Simulation class
    save_fig : `boolean`
        If True, the plot is exported to a .png file.
    fig_name : `string`
        Name of the figure
    fig_path : `string`
        Path to which the figure is exported
    comp_name : `string`
        Name of the compartment which to plot
    face_groups : `boolean`
        If True, the face groups of the compartment are highlighted
    show_normals : `boolean`
        If True, the triangle normal vectors are plotted
    mol_traj : `float64[T,N,3]`
        Molecule trajectories can be passed which are then plotted as lines in 3D. T is the number of time steps, N the number of molecules. Molecules whose trajectory to plot can also be selected in addition by the parameter indces.
    indices : `int64[:]`
        Indices of the molecules whose trajectory to plot
    projection : `string`
        Projection type. Default = `orthographic`
    plot_cube = `boolean`
        If True, the simualtion box is visualized in addition to the selected compartment
    plane : `string` ('-xy', 'xy', '-yz', 'yz', '-xz', 'xz')
        Defines the plane to which the camera is oriented. Default = `-xz`
    alpha : `float64` in [0,1]
        Sets the opacity of the mesh faces.
    show : `boolean`
        If True, the plot is shown directly, otherwise only after the simualtion has ended.
    
    """
    
    if comp_name is None:
        for comp_name in Simulation.System.compartments_name:
            
            plot_compartment_0(Simulation, Simulation.System, Simulation.System.vertices, Simulation.System.Mesh['triangles'], Simulation.System.Compartments, comp_name, save_fig = save_fig, fig_name = fig_name, fig_path = fig_path, face_groups = face_groups, show_normals = show_normals, mol_traj = mol_traj, indices = indices, projection = projection, plot_cube = plot_cube, plane = plane, alpha = alpha, show = show)

    else:
        
        plot_compartment_0(Simulation, Simulation.System, Simulation.System.vertices, Simulation.System.Mesh['triangles'], Simulation.System.Compartments, comp_name, save_fig = save_fig, fig_name = fig_name, fig_path = fig_path, face_groups = face_groups, show_normals = show_normals, mol_traj = mol_traj, indices = indices, projection = projection, plot_cube = plot_cube, plane = plane, alpha = alpha, show = show)

#%%

def plot_compartment_0(Simulation, System, vertices, triangles, Compartments, comp_name, save_fig = False, fig_name = None, fig_path = None, face_groups = True, show_normals = True, mol_traj = None, indices = None, projection = 'orthographic', plot_cube = True, plane = '-xz', alpha = None, show = True):
    
    """A brief description of what the function (method in case of classes) is and what it’s used for
    
    Parameters
    ----------
    parameter_1 : dtype
        Some Information
    parameter_2 : dtype
        Some Information
    
    Raises
    ------
    NotImplementedError (just an example)
        Brief explanation of why/when this exception is raised
    
    Returns
    -------
    dtype
        Some information
    
    """

    """
    https://chart-studio.plotly.com/~empet/14749/mesh3d-with-intensities-and-flatshading/#/
    """
    
    scale = np.min(System.box_lengths)
    
    if comp_name!='Box':
        comp_id = System.compartments_id[str(comp_name)]
        
        triangle_ids = Compartments[comp_id].triangle_ids
    else:
        triangle_ids = System.triangle_ids
        
    #----------------
    
    mygrey=[0, 'rgb(153, 153, 153)'], [1., 'rgb(255,255,255)']
    myred=[0, 'rgb(153, 0, 0)'], [1., 'rgb(255,0,0)']
    mygreen=[0, 'rgb(0, 153, 0)'], [1., 'rgb(0,255,0)']
    myblue=[0, 'rgb(0, 0, 153)'], [1., 'rgb(0,0,255)']
    mymagenta=[0, 'rgb(153, 0, 153)'], [1., 'rgb(255,0,255)']
    myyellow=[0, 'rgb(153, 153, 0)'], [1., 'rgb(255,255,0)']
    
    cone_color = [0, 'rgb(0, 70, 0)'], [1, 'rgb(0, 70, 0)']
    
    #----------------
    
    face_colors = np.array(['rgb(153, 153, 153)']*(max(triangle_ids)+1))
    
    #----------------
    
    if comp_name!='Box':
        
        x_cube=np.array([0, 0, 1, 1, 0, 0, 1, 1])*System.box_lengths[0]-System.box_lengths[0]/2
        y_cube=np.array([0, 1, 1, 0, 0, 1, 1, 0])*System.box_lengths[1]-System.box_lengths[1]/2
        z_cube=np.array([0, 0, 0, 0, 1, 1, 1, 1])*System.box_lengths[2]-System.box_lengths[2]/2
        
        i_cube = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
        j_cube = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
        k_cube = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
        
        Cube = go.Mesh3d(
            # 8 vertices of a cube
            x=x_cube,
            y=y_cube,
            z=z_cube,
            colorscale=mygrey, 
            intensity = z_cube,
            flatshading=True,
            i = i_cube,
            j = j_cube,
            k = k_cube,
            name='Simulation box',
            showscale=False
        )
        
        Cube.update(cmin=-7,# atrick to get a nice plot (z.min()=-3.31909)
                       lighting=dict(ambient=0.18,
                                     diffuse=1,
                                     fresnel=0.1,
                                     specular=1,
                                     roughness=0.05,
                                     facenormalsepsilon=1e-15,
                                     vertexnormalsepsilon=1e-15),
                       lightposition=dict(x=100,
                                          y=200,
                                          z=0
                                         ),
                       opacity = 0.1
                              );

    #----------------
    
    data = []
        
    # ----------------------------------
    
    if comp_name!='Box':
        
        if face_groups:
        
            if len(Compartments[comp_id].border_2d)>0:
            
                Xe = []
                Ye = []
                Ze = []
            
                Xe_dir = []
                Ye_dir = []
                Ze_dir = []
                
                Xe_cone = []
                Ye_cone = []
                Ze_cone = []
                
                for i in range(len(Compartments[comp_id].border_2d)):
                    
                    border_tri = Compartments[comp_id].border_2d[i]['triangle_ids']
                    
                    edge_id = Compartments[comp_id].border_2d[i]['edge_id']
                    
                    direction = Compartments[comp_id].border_2d[i]['direction_normal']*scale/10
                    
                    e0 = System.Mesh[border_tri]['edges'][edge_id][0]
                    e1 = System.Mesh[border_tri]['edges'][edge_id][1]
                    
                    p0 = vertices[e0]
                    p1 = vertices[e1]
                    
                    Xe.extend([p0[0], p1[0]]+[ None])
                    Ye.extend([p0[1], p1[1]]+[ None])
                    Ze.extend([p0[2], p1[2]]+[ None])
                    
                    Xe_dir.extend([p0[0]+(p1[0]-p0[0])/2, p0[0]+(p1[0]-p0[0])/2+direction[0]]+[ None])
                    Ye_dir.extend([p0[1]+(p1[1]-p0[1])/2, p0[1]+(p1[1]-p0[1])/2+direction[1]]+[ None])
                    Ze_dir.extend([p0[2]+(p1[2]-p0[2])/2, p0[2]+(p1[2]-p0[2])/2+direction[2]]+[ None])
                    
                    Xe_cone.append([p0[0]+(p1[0]-p0[0])/2, p0[0]+(p1[0]-p0[0])/2+direction[0]])
                    Ye_cone.append([p0[1]+(p1[1]-p0[1])/2, p0[1]+(p1[1]-p0[1])/2+direction[1]])
                    Ze_cone.append([p0[2]+(p1[2]-p0[2])/2, p0[2]+(p1[2]-p0[2])/2+direction[2]])
                        
                Xe_cone = np.array(Xe_cone)
                Ye_cone = np.array(Ye_cone)
                Ze_cone = np.array(Ze_cone)
                
                u = np.diff(Xe_cone).squeeze()
                v = np.diff(Ye_cone).squeeze()
                w = np.diff(Ze_cone).squeeze()
                
                cone_norm = np.sqrt(u**2+v**2+w**2)
                
                u/=cone_norm
                v/=cone_norm
                w/=cone_norm
                
                edges = go.Scatter3d(
                                   x=Xe,
                                   y=Ye,
                                   z=Ze,
                                   mode='lines',
                                   name='',
                                   line=dict(color= 'rgb(153, 0, 153)', width=10))  
                    
                directions = go.Scatter3d(
                                   x=Xe_dir,
                                   y=Ye_dir,
                                   z=Ze_dir,
                                   mode='lines',
                                   name='',
                                   line=dict(color= 'rgb(0,70,0)', width=5))  
                
                cones = go.Cone(
                                x=Xe_cone[:,1],
                                y=Ye_cone[:,1],
                                z=Ze_cone[:,1],
                                u=u,
                                v=v,
                                w=w,
                                sizemode="absolute",
                                sizeref=scale/300,
                                anchor="tail",
                                colorscale=cone_color,
                                showscale=False)
                
                
                
                data.extend([edges, directions, cones])
            
            
            #-----------------------
            
            
            if len(Compartments[comp_id].border_2d)>0:
                triangle_ids_group = Compartments[comp_id].border_2d['triangle_ids']
                
                
                face_colors[triangle_ids_group] = 'rgb(153, 0, 0)'
                
                
            # if len(Compartments[comp_id].border_3d)>0:
            #     triangle_ids_group = Compartments[comp_id].border_3d['triangle_ids']
                
            #     face_colors[triangle_ids_group] = 'rgb(153, 153, 0)'
                
            if len(Compartments[comp_id].groups)>0:
                for group in Compartments[comp_id].groups:
                    triangle_ids_group = Compartments[comp_id].groups[group]['triangle_ids']
                    if group == 'transparent':
                        face_colors[triangle_ids_group] = 'rgb(153, 153, 0)'
                    else:
                        face_colors[triangle_ids_group] = 'rgb(0, 0, 153)'
    
    else:
        
        if len(System.box_border)>0:
            triangle_ids_group = System.box_border['triangle_ids']
            
            x = np.array(vertices).T[0]
            y = np.array(vertices).T[1]
            z = np.array(vertices).T[2]
            
            I = np.array(triangles)[triangle_ids_group].T[0]
            J = np.array(triangles)[triangle_ids_group].T[1]
            K = np.array(triangles)[triangle_ids_group].T[2]
            
            # lighting_effects = dict(ambient=0.4, diffuse=0.5, roughness = 0.9, specular=0.6, fresnel=0.2)
            
                                       
            group_mesh = go.Mesh3d(x=x,
                                y=y,
                                z=z,
                                colorscale=myyellow, 
                                intensity= z,
                                flatshading=True,
                                i=I,
                                j=J,
                                k=K,
                                name=comp_name,
                                showscale=False
                                )
            
            group_mesh.update(cmin=-7,# atrick to get a nice plot (z.min()=-3.31909)
                            lighting=dict(ambient=0.18,
                                          diffuse=1,
                                          fresnel=0.1,
                                          specular=1,
                                          roughness=0.05,
                                          facenormalsepsilon=1e-15,
                                          vertexnormalsepsilon=1e-15),
                            lightposition=dict(x=100,
                                              y=200,
                                              z=0
                                              ),
                            opacity = 0.5
                                  );
        
            data.append(group_mesh)
            
            
    #-----------------------
    
    x = np.array(vertices).T[0]
    y = np.array(vertices).T[1]
    z = np.array(vertices).T[2]
    
    I = np.array(triangles)[triangle_ids].T[0]
    J = np.array(triangles)[triangle_ids].T[1]
    K = np.array(triangles)[triangle_ids].T[2]
    
    # lighting_effects = dict(ambient=0.4, diffuse=0.5, roughness = 0.9, specular=0.6, fresnel=0.2)
    
    mesh = go.Mesh3d(x=x,
                        y=y,
                        z=z,
                        # colorscale=mygrey, 
                        # intensity= z,
                        facecolor = face_colors[triangle_ids],
                        flatshading=True,
                        i=I,
                        j=J,
                        k=K,
                        name=comp_name,
                        showscale=False
                        )
    
    if projection == 'perspective':
        ambient = 0.18
        if alpha is None:
            opacity = 1.0 # 0.5
        else:
            opacity = alpha
        fresnel=0.1
        x_light = 100
        y_light = 200
        z_light = 0
    else:
        ambient = 0.28
        if alpha is None:
            opacity = 1.0 # 0.8
        else:
            opacity = alpha
        fresnel=0.5
        x_light = 20000
        y_light = 100000
        z_light = -100
        
    mesh.update(cmin=-7,# atrick to get a nice plot (z.min()=-3.31909)
                   lighting=dict(ambient=ambient,
                                 diffuse=1,
                                 fresnel=fresnel,
                                 specular=1,
                                 roughness=0.05,
                                 facenormalsepsilon=1e-15,
                                 vertexnormalsepsilon=1e-15),
                   lightposition=dict(x=x_light,
                                      y=y_light,
                                      z=z_light
                                     ),
                   opacity = opacity
                          );
    
    
    
    tri_points = np.array(vertices)[np.array(triangles)[triangle_ids]]
    
    #extract the lists of x, y, z coordinates of the triangle vertices and connect them by a line
    Xe = []
    Ye = []
    Ze = []
    for T in tri_points:
        Xe.extend([T[k%3][0] for k in range(4)]+[ None])
        Ye.extend([T[k%3][1] for k in range(4)]+[ None])
        Ze.extend([T[k%3][2] for k in range(4)]+[ None])
    
    lines = go.Scatter3d(
                       x=Xe,
                       y=Ye,
                       z=Ze,
                       mode='lines',
                       name='',
                       line=dict(color= 'rgb(70,70,70)', width=1))  
    
    
    # ----------------------------------
    if comp_name!='Box' and plot_cube:
        data.extend([mesh, lines, Cube])
    else:
        data.extend([mesh, lines])
        
    #-----------------------
    
    if show_normals:
        
        Xe_norm = []
        Ye_norm = []
        Ze_norm = []
    
        
        for tri_id in triangle_ids:
            
            centroid = System.Mesh[tri_id]['triangle_centroid']
            normal = System.Mesh[tri_id]['triangle_coord'][3]*scale/20
            
            
            Xe_norm.extend([centroid[0], centroid[0]+normal[0]]+[ None])
            Ye_norm.extend([centroid[1], centroid[1]+normal[1]]+[ None])
            Ze_norm.extend([centroid[2], centroid[2]+normal[2]]+[ None])
            
        normals = go.Scatter3d(
                           x=Xe_norm,
                           y=Ye_norm,
                           z=Ze_norm,
                           mode='lines',
                           name='',
                           line=dict(color= 'rgb(0,0,0)', width=2))  
    
        data += [normals]
        
    #-----------------------
    
    if mol_traj is not None:
        
        if indices is None:
            indices = range(len(mol_traj[0]))
            
        for i in indices:
                
            x = mol_traj[:,i,0]
            y = mol_traj[:,i,1]
            z = mol_traj[:,i,2]
            
            mol_trace = go.Scatter3d(
                mode='lines',
                x=x, y=y, z=z,
                line=dict(
                    color='rgb(255,0,0)',
                    width=8
                )
            )
        
            data += [mol_trace]
    #-----------------------
    
    
    layout = go.Layout(
             #title=comp_name,
             font=dict(size=16, color='black'),
             paper_bgcolor='rgb(255,255,255)',
             #titlefont=dict(size=24, color='black'), 
             #width = 2000, 
             #height = 2000,
             #margin=dict(l=400, r=400, t=100, b=100),
             scene = dict(camera = dict(projection = dict(type = projection))),
            )
        
    fig = go.Figure(data=data, layout=layout)
    
    x_visibility = True
    y_visibility = True
    z_visibility = True
    if 'xz' in plane:
        camera = dict(
            eye=dict(x=0., y=int(plane[0]+'1')*1.0, z=0.)
        )
        y_visibility = False
    elif 'xy' in plane:
        camera = dict(
            eye=dict(x=0., y=0., z=int(plane[0]+'1')*1.0)
        )
        z_visibility = False
    elif 'yz' in plane:
        camera = dict(
            eye=dict(x=int(plane[0]+'1')*1.0, y=0., z=0.)
        )
        x_visibility = False
    
    fig.update_layout(scene = dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    xaxis = dict(showgrid =  True,
                                 visible = x_visibility,
                                 showline = True,
                                 backgroundcolor="white",
                                 gridcolor="black",
                                 showbackground=False,
                                 zerolinecolor="black"),
                    yaxis = dict(showgrid =  True,
                                 visible = y_visibility,
                                 showline = True,
                                 backgroundcolor="white",
                                 gridcolor="black",
                                 showbackground=False,
                                 zerolinecolor="black"),
                    zaxis = dict(showgrid =  True,
                                 visible = z_visibility,
                                 showline = True,
                                 backgroundcolor="white",
                                 gridcolor="black",
                                 showbackground=False,
                                 zerolinecolor="black"),
                    ),
                    #width = 2000, 
                    #height = 2000,
                    autosize=True,
                    margin=dict(l=0,
                                r=0,
                                b=0,
                                t=50,
                                pad=0),
                    scene_camera=camera,
                    showlegend=False,
                    )
    
    
    # if projection == 'orthographic':
    #     fig.update_yaxes(visible=False)
    
    
    if save_fig == True:
        if fig_name is None:
            fig_name = Simulation.file_name + ''
        if fig_path is None:
            fig_path = Simulation.fig_path
        
        fig.write_image(fig_path / (fig_name+'_Compartment_'+comp_name+'.png'), width=2000, height=2000, scale=1) #, engine='kaleido') 
        
    if show:
        fig.show()  


#%%

def plot_scene(Simulation, save_fig = False, fig_name = None, fig_path = None, projection = 'orthographic', plane = '-xz', show = True):
    
    """Plots the simulation scence with all its compartments and molecule positions. Molecule positions are represented by a scatter plot.
    
    Parameters
    ----------
    Simulation : `object`
        Instance of the Simulation class
    save_fig : `boolean`
        If True, the plot is exported to a .png file.
    fig_name : `string`
        Name of the figure
    fig_path : `string`
        Path to which the figure is exported
    projection : `string`
        Projection type. Default = `orthographic`
    plane : `string` ('-xy', 'xy', '-yz', 'yz', '-xz', 'xz')
        Defines the plane to which the camera is oriented. Default = `-xz`
    show : `boolean`
        If True, the plot is shown directly, otherwise only after the simualtion has ended.
    
    """
    
    # https://chart-studio.plotly.com/~empet/14749/mesh3d-with-intensities-and-flatshading/#/
    
    if fig_name is None:
        fig_name = Simulation.file_name + ''
            
    if Simulation.RBs.occupied.n>0:
    
        if len(Simulation.System.Compartments)>0:
            vertices = Simulation.System.vertices
            triangle_ids = Simulation.System.triangle_ids
            triangles = Simulation.System.Mesh[triangle_ids]['triangles']
            # molecules = Simulation.RBs[0:]['pos']
            # type_ids = Simulation.RBs[0:]['type_id']
        
        molecules = []
        type_ids = []
        for i0 in range(Simulation.RBs.occupied.n):
            i = Simulation.RBs.occupied[i0] 
            molecules.append(Simulation.RBs[i]['pos'])
            type_ids.append(Simulation.RBs[i]['type_id'])
        molecules = np.array(molecules)
        type_ids = np.array(type_ids)
        
        Diameter_Types = []
        for mol in Simulation.System.molecule_id_to_name:
            Diameter_Types.append(2*Simulation.System.molecule_types[str(mol)].radius)
        Diameter_Types = np.array(Diameter_Types)
        Diameter = Diameter_Types[type_ids]
        
        # Radii*=10/np.min(Radii_Types)+1
        
        #----------------
        
        mygrey=[0, 'rgb(153, 153, 153)'], [1., 'rgb(255,255,255)']
        
        x_cube=np.array([0, 0, 1, 1, 0, 0, 1, 1])*Simulation.System.box_lengths[0]-Simulation.System.box_lengths[0]/2
        y_cube=np.array([0, 1, 1, 0, 0, 1, 1, 0])*Simulation.System.box_lengths[1]-Simulation.System.box_lengths[1]/2
        z_cube=np.array([0, 0, 0, 0, 1, 1, 1, 1])*Simulation.System.box_lengths[2]-Simulation.System.box_lengths[2]/2
        
        i_cube = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
        j_cube = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
        k_cube = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
        
        Cube = go.Mesh3d(
            # 8 vertices of a cube
            x=x_cube,
            y=y_cube,
            z=z_cube,
            colorscale=mygrey, 
            intensity = z_cube,
            flatshading=True,
            i = i_cube,
            j = j_cube,
            k = k_cube,
            name='Simulation box',
            showscale=False
        )
        
        Cube.update(cmin=-7,# atrick to get a nice plot (z.min()=-3.31909)
                       lighting=dict(ambient=0.18,
                                     diffuse=1,
                                     fresnel=0.1,
                                     specular=1,
                                     roughness=0.05,
                                     facenormalsepsilon=1e-15,
                                     vertexnormalsepsilon=1e-15),
                       lightposition=dict(x=100,
                                          y=200,
                                          z=0
                                         ),
                       opacity = 0.1
                              );
        
        #----------------
        
        if len(Simulation.System.Compartments)>0:
            
            x = vertices.T[0]
            y = vertices.T[1]
            z = vertices.T[2]
            
            I = triangles.T[0]
            J = triangles.T[1]
            K = triangles.T[2]
            
            # lighting_effects = dict(ambient=0.4, diffuse=0.5, roughness = 0.9, specular=0.6, fresnel=0.2)
            
                                       
            mesh = go.Mesh3d(x=x,
                                y=y,
                                z=z,
                                colorscale=mygrey, 
                                intensity= z,
                                flatshading=True,
                                i=I,
                                j=J,
                                k=K,
                                name=fig_name,
                                showscale=False
                                )
            
            mesh.update(cmin=-7,# atrick to get a nice plot (z.min()=-3.31909)
                           lighting=dict(ambient=0.18,
                                         diffuse=1,
                                         fresnel=0.1,
                                         specular=1,
                                         roughness=0.05,
                                         facenormalsepsilon=1e-15,
                                         vertexnormalsepsilon=1e-15),
                           lightposition=dict(x=100,
                                              y=200,
                                              z=0
                                             ),
                           opacity = 0.5
                                  );
            
            
            
            tri_points = vertices[triangles]
            
            #extract the lists of x, y, z coordinates of the triangle vertices and connect them by a line
            Xe = []
            Ye = []
            Ze = []
            for T in tri_points:
                Xe.extend([T[k%3][0] for k in range(4)]+[ None])
                Ye.extend([T[k%3][1] for k in range(4)]+[ None])
                Ze.extend([T[k%3][2] for k in range(4)]+[ None])
            
            lines = go.Scatter3d(
                               x=Xe,
                               y=Ye,
                               z=Ze,
                               mode='lines',
                               name='',
                               line=dict(color= 'rgb(70,70,70)', width=1))  
            
            
            
            data=[Cube, mesh, lines]
            
        else:
            
            data=[Cube]
        
        
        layout = go.Layout(
                 title=Simulation.file_name,
                 font=dict(size=16, color='black'),
                 # width=700,
                 # height=700,
                 # scene_xaxis_visible=False,
                 # scene_yaxis_visible=False,
                 # scene_zaxis_visible=False,
                 paper_bgcolor='rgb(255,255,255)',
                 scene = dict(camera = dict(projection = dict(type = projection))),
               
                )
        

            
        for rb_id in set(type_ids):
            
            Mask = type_ids == rb_id
            
            xP = molecules[Mask,0]
            yP = molecules[Mask,1]
            zP = molecules[Mask,2]
            
            points = go.Scatter3d(x=xP, y=yP, z=zP, mode='markers', name = Simulation.System.molecule_id_to_name[rb_id],  
                 marker=dict(
                 size=Diameter[Mask]/(np.max(Diameter)/15),
                 # sizemode='diameter',
                 # sizeref=np.max(Diameter)/30,
                 sizemin=20,
                 color=rb_id,                # set color to an array/list of desired values
                 colorscale='Rainbow',   # choose a colorscale
                 opacity=1.0
             ))
        
            data.append(points)
            
            
        fig = go.Figure(data=data, layout=layout)
        
        
        
        x_visibility = True
        y_visibility = True
        z_visibility = True
        if 'xz' in plane:
            camera = dict(
                eye=dict(x=0., y=int(plane[0]+'1')*1.0, z=0.)
            )
            y_visibility = False
        elif 'xy' in plane:
            camera = dict(
                eye=dict(x=0., y=0., z=int(plane[0]+'1')*1.0)
            )
            z_visibility = False
        elif 'yz' in plane:
            camera = dict(
                eye=dict(x=int(plane[0]+'1')*1.0, y=0., z=0.)
            )
            x_visibility = False
        
        fig.update_layout(scene = dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z',
                        xaxis = dict(showgrid =  True,
                                     visible = x_visibility,
                                     showline = True,
                                     backgroundcolor="white",
                                     gridcolor="black",
                                     showbackground=False,
                                     zerolinecolor="black"),
                        yaxis = dict(showgrid =  True,
                                     visible = y_visibility,
                                     showline = True,
                                     backgroundcolor="white",
                                     gridcolor="black",
                                     showbackground=False,
                                     zerolinecolor="black"),
                        zaxis = dict(showgrid =  True,
                                     visible = z_visibility,
                                     showline = True,
                                     backgroundcolor="white",
                                     gridcolor="black",
                                     showbackground=False,
                                     zerolinecolor="black"),
                        ),
                        #width = 2000, 
                        #height = 2000,
                        autosize=True,
                        margin=dict(l=0,
                                    r=0,
                                    b=0,
                                    t=50,
                                    pad=0),
                        scene_camera=camera,
                        showlegend=True,
                        )
        
        
        # fig.update_layout(
        #     mapbox_layers=[
        #         {
        #             # "below": "traces",
        #             # "circle": {"radius": 10},
        #             # "color":"red",
        #             "minzoom": 4,
        #             # "source": gpd.GeoSeries(
        #                 # df.loc[:, ["Longitude", "Latitude"]].apply(
        #                     # shapely.geometry.Point, axis=1
        #                 # )
        #             # ).__geo_interface__,
        #         },
        #     ],
        #     mapbox_style="carto-positron",
        # )
        
        if show:
            fig.show()
        
        if save_fig == True:
            if fig_name is None:
                fig_name = Simulation.file_name + ''
            if fig_path is None:
                fig_path = Simulation.fig_path
            
            fig.write_image(fig_path / (fig_name+'_MolDistr_.png'), width=2000, height=2000, scale=1) #, engine='kaleido') 
        
    else:
    
        print('You have not placed any molecules in the scene yet. Therefore, PyRID will not create a figure showing molecule positions.')

#%%

def cuboid_data(origin, size=[1,1,1]):
    
    """Returns the vertex array of a cuboid given an origin vector.
    
    Parameters
    ----------
    origin : `float64[3]`
        Origin of the cuboid
    size : `float64[3]`
        Extend of the cuboid in each dimension. Default = [1,1,1]
    
    """
    
    # https://stackoverflow.com/questions/49277753/python-matplotlib-plotting-cuboids
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(origin)
    return X

def draw_cuboid(positions ,sizes=None, colors=None, edgecolor='k', edgewidth=1, edgealpha=1):
    
    """Draws a cuboid of a given size at a given position.
    
    Parameters
    ----------
    positions : `float[:]`
        Position vector
    size : `float[3]`
        Extend of the cuboid in each dimension. Default = [1,1,1]
    colors : `array_like`
        Array of face colors
    edgecolor : `string`
        Edge color. Default = `k`
    edgewidth : `float`
        Width of the cuboid edges. Default = 1
    edgealpha : `float`
        Opacity of the cuboid edges. Default = 1
        
    Returns
    -------
    `tuple(object, object)`
        Poly3DCollection, Line3DCollection
    
    """
    
    # https://stackoverflow.com/questions/49277753/python-matplotlib-plotting-cuboids
    if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( cuboid_data(p, size=s) )
    return Poly3DCollection(np.concatenate(g),  
                            facecolors=np.repeat(colors,6), alpha=0.05), Line3DCollection(np.concatenate(g), colors=edgecolor, linewidths=edgewidth, linestyles='-', alpha = edgealpha)


def draw_sphere(ax, pos, radius, col):
    
    """Draws a sphere of given radius and centered at a given position vector.
    
    Parameters
    ----------
    ax : `object`
        Axis object of a matplotlib figure
    pos : `float[3]`
        Position of the sphere center
    radius : `float`
        Sphere radius
    col : `string`
        Sphere color
    
    
    Returns
    -------
    `float[2,3]`
        Lower left and upper right sphere vertices (Used to set figure aspect ratio).
    
    """
    
    # draw sphere
    res = 16j
    u, v = np.mgrid[0:2*np.pi:2*res, 0:np.pi:res]
    x = pos[0]+radius*np.cos(u)*np.sin(v)
    y = pos[1]+radius*np.sin(u)*np.sin(v)
    z = pos[2]+radius*np.cos(v)
    # ax.plot_wireframe(x, y, z, color=col)
    ax.plot_surface(x, y, z, color=col, alpha=0.8, shade = True, antialiased=True, linewidth = 0)
    
    return np.array([np.max(x), np.max(y), np.max(z)]), np.array([np.min(x), np.min(y), np.min(z)])

#%%
    
def plot_triangle(p0,p1,p2, ax, show=True):
    
    """Plots a triangle in 3D.
    
    Parameters
    ----------
    p0 : `float64[3]`
        Vertex 1
    p1 : `float64[3]`
        Vertex 2
    p2 : `float64[3]`
        Vertex 3
    ax : `object`
        Axis object of a matplotlib figure
    
    
    """
    
    # fig = plt.figure()
    # ax = plt.axes(projection='3d', proj_type = 'ortho')
    ax.set_box_aspect((1,1,1))
    
    
    verts = [p0,p1,p2]
    mesh = Poly3DCollection([verts], linewidths=1, alpha=0.5)
    edge_color = (50 / 255, 50 / 255, 50 / 255)
    # mesh.set_facecolor(face_color)
    mesh.set_edgecolor(edge_color)
    ax.add_collection3d(mesh)    
    
    ax.view_init(30, 185)
    
    ax.set_xlim(np.min(np.array([p0,p1,p2])[:,0]), np.max(np.array([p0,p1,p2])[:,0]))
    ax.set_ylim(np.min(np.array([p0,p1,p2])[:,1]), np.max(np.array([p0,p1,p2])[:,1]))
    ax.set_zlim(np.min(np.array([p0,p1,p2])[:,2]), np.max(np.array([p0,p1,p2])[:,2]))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if show:
        plt.show()
    

def plot_triangles(Simulation, triangle_ids, points = None):
    
    vertices = Simulation.System.vertices
    triangle_ids
    triangles = Simulation.System.Mesh[triangle_ids]['triangles']

    #----------------
    
    mygrey=[0, 'rgb(153, 153, 153)'], [1., 'rgb(255,255,255)']
    
    #----------------
    
    x = vertices.T[0]
    y = vertices.T[1]
    z = vertices.T[2]
    
    I = triangles.T[0]
    J = triangles.T[1]
    K = triangles.T[2]
    
                               
    mesh = go.Mesh3d(x=x,
                        y=y,
                        z=z,
                        colorscale=mygrey, 
                        intensity= z,
                        flatshading=True,
                        i=I,
                        j=J,
                        k=K,
                        name='triangles',
                        showscale=False
                        )
    
    mesh.update(cmin=-7,# atrick to get a nice plot (z.min()=-3.31909)
                   lighting=dict(ambient=0.18,
                                 diffuse=1,
                                 fresnel=0.1,
                                 specular=1,
                                 roughness=0.05,
                                 facenormalsepsilon=1e-15,
                                 vertexnormalsepsilon=1e-15),
                   lightposition=dict(x=100,
                                      y=200,
                                      z=0
                                     ),
                   opacity = 0.5
                          );
    
    
    
    tri_points = vertices[triangles]
    
    #extract the lists of x, y, z coordinates of the triangle vertices and connect them by a line
    Xe = []
    Ye = []
    Ze = []
    for T in tri_points:
        Xe.extend([T[k%3][0] for k in range(4)]+[ None])
        Ye.extend([T[k%3][1] for k in range(4)]+[ None])
        Ze.extend([T[k%3][2] for k in range(4)]+[ None])
    
    lines = go.Scatter3d(
                       x=Xe,
                       y=Ye,
                       z=Ze,
                       mode='lines',
                       name='',
                       line=dict(color= 'rgb(70,70,70)', width=1))  
    
    
    
    data=[mesh, lines]
    
    
    
    layout = go.Layout(
             title='triangles',
             font=dict(size=16, color='black'),
             paper_bgcolor='rgb(255,255,255)',
            )
    

    if points is not None:
        
        xP = points[:,0]
        yP = points[:,1]
        zP = points[:,2]
        
        points = go.Scatter3d(x=xP, y=yP, z=zP, mode='markers', name = 'points', opacity=1.0)
    
        data.append(points)
        
        
    fig = go.Figure(data=data, layout=layout)

    
    x_min = np.min(vertices[triangles].T[0])
    x_max = np.max(vertices[triangles].T[0])
    y_min = np.min(vertices[triangles].T[1])
    y_max = np.max(vertices[triangles].T[1])
    z_min = np.min(vertices[triangles].T[2])
    z_max = np.max(vertices[triangles].T[2])
    
    fig.update_layout(scene = dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    xaxis = dict(nticks=4, range=[x_min, x_max],),
                    yaxis = dict(nticks=4, range=[y_min, y_max],),
                    zaxis = dict(nticks=4, range=[z_min, z_max],),
                    ),
                    margin=dict(l=0,
                                r=0,
                                b=0,
                                t=50,
                                pad=0),
                    showlegend=True,
                    )
    
    fig.show()
    
#%%

def plot_path(file_path, molecule, indices, show=True):
    
    """A brief description of what the function (method in case of classes) is and what it’s used for
    
    Parameters
    ----------
    parameter_1 : dtype
        Some Information
    parameter_2 : dtype
        Some Information
    
    Raises
    ------
    NotImplementedError (just an example)
        Brief explanation of why/when this exception is raised
    
    Returns
    -------
    dtype
        Some information
    
    """
    
    #file_path+'hdf5/'+file_name+'.h5'
    hdf = h5py.File(file_path, 'r', track_order=True)
    
    measure = 'Position'
    
    Pos = []
    
    for key in hdf[measure][molecule].keys():
        Pos.append(hdf[measure][molecule][key])
        
    
    Pos = np.array(Pos)
    
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')# fig.gca(projection='3d')
    for i in indices:
        ax.plot(Pos[:,i,0], Pos[:,i,1], Pos[:,i,2])
    # ax.legend()
    
    if show:
        plt.show()
    
    hdf.close()

#%%

def plot_path_mesh(file_path, molecule, indices, show=True):
    
    """A brief description of what the function (method in case of classes) is and what it’s used for
    
    Parameters
    ----------
    parameter_1 : dtype
        Some Information
    parameter_2 : dtype
        Some Information
    
    Raises
    ------
    NotImplementedError (just an example)
        Brief explanation of why/when this exception is raised
    
    Returns
    -------
    dtype
        Some information
    
    """
    
    #file_path+'hdf5/'+file_name+'.h5'
    hdf = h5py.File(file_path, 'r', track_order=True)
    
    measure = 'Position'
    
    Pos = []
    
    for key in hdf[measure][molecule].keys():
        Pos.append(hdf[measure][molecule][key])
        
    
    Pos = np.array(Pos)
    
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')# fig.gca(projection='3d')
    for i in indices:
        ax.plot(Pos[:,i,0], Pos[:,i,1], Pos[:,i,2])
    # ax.legend()
    
    if show:
        plt.show()
    
    hdf.close()

#%%

def plot_cell_grid(Simulation, save_fig = False, fig_name = None, fig_path = None, Compartments = None, show=True):
    
    """A brief description of what the function (method in case of classes) is and what it’s used for
    
    Parameters
    ----------
    parameter_1 : dtype
        Some Information
    parameter_2 : dtype
        Some Information
    
    Raises
    ------
    NotImplementedError (just an example)
        Brief explanation of why/when this exception is raised
    
    Returns
    -------
    dtype
        Some information
    
    """
    
    # col=sns.color_palette("colorblind", 10)
    col=sns.color_palette("rocket", Simulation.System.cells_per_dim[0])
    
    plt.figure(figsize=(4,4), dpi=150)
    ax = plt.axes(projection='3d', proj_type = 'persp')
    # ax = plt.axes(projection='3d', proj_type = 'ortho')
    ax.set_box_aspect((1,1,1))
    
    if Simulation.System.mesh == True and Compartments is not None:
        for name in Compartments:
            for j,X in enumerate(Compartments[name].triangles):
        
                verts = [Simulation.System.vertices[X[0]], Simulation.System.vertices[X[1]], Simulation.System.vertices[X[2]]]
                
                inside = True
                for dim in range(3):
                    for i in range(3):
                        if verts[i][dim]<-Simulation.System.box_lengths[dim]/2:
                            inside=False
                        if verts[i][dim]>Simulation.System.box_lengths[dim]/2:
                            inside=False
                 
                if inside == True:
                    mesh = Poly3DCollection([verts], linewidths=0.25, alpha=0.5)
                    face_color = (141 / 255, 184 / 255, 226 / 255)
                    edge_color = (50 / 255, 50 / 255, 50 / 255)
                    mesh.set_facecolor(face_color)
                    mesh.set_edgecolor(edge_color)
                    ax.add_collection3d(mesh)
                
                
    pc, lc = draw_cuboid(positions=[-Simulation.System.box_lengths/2], sizes=[Simulation.System.box_lengths])
    ax.add_collection3d(pc)    
    # ax.add_collection3d(lc) 
    
    # for cell in range(Simulation.System.Ncell):

    for cx in range(Simulation.System.cells_per_dim[0]):
        for cy in range(Simulation.System.cells_per_dim[1]):
            for cz in range(Simulation.System.cells_per_dim[2]):
        
                cell = cx + cy * Simulation.System.cells_per_dim[0] + cz * Simulation.System.cells_per_dim[0] * Simulation.System.cells_per_dim[1]
                
                # print(cell)
                
                origin = Simulation.System.AABB_Centers[cell]-Simulation.System.cell_length_per_dim/2
                extent = Simulation.System.cell_length_per_dim
                
                pc, lc = draw_cuboid(positions=[origin], sizes=[extent], edgecolor=col[cx], edgewidth=0.2, edgealpha=0.5)
                # ax.add_collection3d(pc)    
                ax.add_collection3d(lc) 

                
    #------------------------------------------------------------------
    
           
    plt.title('Number of cells = {}'.format(Simulation.System.Ncell), y=1, pad=2, fontsize=10)
    
    ax.set_xlim(-Simulation.System.box_lengths[0]/2,Simulation.System.box_lengths[0]/2)
    ax.set_ylim(-Simulation.System.box_lengths[1]/2,Simulation.System.box_lengths[1]/2)
    ax.set_zlim(-Simulation.System.box_lengths[2]/2,Simulation.System.box_lengths[2]/2)
    ax.view_init(30, 185)
    # ax.view_init(90, -90)
    
    ax.tick_params(axis='x', pad=6)
    ax.tick_params(axis='y', pad=-7)
    ax.tick_params(axis='z', pad=5)
    ax.set_xlabel(r'x in $\mu m$',labelpad=14)
    ax.set_ylabel(r'y in $\mu m$',labelpad=0)
    plt.yticks(rotation=45)
    ax.set_zlabel(r'z in $\mu m$',labelpad=7)
    
    if show:
        plt.show()
    
    if save_fig == True:
        if fig_name is None:
            fig_name = Simulation.file_name + ''
        if fig_path is None:
            fig_path = Simulation.fig_path
        plt.savefig(fig_path / (fig_name+'_CellGrid.png'), bbox_inches="tight", dpi = 300)
    
#%%

def plot_sphere_packing(Compartment_Number, Simulation, points, ptype, save_fig = False, fig_name = None , fig_path = None, show=True):
    
    """A brief description of what the function (method in case of classes) is and what it’s used for
    
    Parameters
    ----------
    parameter_1 : dtype
        Some Information
    parameter_2 : dtype
        Some Information
    
    Raises
    ------
    NotImplementedError (just an example)
        Brief explanation of why/when this exception is raised
    
    Returns
    -------
    dtype
        Some information
    
    """
    
    
    R = np.zeros(len(Simulation.System.molecule_types))
    for i, moltype in enumerate(Simulation.System.molecule_types):
        R[i] = Simulation.System.molecule_types[moltype].radius
        
    if Compartment_Number > 0:
        Compartment = Simulation.System.Compartments[Compartment_Number]
    elif Compartment_Number==0:
        Compartment = Simulation.System
        
    points = np.array(points)
    ptype = np.array(ptype)

    AABB = Compartment.AABB
    
    dim = points.shape[1]
    width = AABB[1][0]-AABB[0][0]
    height = AABB[1][1]-AABB[0][1]
    depth = AABB[1][2]-AABB[0][2]
        
    n = len(points)
    
    if dim ==2:
        plt.figure(figsize = (3,3), dpi =300)
        plt.scatter(points[:,0], points[:,1], s = 10, c = 'k')
            
        plt.xlim(AABB[0][0],AABB[1][0])
        plt.ylim(AABB[0][1],AABB[1][1])
        
        if n<=10:
            for i in np.arange(0,n,1/n):
                plt.axvline(i)
                plt.axhline(i)
                
        # plt.show()
   
    

    
    if dim ==3:
    
        plt.figure(figsize = (4,4), dpi = 150)
        # ax = plt.axes(projection='3d', proj_type = 'persp')
        ax = plt.axes(projection='3d', proj_type = 'ortho')
        ax.set_box_aspect((width,height,depth))
        
        ax.scatter(points[:,0], points[:,1], points[:,2], s = R[ptype], c=ptype) 
        
        for tri_id in Compartment.triangle_ids:
            X = Simulation.System.Mesh['triangles'][tri_id]
    
            verts = [Simulation.System.vertices[X[0]], Simulation.System.vertices[X[1]], Simulation.System.vertices[X[2]]]
            mesh = Poly3DCollection([verts], linewidths=1)
            face_color = (141 / 255, 184 / 255, 226 / 255, 0.1)
            edge_color = (50 / 255, 50 / 255, 50 / 255)
            mesh.set_facecolor(face_color)
            mesh.set_edgecolor(edge_color)
            ax.add_collection3d(mesh)
        
        if Compartment==Simulation.System:
            pc, lc = draw_cuboid(positions=[-Simulation.System.box_lengths/2], sizes=[Simulation.System.box_lengths])
            ax.add_collection3d(pc)    
            ax.add_collection3d(lc) 
                
        ax.set_xlim(AABB[0][0],AABB[1][0])
        ax.set_ylim(AABB[0][1],AABB[1][1])
        ax.set_zlim(AABB[0][2],AABB[1][2])
        # ax.view_init(30, 185)
        ax.view_init(0, 0)
        # ax.set_xlabel('X')
        ax.set_xticklabels([])
        ax.set_ylabel('Y',labelpad=6)
        ax.tick_params(axis='y', pad=-5)
        plt.yticks(rotation=45)
        ax.set_zlabel('Z',labelpad=14)
        ax.tick_params(axis='z', pad=9)
        
        # plt.show()
        
        
        if len(points)<=2000:
            plt.figure()
            ax = plt.axes(projection='3d', proj_type = 'persp')
            # ax = plt.axes(projection='3d', proj_type = 'ortho')
            ax.set_box_aspect((width,height,depth))
            
            cols = ['r','g', 'b','m', 'y']
            Aspects = []
            for i,pos in enumerate(points):
                Aspects.append(draw_sphere(ax, pos,R[ptype[i]], cols[ptype[i]]))
                
            for tri_id in Compartment.triangle_ids:
                X = Simulation.System.Mesh['triangles'][tri_id]
        
                verts = [Simulation.System.vertices[X[0]], Simulation.System.vertices[X[1]], Simulation.System.vertices[X[2]]]
                mesh = Poly3DCollection([verts], linewidths=1, alpha=0.5)
                face_color = (141 / 255, 184 / 255, 226 / 255)
                edge_color = (50 / 255, 50 / 255, 50 / 255)
                mesh.set_facecolor(face_color)
                mesh.set_edgecolor(edge_color)
                ax.add_collection3d(mesh)
    
            
            # Aspects = np.array(Aspects)
            # ax.set_box_aspect((np.max(Aspects[:,:,0])-np.min(Aspects[:,:,0]),np.max(Aspects[:,:,1])-np.min(Aspects[:,:,1]),(np.max(Aspects[:,:,2])-np.min(Aspects[:,:,2]))*0.9))
            
            if Compartment==Simulation.System:
                pc, lc = draw_cuboid(positions=[-Simulation.System.box_lengths/2], sizes=[Simulation.System.box_lengths])
                ax.add_collection3d(pc)    
                ax.add_collection3d(lc) 
                        
            ax.set_xlim(AABB[0][0],AABB[1][0])
            ax.set_ylim(AABB[0][1],AABB[1][1])
            ax.set_zlim(AABB[0][2],AABB[1][2])
            
            if show:
                plt.show()
            
            if save_fig == True:
                if fig_name is None:
                    fig_name = Simulation.file_name + '_' + Compartment.name
                if fig_path is None:
                    fig_path = Simulation.fig_path
                plt.savefig(fig_path / (fig_name+'_Sphere_packing.png'), bbox_inches="tight", dpi = 300)
                # plt.savefig('Sphere_packing.svg', dpi = 300)
                
        
#%%

def plot_mobility_matrix(molecule, Simulation, save_fig = False, fig_name = None , fig_path = None, color_scheme = 'colored', show = True):
    
    """A brief description of what the function (method in case of classes) is and what it’s used for
    
    Parameters
    ----------
    parameter_1 : dtype
        Some Information
    parameter_2 : dtype
        Some Information
    
    Raises
    ------
    NotImplementedError (just an example)
        Brief explanation of why/when this exception is raised
    
    Returns
    -------
    dtype
        Some information
    
    """
    
    my_molecule = Simulation.System.molecule_types[molecule]
    
    # Titles = [r'$D_{tt} \, (\mu m^2/s)$', '$D_{rr} \, (rad/s)$', '$D_{tr}  \, (\mu m/s)$', '$D_{rt} \, (\mu m/s)$']
    Titles = [r'$D_{tt} \, $'+'$({0}^2/{1})$'.format(Simulation.units['Length'], Simulation.System.time_unit), '$D_{rr} \,$'+'$(rad^2/{})$'.format(Simulation.System.time_unit), '$D_{tr}  \, $'+'$({0}/{1})$'.format(Simulation.System.length_unit, Simulation.System.time_unit), '$D_{rt} \,  $'+'$({0}/{1})$'.format(Simulation.System.length_unit, Simulation.System.time_unit)]
    
    Data = [my_molecule.mu_tb*(Simulation.System.kbt), my_molecule.mu_rb*(Simulation.System.kbt)]
    
    
    # create figure
    fig = plt.figure(figsize=(10,3), constrained_layout=False, dpi = 150)
    fig.subplots_adjust(hspace = 0.0, wspace = 0.5)
    
    # define widths and heights of the individual subfigures
    widths = [1,1,1]
    heights = [1]
    # create the grid structure
    gs = fig.add_gridspec(ncols=3, nrows=1, width_ratios=widths, height_ratios=heights)
    
    # construct the top (axes object) level subfigure
    left = fig.add_subplot(gs[0,0], projection='3d', proj_type = 'persp')
    mid = fig.add_subplot(gs[0,1])
    mid.set_aspect('equal', adjustable='box')
    right = fig.add_subplot(gs[0,2])
    right.set_aspect('equal', adjustable='box')

    
    Aspects = []
    # for pos, R in zip(my_molecule.pos_rb[:], my_molecule.radii_rb[:]):
    #     Aspects.append(draw_sphere(top, pos*1e9, R*1e9, 'grey'))

    for particle_type, pos, R in zip(my_molecule.types, my_molecule.pos, my_molecule.radii):
        
        i = int(Simulation.System.particle_types[particle_type][0]['id'])
        
        if R==0:
            if color_scheme == 'flat':
                Aspects.append(draw_sphere(left, pos, min(my_molecule.radii_rb)*(3-2*(min(my_molecule.radii_rb)/max(my_molecule.radii_rb))**(1/3))/3, 'red'))
            else:
                Aspects.append(draw_sphere(left, pos, min(my_molecule.radii_rb)*(3-2*(min(my_molecule.radii_rb)/max(my_molecule.radii_rb))**(1/3))/3, patch_colors[i%24]))
        else:
            if color_scheme == 'flat':
                Aspects.append(draw_sphere(left, pos, R, 'white'))
            else:
                Aspects.append(draw_sphere(left, pos, R, core_colors[i%24]))
            
            
    xmin, xmax = left.get_xlim()
    ymin, ymax = left.get_ylim()
    zmin, zmax = left.get_zlim()
    
    left.plot([1.25*xmin ,1.25*xmax],[0,0],[0,0], 'k', linewidth = 1, zorder = 0)
    left.plot([0 ,0],[1.75*ymin,1.75*ymax],[0,0], 'k', linewidth = 1, zorder = 0)
    left.plot([0 ,0],[0,0],[1.25*zmin,1.2*zmax], 'k', linewidth = 1, zorder = 0)
    
    left.set_box_aspect((xmax-xmin,ymax-ymin,(zmax-zmin)*0.9))
    
    left.set_xlim(xmin,xmax)
    left.set_ylim(ymin,ymax)
    left.set_zlim(zmin*0.9,zmax*0.9)
    
    # Aspects = np.array(Aspects)
    # left.set_box_aspect((np.max(Aspects[:,:,0])-np.min(Aspects[:,:,0]),np.max(Aspects[:,:,1])-np.min(Aspects[:,:,1]),(np.max(Aspects[:,:,2])-np.min(Aspects[:,:,2]))*0.9))

    
    left.set_title(my_molecule.name, y=1.05, pad=0)
    
    color = (1.0, 1.0, 1.0, 0.0)
    left.w_xaxis.set_pane_color(color)
    left.w_yaxis.set_pane_color(color)
    left.w_zaxis.set_pane_color(color)
    left.w_xaxis.line.set_color(color)
    left.w_yaxis.line.set_color(color)
    left.w_zaxis.line.set_color(color)

    left.set_xlabel('x in ${}$'.format(Simulation.units['Length']), labelpad=4)
    left.set_ylabel('y in ${}$'.format(Simulation.units['Length']), labelpad=4)
    left.set_zlabel('z in ${}$'.format(Simulation.units['Length']), labelpad=4)
    
    # left.tick_params(axis='x', pad=-2)
    # left.tick_params(axis='y', pad=-2)
    # left.tick_params(axis='z', pad=-2)
    
    left.locator_params(axis='x', nbins=3)
    left.locator_params(axis='y', nbins=3)
    left.locator_params(axis='z', nbins=3)

    mid.set_title(Titles[0], pad=10)
    
    # divider = make_axes_locatable(mid)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # im1 = mid.matshow(Data[0])
    sns.heatmap(Data[0], ax = mid, linewidths=.5, annot=True, annot_kws={"fontsize":8}, fmt='.3g', cbar_kws={"shrink": .7, "ticks": np.round([np.min(Data[0]), (np.max(Data[0])-np.min(Data[0]))/2, np.max(Data[0])*0.95],3)}, cmap = "YlOrBr")
    # cb = fig.colorbar(im1, cax=cax, orientation='vertical')
    # cb.formatter.set_scientific(True)
    # cb.locator.axis.get_offset_text().set_x(5)
    
    right.set_title(Titles[1], pad=10)
    
    # divider = make_axes_locatable(right)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # im1 = right.matshow(Data[0])
    # cb = fig.colorbar(im1, cax=cax, orientation='vertical')
    # cb.formatter.set_scientific(True)
    # cb.locator.axis.get_offset_text().set_x(5)
    sns.heatmap(Data[1], ax = right, linewidths=.5, annot=True, annot_kws={"fontsize":8}, fmt=".2g", cbar_kws={"shrink": .7, "ticks": np.round([np.min(Data[1]), (np.max(Data[1])-np.min(Data[1]))/2, np.max(Data[1])*0.95],3)}, cmap = "YlOrBr")
        
    if show:
        plt.show()
    
    # fig.tight_layout() 
    if save_fig == True:
        if fig_name is None:
            fig_name = Simulation.file_name + '_' + molecule
        if fig_path is None:
            fig_path = Simulation.fig_path
        print('saved figure.')
        fig.savefig(fig_path / (fig_name+'_DiffusionMatrix.png'), bbox_inches="tight", dpi = 300)
    
    
    
#%%

def plot_potential(Simulation, Potentials, yU_limits, yF_limits, r_limits = None, show = True, save_fig = False):
    
    """A brief description of what the function (method in case of classes) is and what it’s used for
    
    Parameters
    ----------
    parameter_1 : dtype
        Some Information
    parameter_2 : dtype
        Some Information
    
    Raises
    ------
    NotImplementedError (just an example)
        Brief explanation of why/when this exception is raised
    
    Returns
    -------
    dtype
        Some information
    
    """
    
    sns.set_style("whitegrid")
    # sns.set_context("paper")
    # sns.set_context("talk")
    # sns.set_context("poster")
    # sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    
    length_units_prefix = Simulation.System.length_units_prefix[Simulation.System.length_unit]
    
    if r_limits is None:
        r = np.linspace(0,10*length_units_prefix/1e9,100)[1:]
    else:
        r = np.linspace(r_limits[0],r_limits[1],100)[1:]
    
    plt.figure('figU', figsize = (4,3), dpi = 150)
    plt.figure('figF', figsize = (4,3), dpi = 150)
    for pot, args in Potentials:
        U=[]
        F=[]
        for ri in r:
            U.append(pot(ri, np.array(args))[0])
            F.append(pot(ri, np.array(args))[1])
        
        plt.figure('figU')
        plt.plot(r, U, label = pot.name.replace('_', ' '))

        plt.figure('figF')
        plt.plot(r, F, label = pot.name.replace('_', ' '))
    
    plt.figure('figU')
    plt.ylim(yU_limits[0],yU_limits[1])
    plt.axhline(0, color= 'k', linewidth = 1, linestyle = '--')
    plt.xlabel('r in {}'.format(Simulation.units['Length']))
    plt.ylabel('U in {}'.format(Simulation.units['Energy']))
    plt.legend()
    
    if save_fig == True:
        fig_name = Simulation.file_name
        fig_path = Simulation.fig_path
        print('saved figure.')
        plt.savefig(fig_path / (fig_name+'_potential_energy.png'), bbox_inches="tight", dpi = 300)
    
    plt.figure('figF')
    plt.ylim(yF_limits[0],yF_limits[1])
    plt.axhline(0, color= 'k', linewidth = 1, linestyle = '--')
    plt.xlabel('r in {}'.format(Simulation.units['Length']))
    plt.ylabel('F in {}'.format(Simulation.units['Force']))
    plt.legend()
    
    if save_fig == True:
        fig_name = Simulation.file_name
        fig_path = Simulation.fig_path
        print('saved figure.')
        plt.savefig(fig_path / (fig_name+'_potential_force.png'), bbox_inches="tight", dpi = 300)
        
    if show:
        plt.show()

    
   
        
#%%

def plot_concentration_profile(Simulation, axis = 0, save_fig = False, fig_name = None , fig_path = None, show=True):
    
    """A brief description of what the function (method in case of classes) is and what it’s used for
    
    Parameters
    ----------
    parameter_1 : dtype
        Some Information
    parameter_2 : dtype
        Some Information
    
    Raises
    ------
    NotImplementedError (just an example)
        Brief explanation of why/when this exception is raised
    
    Returns
    -------
    dtype
        Some information
    
    """
    
    # cells_per_dim = np.array([15,5,5])
    # box_lengths = np.array([0.15,0.05,0.05])
    # cell_length_per_dim = box_lengths/cells_per_dim
    
    cells_axis = 100
    box_lengths = Simulation.System.box_lengths*(1+1e-10) # Some molecules may lie on the border, therefore the small multiplicative factor. Otherwise we could get out of bound errors.
    cell_length_axis = box_lengths[axis]/cells_axis
    
    Histograms = {}
    for mol_type in Simulation.System.molecule_types:
        Histograms[mol_type] = np.zeros(cells_axis)
    
    axis_name = ['X', 'Y', 'Z']
    ax123 = set([0,1,2])
    ax123.remove(axis)
    ax123 = list(ax123)
    volume_cell = cell_length_axis*box_lengths[ax123[0]]*box_lengths[ax123[1]]
    
    for key0 in range(Simulation.RBs.occupied.n):
        key = Simulation.RBs.occupied[key0] 
        
        mol_type = Simulation.RBs[key]['name']
        
        i = int((Simulation.RBs[key]['pos'][axis]+box_lengths[axis]/2) / cell_length_axis)
        
        volume_mol = np.sum(4/3*np.pi*Simulation.System.molecule_types[mol_type].radii**3)
        
        Histograms[mol_type][i] += volume_mol
    
    
    for mol_type in Simulation.System.molecule_types:
        Histograms[mol_type]/=volume_cell
    
    plt.figure(figsize = (5,3), dpi =150)
    for mol_type in Simulation.System.molecule_types:
        plt.plot(np.linspace(0,box_lengths[axis],cells_axis),Histograms[mol_type], label = '{}'.format(mol_type))
    plt.legend()
    plt.xlabel(axis_name[axis]+' in $\mu m$')
    plt.ylabel('Packing fraction $\Phi$')
    plt.ylim(0)
    
        
    if save_fig == True:
        if fig_name is None:
            fig_name = Simulation.file_name + '_' + axis_name[axis]
        if fig_path is None:
            fig_path = Simulation.fig_path
        plt.savefig(fig_path / (fig_name+'_Profile.png'), bbox_inches="tight", dpi = 300)
    
    
    if show:
        plt.show()
