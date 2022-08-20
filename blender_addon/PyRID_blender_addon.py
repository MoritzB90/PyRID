bl_info = {
    "name" : "PyRID visualization",
    "author" : "Moritz Becker",
    "version" : (0, 1),
    "blender" : (3, 0, 0),
    "location" : "View3d > Tool",
    "warning" : "rendering does crash with preview enabled!",
    "wiki_url" : "...",
    "category" : "Molecular Dynamics Visualization",
}

import bpy
from mathutils import Vector
import numpy as np
#import pandas as pd
import logging
import os
from bpy.app.handlers import persistent
import ctypes
import os
from sys import platform


#%%

"""
You need to follow certain naming conventions:
    * The simulation box must be named 'Box'.
    * The triangle border of a compartment, adjacent to the simulation 
    box, must be named 'border'.
    * The triangles of the simulation box that align with the cross section 
      of the compartment with the 'Box' must have the same name as the compartment.

Also, meshes must be triangulated (Modifiers -> Triangulate)! We advise you to use
'Fixed' as Quad Method.
For periodic boundary conditions to work properly, the compartment should extend
above the Box itself.
"""

class compartment(object):
    
    def __init__(self):
        
        self.triangles = []
        self.vertices = []
        self.face_maps = {}


def export_obj(objFile_path):

    Compartment_names = []

    for comps in bpy.data.objects:
        Compartment_names.append(comps.name)
        
    print(Compartment_names)

    Compartments = {}

    for comp_name in Compartment_names:
        
        Compartments[comp_name] = compartment()
        
        #bpy.ops.object.select_all(action='DESELECT')
        
        bpy.context.view_layer.objects.active = bpy.data.objects[comp_name]

        me = bpy.data.objects[comp_name]

        #------------------------------------
        # Triangles
        #------------------------------------
        print(len(me.data.polygons))

            
        for i in range(len(me.data.polygons)):
            
            pol = me.data.polygons[i]
            
            tri0 = pol.vertices[0]
            tri1 = pol.vertices[1]
            tri2 = pol.vertices[2]
            
            Compartments[comp_name].triangles.append([tri0+1, tri1+1, tri2+1])
                
        #------------------------------------
        # Vertices
        #------------------------------------
        
        global_mat = me.matrix_world
        
        for i in range(len(me.data.vertices)):
            v = global_mat @ me.data.vertices[i].co
            vertex = [v.x, v.y, v.z]

            Compartments[comp_name].vertices.append(vertex)
            
        #------------------------------------
        # Face Maps
        #------------------------------------
            
        for i in range(len(me.face_maps)):
            
            map_name = me.face_maps[i].name
            #print(map_name)
            
            
            bpy.ops.object.editmode_toggle()
            bpy.ops.mesh.select_all(action='DESELECT')

            me.face_maps.active_index = me.face_maps[i].index

            bpy.ops.object.face_map_select()
            
            bpy.ops.object.editmode_toggle()
            
            Compartments[comp_name].face_maps[map_name] = []

            for j in range(len(me.data.polygons)):
                
                if me.data.polygons[j].select:
                    
                    Compartments[comp_name].face_maps[map_name].append(me.data.polygons[j].index)
                    
            print(comp_name, map_name, ': ', len(Compartments[comp_name].face_maps[map_name]))
                
        #print(face_maps)
                
    #-----------------

    #%%

    f = open(objFile_path+'.obj', "w")

    n_vertices = 0
    for comp_name in Compartments:
        
        vertex_counter = 0
        
        f = open(objFile_path+'.obj', "a")
        
        f.write('o {}\n'.format(comp_name))
        
        current_map = 'None'
        for vertex in Compartments[comp_name].vertices:
            f.write('v {0} {1} {2}\n'.format(vertex[0], vertex[1], vertex[2]))
            vertex_counter += 1
            
        for i, triangle in enumerate(Compartments[comp_name].triangles):
            in_map = False
            for map_name in Compartments[comp_name].face_maps:
                if i in Compartments[comp_name].face_maps[map_name]:
                    in_map = True
                    if map_name!=current_map:
                        current_map = map_name
                        f.write('g {}\n'.format(map_name))
                        
            if in_map == False and current_map != '(null)':
                current_map = '(null)'
                f.write('g (null)\n')
                
            f.write('f {0} {1} {2}\n'.format(triangle[0]+n_vertices, triangle[1]+n_vertices, triangle[2]+n_vertices))
            
        f.close()
        
        n_vertices += vertex_counter
    

    print('exported obj')

#%%



# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------


def make_path_absolute(key): 
    """ Prevent Blender's relative paths of doom 
        Source: https://sinestesia.co/blog/tutorials/avoid-relative-paths/
    """ 
    # This can be a collection property or addon preferences 
    props = bpy.context.scene.my_tool 
    sane_path = lambda p: os.path.abspath(bpy.path.abspath(p)) 
    
    if key in props and props[key].startswith('//'): 
        props[key] = sane_path(props[key]) 


#---
# ID Props
#---

global Types, Mol_Types, Type_Masks, Type_color, Radii, Radii_cutoff, mat, molecule, screensize, Mesh_Names, scene_gen

Types = []
Mesh_Names = []
Radii = {}
Radii_cutoff = {}
mat = {}
scene_gen = False

global start, end

start = []
end = []

global box_size

box_size = None

#user32 = ctypes.windll.user32
screensize = None #user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

#------
# Functions
#------


def read_table(Path):
    
    global start, end, Types
    
    lines = filter(None, (line.rstrip().replace('\t', ' ').replace('\n', ' ').split() for line in open(Path, 'r')))
    lines = list(lines)


    item_t = np.dtype([('atom', 'U24'), ('x', np.float64), ('y', np.float64), ('z', np.float64)],  align=True)
    array  = np.zeros(len(lines), dtype = item_t)
    
    start0=0
    end0=0
    Types_Set = set()
    
    for i,l in enumerate(lines):
        if len(l)==1:
            start.append(start0+1)
            start0 += int(l[0])+1
            end.append(start0) 
        else:
            for j,value in enumerate(l):
                array[i][j] = value
                Types_Set.add(array['atom'][i])
        
        if i%100000==0:
            print('lines read:', i, '/', len(lines))

    Types = list(Types_Set)
    Types.sort()
    Types.reverse()
    
    print(Types)
    
    return array



def read_box_table(Path):
    
    lines = filter(None, (line.rstrip().replace('\t', ' ').replace('\n', ' ').split() for line in open(Path, 'r')))
    lines = list(lines)

    item_t = np.dtype([('x', np.float64), ('y', np.float64), ('z', np.float64)],  align=True)
    array  = np.zeros(len(lines), dtype = item_t)
    
    for i,l in enumerate(lines):

        for j,value in enumerate(l):
            array[i][j] = value
    
    return array

        

def Viewport_Settings():
    
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas: # iterate through areas in current screen
            if area.type == 'VIEW_3D':
                for space in area.spaces: # iterate through spaces in current VIEW_3D area
                    if space.type == 'VIEW_3D': # check if space is a 3D view
                        space.shading.light = 'STUDIO'#'MATCAP'
                        space.shading.show_cavity = True
                        space.shading.cavity_type = 'SCREEN'
                        space.shading.curvature_ridge_factor = 1
                        space.shading.curvature_valley_factor = 1
                        space.shading.show_object_outline = True

    bpy.context.preferences.themes['Default'].view_3d.space.gradients.gradient = (0.7333333492279053, 0.9725490808486938, 1.0)
    bpy.context.preferences.themes['Default'].view_3d.space.gradients.high_gradient = (1.0,1.0,1.0)

    bpy.context.preferences.themes['Default'].view_3d.vertex_select = (0.3,1.0,0.0)
    bpy.context.preferences.themes['Default'].view_3d.edge_select = (0.3,1.0,0.0)
    bpy.context.preferences.themes['Default'].view_3d.face_select = (0.3,1.0,0.0, 0.18)
    bpy.context.preferences.themes['Default'].view_3d.editmesh_active = (0.3,1.0,0.0, 0.2)
    bpy.context.preferences.themes['Default'].view_3d.edge_sharp = (1.0,0.5,0.8)

    #Change render settings:
    bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
    bpy.context.scene.display.shading.light = 'STUDIO'#'MATCAP'
    bpy.context.scene.display.shading.show_cavity = True
    bpy.context.scene.display.shading.cavity_type = 'SCREEN'
    bpy.context.scene.display.shading.curvature_ridge_factor = 1
    bpy.context.scene.display.shading.curvature_valley_factor = 1
    bpy.context.scene.display.shading.show_object_outline = True
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.look = 'Medium High Contrast'

    bpy.context.scene.world.color = (0.65, 0.85, 0.86)
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0.65, 0.85, 0.86, 1)
    
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'

    bpy.context.scene.render.resolution_x = 1080
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.use_file_extension = False
    bpy.context.scene.render.ffmpeg.constant_rate_factor = 'PERC_LOSSLESS' #'HIGH'
    
    bpy.context.scene.world.cycles_visibility.diffuse = False
    bpy.context.scene.world.cycles_visibility.glossy = False
    bpy.context.scene.world.cycles_visibility.transmission = False
    bpy.context.scene.world.cycles_visibility.scatter = False


def add_mol_shape(mol_pos_name, name, radius, radius_cutoff, col_name="Collection"):
    
    particle_radius = 0.0
    if radius>0:
        particle_radius = radius
    elif radius_cutoff>0:
        particle_radius = radius_cutoff
        
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=particle_radius, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.context.active_object.name = name
    bpy.ops.object.shade_smooth()
    bpy.data.objects[name].data.materials.append(mat[name])
    #bpy.data.objects[name].hide_set(True)
    #bpy.data.objects[name].hide_render = True
    
        
    objects = bpy.data.objects
    a = objects[mol_pos_name]
    b = objects[name]
    b.parent = a
    a.instance_type = 'VERTS'

def add_mesh(name, verts, faces=None, edges=None, col_name="Collection"):    
    if edges is None:
        edges = []
    if faces is None:
        faces = []
        
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(mesh.name, mesh)
    col = bpy.data.collections.get(col_name)
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    mesh.from_pydata(verts, edges, faces)
    

def clear_system():
    
    global Types, Radii, Radii_cutoff, mat, Mesh_Names
    global start, end
    
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
        
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
        
    Types = []
    Mesh_Names = []
    Radii = {}
    Radii_cutoff = {}
    mat = {}
    
    start = []
    end = []
    
def load_mol_trace():
    
    global Types, Mol_Types, Type_Masks, Type_color, Radii, Radii_cutoff, molecule
    global start, end
    global box_size
    
    #filepath = bpy.data.filepath
    #directory = os.path.dirname(filepath)

    mytool = bpy.context.scene.my_tool
    
    #molecule = pd.read_table(directory+bpy.context.scene.my_tool.file_path, skiprows=0, delim_whitespace=True, names=['atom', 'x', 'y', 'z'])
    
    molecule = read_table(bpy.context.scene.my_tool.file_path)
    
    mytool.N = end[0]-start[0] #int(molecule['atom'][0])
    mytool.NFrames = len(start) #int(len(molecule)/(int(molecule['atom'][0])+1))
    
    #Types = list(set(molecule['atom']))
    #Types.remove(molecule['atom'][0])
    #Types.sort()
    #Types.reverse()
    #get_object_list_callback()

    #print(Types)

    #Mol_Types = molecule['atom'][0*(mytool.N+1)+1:(0+1)*(mytool.N+1)]
    Mol_Types = molecule['atom'][start[0]:end[0]]
    
    Type_Masks = {}
    Type_color = {}
    
    #directory_rel = os.path.dirname(mytool.props_path)
    #filename = os.path.basename(mytool.props_path)
    #if directory==directory_rel:
    #    path = mytool.props_path
    #else:
    #    path = directory+mytool.props_path
    path = mytool.props_path
    print(path)
    try: 
        lines = list(filter(None, (line.rstrip().replace('\t', ' ').replace('\n', ' ').split('=') for line in open(path, 'r'))))
        
        Radii = eval(lines[0][1]) 
        Radii_cutoff = eval(lines[1][1]) 
        mytool.box_lengths = eval(lines[7][1]) 
        mytool.mesh_scale = eval(lines[6][1]) 
        
        x = 1/max(mytool.box_lengths)
        exponent = int(np.ceil(np.log10(abs(x)))) if x != 0 else 0
        mytool.system_scale = 10**exponent
        
        #print('system_scale: ', mytool.system_scale)
        
        #for key in Radii:
        #    if Radii[key] == 0.0:
        #        Radii[key] = min([Radii[i] for i in Radii.keys() if Radii[i]>0])/10
                
        #for key in Radii_cutoff:
        #    if Radii_cutoff[key] == 0.0:
        #        Radii_cutoff[key] = min([Radii_cutoff[i] for i in Radii_cutoff.keys() if Radii_cutoff[i]>0])/10
        
    except:
        for name in Types:
            Radii[name+"_shape"] = 0.01        
            Radii_cutoff[name+"_shape"] = 0.01 
    
    radius_min = 1e6
    for key in Radii:
        if 0.0<Radii[key]<radius_min:
             radius_min = Radii[key]
        if 0.0<Radii_cutoff[key]<radius_min:
             radius_min = Radii_cutoff[key]
         
    for key in Radii:
        if Radii[key]==0.0 and Radii_cutoff[key]==0.0:
            Radii[key] = radius_min
    
    # colorblind
    colors = [(0.00392156862745098, 0.45098039215686275, 0.6980392156862745, 1.0),
     (0.8705882352941177, 0.5607843137254902, 0.0196078431372549, 1.0),
     (0.00784313725490196, 0.6196078431372549, 0.45098039215686275, 1.0),
     (0.8352941176470589, 0.3686274509803922, 0.0, 1.0),
     (0.8, 0.47058823529411764, 0.7372549019607844, 1.0),
     (0.792156862745098, 0.5686274509803921, 0.3803921568627451, 1.0),
     (0.984313725490196, 0.6862745098039216, 0.8941176470588236, 1.0),
     (0.5803921568627451, 0.5803921568627451, 0.5803921568627451, 1.0),
     (0.9254901960784314, 0.8823529411764706, 0.2, 1.0),
     (0.33725490196078434, 0.7058823529411765, 0.9137254901960784, 1.0)]
     
    #-----------------------------------
    # import seaborn as sns
    # Colors = sns.color_palette("tab10")+sns.color_palette("Set2")+sns.color_palette("hls")
    # Colors = np.array(Colors)
    # Colors = np.hstack((Colors,np.ones((len(Colors),1))))
    #-----------------------------------
    
    # tab10 + Set2 + hls
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
     
    core_idx = 0
    patch_idx = 0
    idx = 0
    for j,name in enumerate(Types):
        Type_Masks[name] = np.array(Mol_Types==name)
        if 'Core' in name:
            Type_color[name+"_shape"] = core_colors[core_idx%24]
            core_idx += 1
        elif 'Patch' in name:
            Type_color[name+"_shape"] = patch_colors[patch_idx%24]
            patch_idx += 1
        else:
            Type_color[name+"_shape"] = colors[idx%10]
            idx+=1
                    
        Radii[name+"_shape"] *= mytool.system_scale
        Radii_cutoff[name+"_shape"] *= mytool.system_scale
            
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = int(len(start)/mytool.step) #int(len(molecule)/(mytool.N+1))#1000
    
    path = mytool.box_path
    if path!='':
        box_size = read_box_table(path)


def generate_scene():
    
    bpy.context.scene.cycles.caustics_reflective = False
    bpy.context.scene.cycles.caustics_refractive = False
    bpy.context.scene.world.cycles.volume_sampling = 'MULTIPLE_IMPORTANCE'
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.cycles.preview_samples = 8


    mytool = bpy.context.scene.my_tool
    global mat
    
    bpy.context.scene.cursor.location[0] = 0
    bpy.context.scene.cursor.location[1] = 0
    bpy.context.scene.cursor.location[2] = 0

    bpy.context.scene.cursor.rotation_euler[0] = 0
    bpy.context.scene.cursor.rotation_euler[1] = 0
    bpy.context.scene.cursor.rotation_euler[2] = 0


    bpy.ops.mesh.primitive_cube_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))

    bpy.data.objects["Cube"].select_set(True)
    bpy.ops.transform.resize(value=(1.001*mytool.box_lengths[0]/2*mytool.system_scale, 1.001*mytool.box_lengths[1]/2*mytool.system_scale, 1.001*mytool.box_lengths[2]/2*mytool.system_scale), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    bpy.data.objects["Cube"].select_set(False)
    mat['Cube'] = bpy.data.materials.new(name='Cube'+'_material')
    mat['Cube'].diffuse_color = (0.8,0.8,0.8,0.05)
    bpy.data.objects['Cube'].data.materials.append(mat['Cube'])

    #bpy.data.objects["Cube"].show_wire = True
    bpy.data.objects["Cube"].display_type = 'SOLID'#'WIRE'
    bpy.data.objects["Cube"].display.show_shadows = False
    bpy.data.objects["Cube"].hide_render = False # True

    bpy.data.objects["Cube"].active_material.blend_method = 'BLEND'
    bpy.data.objects["Cube"].active_material.show_transparent_back = False
    bpy.data.objects["Cube"].active_material.use_nodes = True
    bpy.data.materials["Cube_material"].node_tree.nodes["Principled BSDF"].inputs[21].default_value = 0.0
    bpy.data.objects["Cube"].active_material.shadow_method = 'NONE'
    
    bpy.ops.object.modifier_add(type='BEVEL')
    bpy.data.objects["Cube"].modifiers["Bevel"].width = 0.01
    bpy.data.objects["Cube"].modifiers["Bevel"].limit_method = 'NONE'
    bpy.data.objects["Cube"].modifiers["Bevel"].segments = 2
    bpy.data.objects["Cube"].modifiers["Bevel"].profile = 1
    bpy.data.objects["Cube"].modifiers["Bevel"].material = 1
    mat['Cube_edge'] = bpy.data.materials.new(name='Cube'+'_material_edge')
    mat['Cube_edge'].diffuse_color = (0.0,0.0,0.0,1.0)
    bpy.data.objects['Cube'].data.materials.append(mat['Cube_edge'])
    bpy.data.objects['Cube'].active_material_index = 1
    bpy.data.objects["Cube"].active_material.use_nodes = True
    bpy.data.materials["Cube_material_edge"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.0,0.0,0.0,1.0)
    bpy.data.materials["Cube_material_edge"].node_tree.nodes["Principled BSDF"].inputs[7].default_value = 0
    
    bpy.data.objects["Cube"].visible_shadow = False
    bpy.data.objects["Cube"].visible_volume_scatter = False
    bpy.data.objects["Cube"].visible_transmission = False
    bpy.data.objects["Cube"].visible_glossy = False
    bpy.data.objects["Cube"].visible_diffuse = False

    
    for name in Types:
        
        # Initialize Mesh with on vertex
        verts = [(0, 0, 0),]
        edges = []
        faces = []
        add_mesh(name, verts)

        mat[name+"_shape"] = bpy.data.materials.new(name=name+'_material')
        mat[name+"_shape"].diffuse_color = Type_color[name+"_shape"]
        add_mol_shape(name, name+"_shape", Radii[name+"_shape"], Radii_cutoff[name+"_shape"], col_name="Collection")
        bpy.data.objects[name+"_shape"].hide_set(True)
        
        
        
    # Prepare particle system
    n = []
    numbers = []
    for j,name in enumerate(Types):

        #degp = bpy.context.evaluated_depsgraph_get()
        
        n.append(sum(Type_Masks[name]))
        numbers.append(sum(Type_Masks[name]))


    #Mesh_Names = ['Postsynapse', 'Presynapse']
    for Mesh_Name in Mesh_Names:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[Mesh_Name].select_set(True)
        bpy.ops.transform.resize(value=(mytool.mesh_scale*mytool.system_scale, mytool.mesh_scale*mytool.system_scale, mytool.mesh_scale*mytool.system_scale), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
        mat[Mesh_Name] = bpy.data.materials.new(name=Mesh_Name+'_material')
        mat[Mesh_Name].diffuse_color = (0.8,0.8,0.8,0.5)
        bpy.data.objects[Mesh_Name].data.materials.append(mat[Mesh_Name])
        bpy.data.objects[Mesh_Name].data.materials.pop(index=0)
        bpy.data.objects[Mesh_Name].select_set(False)
        
        bpy.data.objects[Mesh_Name].active_material.blend_method = 'BLEND'
        bpy.data.objects[Mesh_Name].active_material.show_transparent_back = False
        bpy.data.objects[Mesh_Name].active_material.use_nodes = True
        bpy.data.materials[Mesh_Name+"_material"].node_tree.nodes["Principled BSDF"].inputs[21].default_value = 0.3
        
        #!!!
        #bpy.ops.object.select_all(action='DESELECT')
        #bpy.data.objects[Mesh_Name].select_set(True)
        #bpy.ops.object.editmode_toggle()
        #bpy.ops.mesh.select_all(action='SELECT')
        #bpy.ops.mesh.mark_sharp(clear=True)
        #bpy.ops.object.editmode_toggle()

        
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, -2*max(mytool.box_lengths[:])*mytool.system_scale, 0), rotation=(np.pi/2, 0, 0), scale=(1, 1, 1))
    bpy.data.objects['Camera'].data.type = 'ORTHO'
    bpy.data.objects['Camera'].data.ortho_scale = max(mytool.box_lengths[:])*mytool.system_scale*1.05
    
    bpy.ops.object.empty_add(type='SPHERE', align='WORLD', location=(0, 0, 0), scale=(1*mytool.system_scale, 1*mytool.system_scale, 1*mytool.system_scale))
    
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Empty'].select_set(True)
    bpy.ops.transform.resize(value=(5, 5, 5))
    
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Camera'].select_set(True)
    bpy.data.objects['Empty'].select_set(True)
    #bpy.data.objects['Camera'].parent = bpy.data.objects['Empty']
    bpy.ops.object.parent_set(type='OBJECT')

    global scene_gen
    scene_gen = True


    


@persistent
def frame_change_handler(scene):
    """ Update the viz data every time a frame is changed. """
    
    #print('scene_gen:', scene_gen)
    global start, end
    
    if scene_gen:
        
        mytool = bpy.context.scene.my_tool
        
        if scene.frame_current<mytool.NFrames: # and bpy.context.scene['index']!=scene.frame_current
            #if (not curr_frame == scene.frame_current):
            print('current frame: ', scene.frame_current)
            
            scene = bpy.context.scene
            cFrame = scene.frame_current
            i=cFrame*mytool.step
            mytool.N = end[i]-start[i]
            
            bpy.context.scene['index'] = i
            #x = molecule['x'][i*(mytool.N+1)+1:(i+1)*(mytool.N+1)]*mytool.system_scale
            #y = molecule['y'][i*(mytool.N+1)+1:(i+1)*(mytool.N+1)]*mytool.system_scale
            #z = molecule['z'][i*(mytool.N+1)+1:(i+1)*(mytool.N+1)]*mytool.system_scale
            #Mol_Types = molecule['atom'][i*(mytool.N+1)+1:(i+1)*(mytool.N+1)]
            x = molecule['x'][start[i]:end[i]]*mytool.system_scale
            y = molecule['y'][start[i]:end[i]]*mytool.system_scale
            z = molecule['z'][start[i]:end[i]]*mytool.system_scale
            Mol_Types = molecule['atom'][start[i]:end[i]]
                        
        #    # Make the new mesh our object of interest
            #mol_pos_mesh = bpy.data.objects['Patch_1'].data

        #    # Read new vertex position data from file (depending on current frame) and update mesh:
            #current_frame = bpy.context.scene.frame_current
            #mol_pos = read_data(current_frame)
            
            for j,name in enumerate(Types):    
                
                mol_pos_mesh = bpy.data.objects[name].data
                mol_pos_obj = bpy.data.objects[name]
                
                Type_Masks[name] = np.array(Mol_Types==name)
                mol_pos = np.dstack((x[Type_Masks[name]],y[Type_Masks[name]],z[Type_Masks[name]])).ravel()

                change_mesh(mol_pos_obj, mol_pos_mesh, mol_pos)
                
            
            if box_size is not None:
                bpy.data.objects['Cube'].dimensions = Vector(box_size[i])*mytool.system_scale
            
            print('----------------------')


        
def read_data(current_frame):
    mol_pos = np.random.rand(3*(current_frame+1))*current_frame*0.01
    
    return mol_pos

def change_mesh(mol_pos_obj,mol_pos_mesh, mol_pos):
    
    #bpy.context.view_layer.objects.active = obj
    #bpy.ops.object.mode_set(mode='EDIT')
    #bpy.ops.mesh.delete(type='VERT')
    #bpy.ops.object.mode_set(mode='OBJECT')
    
    scn_objs = bpy.context.scene.collection.children[0].objects
    meshes = bpy.data.meshes
    objs = bpy.data.objects
    
    mol_pos_mesh_name = mol_pos_mesh.name
    print(mol_pos_mesh_name)
    
    scn_objs.unlink(mol_pos_obj)
    objs.remove(mol_pos_obj)
    meshes.remove(mol_pos_mesh)
    bpy.data.objects[mol_pos_mesh_name+'_shape'].parent = None
    
    mol_pos_mesh = meshes.new(mol_pos_mesh_name)
    mol_obj = objs.new(mol_pos_mesh_name, mol_pos_mesh)
    scn_objs.link(mol_obj)
    
    # Add and set values of vertices at positions of molecules
    # This uses vertices.add(), but where are the old vertices removed?
    mol_pos_mesh.vertices.add(len(mol_pos)//3)
    print('number molecules: ', len(mol_pos)//3)
    mol_pos_mesh.vertices.foreach_set("co", mol_pos)
    #mol_pos_mesh.vertices.foreach_set("normal", mol_orient)
    
    objects = bpy.data.objects
    a = objects[mol_pos_mesh_name]
    b = objects[mol_pos_mesh_name+'_shape']
    b.parent = a
    a.instance_type = 'VERTS'
    
    
def view_animation():
    
    # Modify render settings
    #render = bpy.context.scene.render
    #render.resolution_x = render.resolution_x*screensize[1]/render.resolution_x
    #render.resolution_y = render.resolution_y*screensize[0]/render.resolution_y
    #render.resolution_percentage = 100

    ## Modify preferences (to guaranty new window)
    prefs = bpy.context.preferences
    prefs.view.render_display_type = "WINDOW"

    ## Call image editor window
    bpy.ops.render.view_show("INVOKE_DEFAULT")

    ## Change area type
    area = bpy.context.window_manager.windows[-1].screen.areas[0]
    area.type = "CLIP_EDITOR"
    
    #bpy.ops.render.play_rendered_anim()
    directory = os.path.dirname(bpy.context.scene.render.filepath)
    filename = os.path.basename(bpy.context.scene.render.filepath)
    bpy.ops.clip.open(directory=directory, files=[{"name": filename}], relative_path=True)

    # Restore render settings and preferences
    #render.resolution_x = 1920
    #render.resolution_y = 1920


def view_render():

    ## Modify preferences (to guaranty new window)
    prefs = bpy.context.preferences
    prefs.view.render_display_type = "WINDOW"

    ## Call image editor window
    bpy.ops.render.view_show("INVOKE_DEFAULT")

    ## Change area type
    area = bpy.context.window_manager.windows[-1].screen.areas[0]
    area.type = "IMAGE_EDITOR"
    bpy.data.images["Render Result"].name = "Render Result"

    
    

def rotate_camera():
    
    mytool = bpy.context.scene.my_tool
        
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Empty'].select_set(True)
    bpy.context.view_layer.objects.active =  bpy.data.objects['Empty']
    
    #bpy.context.scene.frame_current = 0
    bpy.context.object.rotation_euler[2] = 0.0
    #bpy.ops.anim.keyframe_insert_menu(type='Rotation')
    bpy.data.objects['Empty'].keyframe_insert('rotation_euler', frame=0)
    
    #bpy.context.scene.frame_current = mytool.NFrames
    bpy.context.object.rotation_euler[2] = 6.28319
    #bpy.ops.anim.keyframe_insert_menu(type='Rotation')
    bpy.data.objects['Empty'].keyframe_insert('rotation_euler', frame=mytool.NFrames)

def fix_camera():
    
    mytool = bpy.context.scene.my_tool
        
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Empty'].select_set(True)
    bpy.context.view_layer.objects.active =  bpy.data.objects['Empty']

    #bpy.context.scene.frame_current = mytool.NFrames
    bpy.context.object.rotation_euler[2] = 0.0
    #bpy.ops.anim.keyframe_insert_menu(type='Rotation')
    bpy.data.objects['Empty'].keyframe_insert('rotation_euler', frame=mytool.NFrames)
    

def add_lights():

    scene = bpy.context.scene
    mytool = scene.my_tool
    scale_loc = max(mytool.box_lengths[:])*mytool.system_scale/5
    scale_size = max(mytool.box_lengths[:])*mytool.system_scale/5
    scale_energy = (max(mytool.box_lengths[:])*mytool.system_scale)**2/25
    
    #-------------------------
    
    bpy.ops.object.light_add(type='AREA', radius=1, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Area'].select_set(True)
    
    bpy.context.object.rotation_euler[0] = np.pi/3
    bpy.context.object.location[1] = -6/5 * scale_loc
    bpy.context.object.location[2] = 2.5/5 * scale_loc
    bpy.context.object.data.size = 1.5 * scale_size
    bpy.context.object.data.energy = 200 * scale_energy
    bpy.context.object.data.specular_factor = 0
    
    bpy.context.object.parent = bpy.data.objects['Empty']
    
    #-------------------------
    
    bpy.ops.object.light_add(type='AREA', radius=1, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Area.001'].select_set(True)
    
    bpy.context.object.rotation_euler[0] = np.pi/3
    bpy.context.object.rotation_euler[2] = np.pi/4
    bpy.context.object.location[0] = 4/5 * scale_loc
    bpy.context.object.location[1] = -4/5 * scale_loc
    bpy.context.object.location[2] = 2.5/5 * scale_loc
    bpy.context.object.data.size = 1.5 * scale_size
    bpy.context.object.data.energy = 100 * scale_energy
    bpy.context.object.data.specular_factor = 0

    bpy.context.object.parent = bpy.data.objects['Empty']
    
    #-------------------------
    
    bpy.ops.object.light_add(type='AREA', radius=1, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Area.002'].select_set(True)
    
    bpy.context.object.rotation_euler[0] = np.pi/3
    bpy.context.object.rotation_euler[2] = -np.pi/4
    bpy.context.object.location[0] = -4/5 * scale_loc
    bpy.context.object.location[1] = -4/5 * scale_loc
    bpy.context.object.location[2] = 2.5/5 * scale_loc
    bpy.context.object.data.size = 1.5 * scale_size
    bpy.context.object.data.energy = 600 * scale_energy
    
    bpy.context.object.parent = bpy.data.objects['Empty']

    #-------------------------
    
    bpy.ops.object.light_add(type='AREA', radius=1, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Area.003'].select_set(True)
    
    bpy.context.object.rotation_euler[0] = -np.pi/2
    bpy.context.object.location[1] = 6/5 * scale_loc
    bpy.context.object.data.size = 1.5 * scale_size
    bpy.context.object.data.energy = 300 * scale_energy
    bpy.context.object.data.use_shadow = False
    bpy.context.object.data.specular_factor = 0.3
    
    bpy.context.object.parent = bpy.data.objects['Empty']    
    
    #-------------------------
    
    bpy.context.scene.render.engine = 'CYCLES'#'BLENDER_EEVEE'
    bpy.context.space_data.shading.type = 'RENDERED'
    
# ------------------------------------------------------------------------
#    Scene Properties
# ------------------------------------------------------------------------

#Function to create the item list you want to display in the select box
def get_object_list_callback(scene, context):
    items = []
    #Processing to add items to items...
    for j,name in enumerate(Types):
        #items.append(('OP{}'.format(j), name, ""))
        items.append((name+"_shape", name, ""))
    return items

class MyProperties(bpy.types.PropertyGroup):


    objFile_path: bpy.props.StringProperty(
        name = "File name",
        description="Choose a directory:",
        default="",
        maxlen=1024,
        update = lambda s,c: make_path_absolute('objFile_path'),
        subtype='FILE_PATH'
        )
        
        
    file_path: bpy.props.StringProperty(
        name = "Pos(xyz)",
        description="Choose a directory:",
        default="",
        maxlen=1024,
        update = lambda s,c: make_path_absolute('file_path'),
        subtype='FILE_PATH'
        )
        
    mesh_path: bpy.props.StringProperty(
        name = "Compartments",
        description="Choose a directory:",
        default="",
        maxlen=1024,
        update = lambda s,c: make_path_absolute('mesh_path'),
        subtype='FILE_PATH'
        )
        
    props_path: bpy.props.StringProperty(
        name = "Properties",
        description="Choose a directory:",
        default="",
        maxlen=1024,
        update = lambda s,c: make_path_absolute('props_path'),
        subtype='FILE_PATH'
        )
        
    box_path: bpy.props.StringProperty(
        name = "Box",
        description="Choose a directory:",
        default="",
        maxlen=1024,
        update = lambda s,c: make_path_absolute('box_path'),
        subtype='FILE_PATH'
        )
                
    system_scale : bpy.props.FloatProperty(
        name = "system_scale",
        default=1.0,
        )

    mesh_scale : bpy.props.FloatProperty(
        name = "mesh_scale",
        default=1.0,
        )
        
    box_lengths : bpy.props.FloatVectorProperty(
        name = "box_lengths",
        default=(1.0, 1.0, 1.0),
        )

    N : bpy.props.IntProperty(
        name = "N",
        default=0,
        )

    NFrames : bpy.props.IntProperty(
        name = "NFrames",
        default=0,
        )
        
    types_enum : bpy.props.EnumProperty(
        name = "types_enum",
        description = "Select an option",
        items = get_object_list_callback
        )
        
    console_closed : bpy.props.BoolProperty(
        name = "console_closed",
        default=True,
        )
    step : bpy.props.IntProperty(
        name = "step",
        default=1,
        )
                       
        
# ------------------------------------------------------------------------
#    Operators
# ------------------------------------------------------------------------

class XYZ_OT_Export(bpy.types.Operator):
    bl_label = "export to obj"
    bl_idname = "xyz.export_to_obj"

    def execute(self, context):
        
        scene = bpy.context.scene
        mytool = scene.my_tool
        
        export_obj(mytool.objFile_path)
        
        return {'FINISHED'}
    

class XYZ_OT_InitPyRID(bpy.types.Operator):
    bl_label = "Init PyRID"
    bl_idname = "xyz.init_pyrid"

    def execute(self, context):
        
        scene = bpy.context.scene
        mytool = scene.my_tool
        
        if mytool.console_closed and platform == 'win32':
            bpy.ops.wm.console_toggle()
            mytool.console_closed = False
            
        Viewport_Settings()
        bpy.context.space_data.shading.type = 'RENDERED'
        
        return {'FINISHED'}
    
          
class XYZ_OT_Print(bpy.types.Operator):
    bl_label = "Print Parameters"
    bl_idname = "xyz.print"

    def execute(self, context):
        
        global molecule, start, end
        
        scene = bpy.context.scene
        mytool = scene.my_tool

        # print the values to the console
        print("file path:", mytool.file_path)
        print("mesh path:", mytool.mesh_path)
        print("system_scale:", mytool.system_scale)
        print("mesh_scale:", mytool.mesh_scale)
        print('box_lengths:', mytool.box_lengths[:])
        print('Radii:', Radii)
        print('Radii_cutoff:', Radii_cutoff)
        print('Compartments:', Mesh_Names)
        print('Console state:',mytool.console_closed)
        #print('len(molecules):', len(molecule))
        #print('start:', start)
        #print('end:', end)
        return {'FINISHED'}

class XYZ_OT_ViewPortLight(bpy.types.Operator):
    bl_label = "Change viewport setting"
    bl_idname = "xyz.viewport_light"

    def execute(self, context):
        # change viewport settings
        #Viewport_Settings()
        bpy.context.scene.world.color = (0.65, 0.85, 0.86)
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0.65, 0.85, 0.86, 1)
        return {'FINISHED'}

class XYZ_OT_ViewPortDark(bpy.types.Operator):
    bl_label = "Change viewport setting"
    bl_idname = "xyz.viewport_dark"

    def execute(self, context):
        # change viewport settings
        bpy.context.scene.world.color = (0.050876, 0.050876, 0.050876)
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0.050876, 0.050876, 0.050876, 1)
        #bpy.ops.preferences.reset_default_theme()
        return {'FINISHED'}


class XYZ_OT_Clear(bpy.types.Operator):
    bl_label = "Clear System"
    bl_idname = "xyz.clear"

    def execute(self, context):
        # clear system
        global scene_gen
        scene_gen = False
        clear_system()
        return {'FINISHED'}

class XYZ_OT_ImportMesh(bpy.types.Operator):
    bl_label = "Import compartments"
    bl_idname = "xyz.importmesh"

    def execute(self, context):
        #filepath = bpy.data.filepath
        #directory = os.path.dirname(filepath)
        # import mesh
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.import_scene.obj(filepath=bpy.context.scene.my_tool.mesh_path, axis_forward='Y')
        
        global Mesh_Names
        Mesh_Names = []
        for obj in bpy.context.selected_objects:
            Mesh_Names.append(obj.name)
            
        if "Box" in Mesh_Names:
            bpy.data.objects["Box"].hide_render = True
            bpy.data.objects["Box"].hide_viewport = True
            
        return {'FINISHED'}    


class XYZ_OT_LoadXYZ(bpy.types.Operator):
    bl_label = "Import molecule positions"
    bl_idname = "xyz.loadpos"

    def execute(self, context):
        # Import molecule positions
        load_mol_trace()
        return {'FINISHED'}    

class XYZ_OT_GenerateScene(bpy.types.Operator):
    bl_label = "Generate Scene"
    bl_idname = "xyz.genscene"

    def execute(self, context):
        # Import molecule positions
        generate_scene()
        return {'FINISHED'} 
            

class XYZ_OT_Start(bpy.types.Operator):
    bl_label = "Start"
    bl_idname = "xyz.start"

    def execute(self, context):
        bpy.context.scene['index']=0
        # Import molecule positions
        bpy.app.handlers.frame_change_pre.append(frame_change_handler)
        return {'FINISHED'} 
                
class XYZ_OT_Run(bpy.types.Operator):
    bl_label = "Run"
    bl_idname = "xyz.run"

    def execute(self, context):
        
        scene = bpy.context.scene
        mytool = scene.my_tool
        
        # init
        if mytool.console_closed and platform == 'win32':
            bpy.ops.wm.console_toggle()
            mytool.console_closed = False
            
        Viewport_Settings()
        bpy.context.space_data.shading.type = 'RENDERED'
        

        # Import mesh compartments
        
        if bpy.context.scene.my_tool.mesh_path!='':
            try:
                bpy.ops.object.select_all(action='DESELECT')
                bpy.ops.import_scene.obj(filepath=bpy.context.scene.my_tool.mesh_path, axis_forward='Y')
                
                global Mesh_Names
                Mesh_Names = []
                for obj in bpy.context.selected_objects:
                    Mesh_Names.append(obj.name)
                    
                if "Box" in Mesh_Names:
                    bpy.data.objects["Box"].hide_render = True
                    bpy.data.objects["Box"].hide_viewport = True
            except:
                print('Compartment mesh not found!')
        
        
        # import molecule position data
        load_mol_trace()
        
        # apply scene properties
        generate_scene()
        
        # start
        bpy.context.scene['index']=0
        bpy.app.handlers.frame_change_pre.append(frame_change_handler)
        
        bpy.context.scene.my_tool.types_enum = Types[0]+'_shape'
        
        return {'FINISHED'} 
    
    ######
    
class XYZ_OT_ReloadPos(bpy.types.Operator):
    bl_label = "Reload"
    bl_idname = "xyz.reload"

    def execute(self, context):
        
        scene = bpy.context.scene
        mytool = scene.my_tool
        
        if mytool.console_closed and platform == 'win32':
            bpy.ops.wm.console_toggle()
            mytool.console_closed = False
        
        # import molecule position data
        load_mol_trace()

        # Prepare particle system
        n = []
        numbers = []
        for j,name in enumerate(Types):

            #degp = bpy.context.evaluated_depsgraph_get()
            
            n.append(sum(Type_Masks[name]))
            numbers.append(sum(Type_Masks[name]))


        global scene_gen
        scene_gen = True
        
        # start
        bpy.context.scene['index']=0
        bpy.app.handlers.frame_change_pre.append(frame_change_handler)
        
        bpy.context.scene.my_tool.types_enum = Types[0]+'_shape'
        
        return {'FINISHED'} 
    

class XYZ_OT_RenderAnimation(bpy.types.Operator):
    bl_label = "Render animation"
    bl_idname = "xyz.renderanimation"

    def execute(self, context):
        # Render scene
        directory = os.path.dirname(bpy.context.scene.render.filepath)
        filename = os.path.basename(bpy.context.scene.render.filepath)
        format = bpy.context.scene.render.ffmpeg.format
        
        if format not in filename:
            bpy.context.scene.render.filepath = os.path.join(directory, filename+'.'+format)
        else:
            bpy.context.scene.render.filepath = os.path.join(directory, filename)
            
        bpy.ops.render.render(animation=True, use_viewport=True)
        return {'FINISHED'} 


class XYZ_OT_Render(bpy.types.Operator):
    bl_label = "Render image"
    bl_idname = "xyz.render"

    def execute(self, context):
        # Render scene
        bpy.ops.render.render(animation=False, use_viewport=True)
        return {'FINISHED'} 
    
                    
class XYZ_OT_ViewRender(bpy.types.Operator):
    bl_label = "View Render"
    bl_idname = "xyz.viewrender"

    def execute(self, context):
        # Import molecule positions
        view_render()
        return {'FINISHED'} 

class XYZ_OT_ViewAnimation(bpy.types.Operator):
    bl_label = "View Animation"
    bl_idname = "xyz.viewanimation"

    def execute(self, context):
        # Import molecule positions
        view_animation()
        return {'FINISHED'} 

class XYZ_OT_RotateCamera(bpy.types.Operator):
    bl_label = "Rotate Camera"
    bl_idname = "xyz.rotcamera"

    def execute(self, context):
        # Import molecule positions
        rotate_camera()
        return {'FINISHED'} 
    
class XYZ_OT_FixCamera(bpy.types.Operator):
    bl_label = "Fix Camera"
    bl_idname = "xyz.fixcamera"

    def execute(self, context):
        # Import molecule positions
        fix_camera()
        return {'FINISHED'} 

class XYZ_OT_AddLights(bpy.types.Operator):
    bl_label = "Enhanced Rendering"
    bl_idname = "xyz.addlights"

    def execute(self, context):
        # Import molecule positions
        add_lights()
        bpy.context.scene.eevee.taa_render_samples = 16
        bpy.context.scene.cycles.samples = 16
        bpy.context.scene.cycles.denoising_prefilter = 'FAST'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.render.fps = 10

        return {'FINISHED'} 
                               
# ------------------------------------------------------------------------
#    Menus
# ------------------------------------------------------------------------


class OBJECT_MT_ViewPort(bpy.types.Menu):
    bl_label = "Select"
    bl_idname = "OBJECT_MT_view_port"

    def draw(self, context):
        layout = self.layout

        # Built-in operators
        layout.operator("xyz.viewport_light", text = 'light style')
        layout.operator("xyz.viewport_dark", text = 'dark style')


class OBJECT_MT_Radii(bpy.types.Menu):
    bl_label = "Select"
    bl_idname = "OBJECT_MT_Radii"

    def draw(self, context):
        layout = self.layout
        scene = bpy.context.scene
        mytool = scene.my_tool
        
        layout.prop(mytool, "types_enum")
        
        
# ------------------------------------------------------------------------
#    Panels
# ------------------------------------------------------------------------

class PanelMain(bpy.types.Panel):
    
    bl_label = "PyRID visualization"
    bl_idname = "XYZ_PT_PanelMain"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PyRID"
    
    def draw(self, context):
        layout = self.layout
        
        row = layout.row()
        #row.label(text = "file path", icon = 'MATCUBE')
        #row.operator("")
        #layout.operator("xyz.init_pyrid")
        layout.operator("xyz.clear")
        
class PanelA(bpy.types.Panel):
    
    bl_label = "Files Paths"
    bl_idname = "XYZ_PT_PanelA"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "XYZ"
    bl_parent_id = "XYZ_PT_PanelMain"
    bl_options = {'HEADER_LAYOUT_EXPAND'}
    
    def draw(self, context):
        layout = self.layout
        scene = bpy.context.scene
        mytool = scene.my_tool
        
        row = layout.row()
        layout.prop(mytool, "file_path")
        layout.prop(mytool, "mesh_path")
        layout.prop(mytool, "props_path")
        layout.prop(mytool, "box_path")
        
class PanelB(bpy.types.Panel):
    
    bl_label = "Properties"
    bl_idname = "XYZ_PT_PanelB"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "XYZ"
    bl_parent_id = "XYZ_PT_PanelMain"
    bl_options = {'HEADER_LAYOUT_EXPAND'}
    
    def draw(self, context):
        layout = self.layout
        scene = bpy.context.scene
        mytool = scene.my_tool
        
        layout.label(text = 'System')
        box = layout.box()
        #row = layout.row()
        column = box.column()
        #row.operator("")
        #layout.operator("xyz.viewport")   
        column.menu(OBJECT_MT_ViewPort.bl_idname, text="Style", icon="SCENE")  
        column.prop(mytool, "system_scale")
        column.prop(mytool, "mesh_scale")
        column.prop(mytool, "step")
        column.prop(mytool, "box_lengths", text="box_lengths")
        
        
        layout.label(text = 'Particles')
        box = layout.box()
        column = box.column()
        column.menu(OBJECT_MT_Radii.bl_idname, text="Select a particle", icon="SCENE") 
        if scene_gen:
            column.label(text = mytool.types_enum)
            if mytool.types_enum!='':
                column.prop(bpy.data.objects[mytool.types_enum], "dimensions", text = '')
                column.prop(bpy.data.objects[mytool.types_enum].active_material, "diffuse_color", text = '')
            
        layout.label(text = 'Compartments')
        box = layout.box()
        column = box.column()
        for Mesh_Name in Mesh_Names:
            column.prop(bpy.data.objects[Mesh_Name].active_material, "diffuse_color", text = Mesh_Name)
            
        layout.operator("xyz.print")
        
class PanelC(bpy.types.Panel):
    
    bl_label = "Run"
    bl_idname = "XYZ_PT_PanelC"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "XYZ"
    bl_parent_id = "XYZ_PT_PanelMain"
    bl_options = {'HEADER_LAYOUT_EXPAND'}
    
    def draw(self, context):
        layout = self.layout
        
        row = layout.row()
        row.operator("xyz.run")
        #layout.operator("xyz.importmesh")    
        #layout.operator("xyz.loadpos")
        #layout.operator("xyz.genscene")
        #layout.operator("xyz.start")
        row.operator("xyz.reload")
        
        layout.prop(bpy.context.scene.render, "filepath", text="")

        row = layout.row() 
        row.operator("xyz.rotcamera")
        row.operator("xyz.fixcamera")
        row.operator("xyz.addlights")
        row = layout.row() 
        
        row = layout.row() 
        row.operator("xyz.render")
        row.operator("xyz.renderanimation")
        row.operator("xyz.viewrender")
        row.operator("xyz.viewanimation")
        row = layout.row() 
        
#--------------------------------------


class PanelMain_2(bpy.types.Panel):
    
    bl_label = "PyRID export obj"
    bl_idname = "XYZ_PT_PanelMain_2"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PyRID"
    
    def draw(self, context):
        layout = self.layout
        
        scene = bpy.context.scene
        mytool = scene.my_tool
        
        row = layout.row()
        #row.label(text = "file path", icon = 'MATCUBE')
        #row.operator("")
        layout.prop(mytool, "objFile_path")
        layout.operator("xyz.export_to_obj")
        

# ------------------------------------------------------------------------
#    Registration
# ------------------------------------------------------------------------

classes = (
    MyProperties,
    XYZ_OT_Export, PanelMain_2, XYZ_OT_InitPyRID, XYZ_OT_Print, XYZ_OT_ViewPortLight, XYZ_OT_ViewPortDark, XYZ_OT_Clear, XYZ_OT_ImportMesh, XYZ_OT_LoadXYZ, XYZ_OT_GenerateScene, XYZ_OT_Start, XYZ_OT_Run, XYZ_OT_ReloadPos, XYZ_OT_Render, XYZ_OT_RenderAnimation, XYZ_OT_ViewRender, XYZ_OT_ViewAnimation, XYZ_OT_RotateCamera, XYZ_OT_FixCamera, XYZ_OT_AddLights, PanelMain,
    PanelA, PanelB, PanelC, OBJECT_MT_ViewPort, OBJECT_MT_Radii
)
      
def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.my_tool = bpy.props.PointerProperty(type=MyProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
        
    del bpy.types.Scene.my_tool
    
if __name__ == '__main__':
    register()