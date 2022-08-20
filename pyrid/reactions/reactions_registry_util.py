# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""


import numpy as np
import numba as nb
from numba.experimental import jitclass
# from ..math.transform_util import cantor_pairing, unique_pairing
from ..data_structures.dynamic_array_util import DenseArrayReact, HolesArrayReact

#%%

@nb.njit
def add_column(array, dtype):
    
    """Adds a column to a given numpy array.
    
    Parameters
    ----------
    array : `array_like`
        Numpy array to which to add a column
    
    
    Returns
    -------
    array_like
        numpy array with an extra columns
    
    """
    
    array = np.append(array, np.zeros(1, dtype = dtype))
    
    return array

#%%

key_head = nb.int64
value_head = nb.int64[:]

item_t_react = np.dtype([('educts_index', (np.int64, (2,))), ('id', np.int64), ('unique_educts_id', np.float64)],  align=True)

item_t_paths = np.dtype([('rate', np.float64), ('radius', np.float64), ('products_ids', (np.int64, 1000)), ('products_loc', (np.int64, 1000)), ('products_direction', (np.int64, 1000)), ('n_products', np.int64), ('type_id', np.int64), ('type', 'U20'), ('n_success', np.int64), ('n_success_binned', np.int64),('placement_factor', np.float64),],  align=True)
       
spec_darray = [
    ('n', nb.int64),
    ('n_success', nb.int64),
    ('n_total', nb.int64),
    ('n_total_binned', nb.int64),
    ('capacity', nb.int64),
    ('Data', nb.typeof(np.empty(1, dtype = item_t_react))),
    ('item_t', nb.typeof(item_t_react)),
    ('index', HolesArrayReact.class_type.instance_type),
    ('slot', nb.int64),
    ('dummy', nb.typeof(np.empty(1, dtype = item_t_react))),
    ('head', nb.types.DictType(key_head, value_head)),
]

spec_reactions = [
    ('educts', nb.int64[:]),
    ('particle_educts', nb.int64[:]),
    #('products', nb.int64[:]),
    ('rate', nb.float64),
    ('radius', nb.float64),  
    ('current_index', nb.int64),
    ('reaction_educt_type', nb.types.string),
    # ('reaction_type_id', nb.int64),
    ('paths', nb.typeof(np.empty(1, dtype = item_t_paths))),
    ('n_paths', nb.int64),
    ('bimol', nb.types.boolean),
    ('particle_reaction', nb.types.boolean),
    
]

@jitclass(spec_darray+spec_reactions)
class Reaction(DenseArrayReact):
    
    """
    The reaction class stores all particle and molecule reactions of the current time step. At its core the Reaction class is a dense dynamic array (i.e. elements are deleted via a pop and swap mechanism). It also keeps another dynamic array with holes that keeps track of where a reaction moves during swaping. This is necessary, beacause we also store each reaction by its id in a doubly linked list that contains for each particle all its reactions. This doubly linked list is needed whenever we delete all reactions of a certain particle, in which case we need to be able to recover the position of a reaction id in the dynamic array. Also, the elements of the doubly linked list have shape (2,2), each for the next and prev pointer. The first dimension has two elements since a reaction can be assigned to at max two educts (for bimoelcular reactions). Therefore, these two integers point to the next (previous) reaction id belonging to each of the educts. The two elements of the second dimension are required, beacuse we need to be able to track the position of particles i and j in the linked list, which can in principle switch between two positions (to which reaction id should the next pointer go? There are always two possibilities). Thereby, the other two ints store, whether the next (prev) pointer target element 0 or 1.
    
    Attributes
    ----------
    educts : `nb.int32[:]`
        Educt type ids.
    products : `nb.int64[:]`
        Product type ids.
    rate : `nb.float64`
        Reaction rate.
    radius : `nb.float64`
        Reaction radius.
    current_index : `nb.uint64`
        Current/last assigned slot of dynamic array (main -> dense_array_react())
    reaction_type : `nb.types.string` {'bind', 'enzymatic', 'conversion', 'decay', 'enzymatic_rb', 'conversion_rb', 'fusion'}
        Name of reaction type
    reaction_type_id : `nb.int64 {0,1,2,3,4,5,6,7}`
        ID of reaction type
    paths : `array_like`
        Reaction path.
        
        `dtype = np.dtype([('rate', np.float64), ('products', (np.int64, (100,3))), ('n_products', np.int64),],  align=True)`
    n_paths : `nb.int64`
        Number of reaction paths defined for this reaction.
    
    Methods
    -------
    add_path(products, rate)
        Adds a reaction path to the reaction.
    re_initialize()
        Reinitializes the dense dynamic array saving all reactions occuring within a timestep, reseting the array size and deleting all linked lists containg reaction data.
    append_reaction(i,j, Particle_Reaction)
        Appends a new reaction to the reactions list.
    delete_reaction(k, Particle_Reaction)
        Deletes reaction at index k.
    delete_reaction_all(i)
        Deletes all reactions of Particle i assigned for the current time step.
    get_random()
        Returns a random reaction id.
    getit(k)
        Returns reaction data inserted at index k.
    setit(k, value)
        Assigns anew value to reaction at index k.
    reactions_left()
        Prints out all reactions left in list.
    reactions_left_pointer(Particles, i)
        Prints out all reactions left for particle i.
    
    Notes
    -----
    The data structure we need to organize the reactions is a little bit more complex than a simple dense dynamic array or one with holes as is used to keep track of all the rigid body molecules and particles in the sytsem. The Reaction class is a combination of different dynamic arrays and a hash table. Let  me motivate this: Our data structure needs to be able to do 4 things as efficient as possible:
        
        1. Add reactions,
        2. Delete single reactions,
        3. Delete all reactions certain particle participates in,
        4. Return a random reaction from the list.
        
    We need to be able to delete a single reaction whenever this reaction was not successful.
    We need to delete all reactions of a particle whenever a reaction was successful, because this particle is no longer available since it either got deleted or changed its type (except in the case where the particle participates as an enzyme).
    We need to be able to request a random reaction from the list, because processing the reactions in the order they occur in the list will introduce a bias: Reactions of particles with a low index are added to the list first, at least in the beginning of a simulation when the indices of particles have not yet been swapped around (in case there are any decay reactions), beacuse particle distances are evaluated in the order particles occur in the corresponding list. Thereby particle 1 will always have a higher chance of having a successfull reactions when competing with other particles.
    Points 1. an 2. could be easily established with a simple dynamic array. However, point 3 is a bit more complicated but can be solved with a doubly linked list, embedded into a dynamic array with holes, that connects all reaction of a particle. We need a doubly linked list, because we need to be able to delete a single reaction (index k) fom the linked list (point 2). As such, we need to be able to reconnect the linked list's 'chain'. Therefore, we need to know the element in the linked list that pointed to k (prev) in addition to the element k points to (next). Another problem that needs to be solved is that a reaction can be linked to at max. 2 educts. Therefore, each next and prev pointer needs to be 4-dimensional. We need 1 integer for each educt to save the next (prev) reaction index and another integer {0,1} to keep track of whether in the next (prev) reaction the particle is the first or the second educt since this may change from reaction to reaction!
    Since the dynamic array, the doubly linked list is embedded in, has holes, picking a random reaction from the list becomes another issue. This can, however, easily be solved by adding another dynamic array (dense array), which keeps all the reactions that are left in a tightly packed format. Picking a random reaction is that as easy as drawing a uniformly distributed random interger between 0 and n, where n is the length of the dense array.
    A very nice introduction/overview to the kind of data structures used here has been written by Niklas Gray (see `<https://ourmachinery.com/post/data-structures-part-1-bulk-data/>`_)
    
    
    .. image:: ../../../Graphics/Reactions_DynamicArray.png
        :width: 60%
    
    |
    
    The figure depicts the data structure we use to keep track of the reactions.
    
    |
    
    """
    
    __init__DenseArrayReact = DenseArrayReact.__init__
    
    def __init__(self, educts, reaction_type, reaction_educt_type, particle_educts = None, radius = None):
        
        self.__init__DenseArrayReact(item_t_react)
        
        
        self.rate = 0.0
        
        self.reaction_educt_type = reaction_educt_type
        
        if reaction_type == 'bimolecular':
            self.bimol = True
            if radius is not None:
                self.radius = radius
            
            self.particle_reaction = True
            
        elif reaction_type == 'unimolecular':
            self.bimol = False
            
            if reaction_educt_type == 'Particle':
                self.particle_reaction = True
            elif reaction_educt_type == 'Molecule':
                self.particle_reaction = False
                
        self.educts = educts
        
        # In case we have a bimolecular reaction we may also want to save the actual particle educts involved, not just the corresponding molecules (educts):
        if particle_educts is not None:
            self.particle_educts = particle_educts
            
        # self.reaction_type_id = reaction_type_id
        self.n_success = 0
        self.n_total = 0
        self.n_total_binned = 0
        
        self.n_paths = 0
        self.paths = np.zeros(1, dtype = item_t_paths)
        
    
    def add_path(self, System, path_type, rate, products_ids = None, product_surf_vol = None, product_direction = None, radius = None, placement_factor = 0.5):
        
        """Adds a new reaction path to the reaction class instance.
        
        Parameters
        ----------
        System : `object`
            Instance of System class
        path_type : `string`
            Type of reaction
        rate : `float64`
            Reaction rate       
        products_ids : `int64[:]`
            List of product ids. Default = None 
        product_surf_vol : `int64[:]`
            List indicating whether a product is a volume (0) or a surface (1) molecule. 
        product_direction : `int64[:]`. Default = None
            In case of a surface moelcule having a volume moelcule product indicates whether the product is to be released into the simulation box volume (0) or the mesh compartment (1). Default = None
        radius : `float64`
            Reaction radius in case of a bimolecular/biparticle reaction. Default = None
            
        Raises
        ------
        ValueError('Number of reaction products must not be larger than 1000!')
            PyRID limits the number of product moelcules to 1000, otherwise a value error is raised. The user may rayise this limit at any time by changing the size of the structured array of dtype item_t_paths.
        ValueError('Reaction type unknown!')
            If a reaction type is passed that has not yet been implemented, a value error is raised.
        
        """
        
        if products_ids is not None:
            if len(products_ids)>1000:
                raise ValueError('Number of reaction products must not be larger than 1000!')
            
        if path_type in ['bind', 'fusion', 'enzymatic', 'enzymatic_mol', 'absorption']:
            self.bimol = True
        elif path_type in ['conversion', 'conversion_mol', 'decay', 'production', 'fission', 'release']:
            self.bimol = False
        else:
            raise ValueError('Reaction type unknown!')
            
            
        i = self.n_paths
        
        if self.n_paths > 0:
            self.paths = add_column(self.paths, item_t_paths)
            

        if radius is not None:
            self.paths[i]['radius'] = radius # This is the radius, which is used, e.g. when resolving fission or production reactions.
            
        if path_type == 'fusion':
            # TODO: It might make sense to calculate the placement_factor from the diffusion coefficients of the educt molecules. I havent put much thought in this, but it would make sense that a product is placed closer to the heavier molecule educt. Why is this never implenented this way (MCell, ReaDDy ...)? This should then also be considered for the placement in fission reactions.
            self.paths[i]['placement_factor'] = placement_factor
        
        self.paths[i]['rate'] = rate
        self.paths[i]['type'] = path_type
        self.paths[i]['type_id'] = System.react_type_id[str(path_type)]
        
        if products_ids is not None:
            self.paths[i]['n_products'] = len(products_ids)
            self.paths[i]['products_ids'][0:len(products_ids)] = products_ids # product type id
            if product_surf_vol is not None:
                
                self.paths[i]['products_loc'][0:len(products_ids)] = product_surf_vol # product surface (1) or volume (0) molecule
            if product_direction is not None:
                self.paths[i]['products_direction'][0:len(products_ids)] = product_direction # if the educt is a surface molecule but the product is a volume molecule, we need to define one whioch side the product should be released (0: outside (System), 1: inside (Mesh Compartment))
        else:
            self.paths[i]['n_products'] = 0
        
        
        
        # update the total number of paths:
        self.n_paths += 1
        # update the total reaction rate:
        self.rate += rate
        
        
        #%%
        
    def re_initialize(self):
        self.__init__DenseArrayReact(item_t_react)
        
    def append_reaction(self, i,j = None):

        """Appends a new reaction to the current reactions list.
        
        Parameters
        ----------
        i : `int64`
            Index of educt 1
        j : `int64`
            Index of educt 2. Default = None
        
        """
        
        
        if self.n == self.capacity:
            # Double capacity if not enough room
            self._resize(2 * self.capacity)      
    
        
        # The new reaction gets an id assigned. This id will be selected from the available ids (holes) of the index array:
        self.current_index = self.index.allocate_slot()
        # print(self.current_index+1)
        # print(self.n)
        
        current_indexP1 = self.current_index
        self.index[current_indexP1]['id'] = self.n
        
        self.Data[self.n]['id'] = self.current_index
        self.Data[self.n]['educts_index'][0] = i
        # self.Data[self.n]['unique_educts_id'] = Unique_Pairing(i,j)

        # if i%2 == 0:
        #     i0 = i
        #     i = j
        #     j = i0
        
        first_reaction_idx = 0
        first_educt_idx = 0
        second_reaction_idx = 0
        second_educt_idx = 0
        
        # If this is not the first reaction of i:
        if i in self.head:
            # The reaction index of the head:
            first_reaction_idx = self.head[i][0]
            first_educt_idx = self.head[i][1]
            # The reaction index the head (first) points to:
            second_reaction_idx = self.index[first_reaction_idx]['next_reaction'][first_educt_idx][0]
            second_educt_idx = self.index[first_reaction_idx]['next_reaction'][first_educt_idx][1]
            # The current_index is inserted between the first and the second 
            # This makes keeping the linked list circular strait forward:
            self.index[current_indexP1]['next_reaction'][0][0] = second_reaction_idx
            self.index[current_indexP1]['prev_reaction'][0][0] = first_reaction_idx
            self.index[first_reaction_idx]['next_reaction'][first_educt_idx][0] = self.current_index
            self.index[second_reaction_idx]['prev_reaction'][second_educt_idx][0] = self.current_index
            # insert educt_idx
            self.index[current_indexP1]['next_reaction'][0][1] = second_educt_idx
            self.index[current_indexP1]['prev_reaction'][0][1] = first_educt_idx
            self.index[first_reaction_idx]['next_reaction'][first_educt_idx][1] = 0
            self.index[second_reaction_idx]['prev_reaction'][second_educt_idx][1] = 0
        else:
            # the reaction will simply point to itself:
            self.index[current_indexP1]['next_reaction'][0][0] = self.current_index
            self.index[current_indexP1]['prev_reaction'][0][0] = self.current_index
            self.index[current_indexP1]['next_reaction'][0][1] = 0
            self.index[current_indexP1]['prev_reaction'][0][1] = 0
            # set this to be the first reaction (head)
            self.head[i] = np.array([self.current_index, 0, 0], dtype=np.int64)
            
        self.head[i][2] += 1

        
        if self.bimol == True and j is not None:
            
            self.Data[self.n]['educts_index'][1] = j
            
            # Repeat the same for j:
            # If this is not the first reaction of j:
            if j in self.head:
                # The reaction index of the head:
                first_reaction_idx = self.head[j][0]
                first_educt_idx = self.head[j][1]
                # The reaction index the head (first) points to:
                second_reaction_idx = self.index[first_reaction_idx]['next_reaction'][first_educt_idx][0]
                second_educt_idx = self.index[first_reaction_idx]['next_reaction'][first_educt_idx][1]
                # The current_index is inserted between the first and the second 
                # This makes keeping the linked list circular strait forward:
                self.index[current_indexP1]['next_reaction'][1][0] = second_reaction_idx
                self.index[current_indexP1]['prev_reaction'][1][0] = first_reaction_idx
                self.index[first_reaction_idx]['next_reaction'][first_educt_idx][0] = self.current_index
                self.index[second_reaction_idx]['prev_reaction'][second_educt_idx][0] = self.current_index
                # insert educt_idx
                self.index[current_indexP1]['next_reaction'][1][1] = second_educt_idx
                self.index[current_indexP1]['prev_reaction'][1][1] = first_educt_idx
                self.index[first_reaction_idx]['next_reaction'][first_educt_idx][1] = 1
                self.index[second_reaction_idx]['prev_reaction'][second_educt_idx][1] = 1     
            else:
                # the reaction will simply point to itself:
                self.index[current_indexP1]['next_reaction'][1][0] = self.current_index
                self.index[current_indexP1]['prev_reaction'][1][0] = self.current_index
                self.index[current_indexP1]['next_reaction'][1][1] = 1
                self.index[current_indexP1]['prev_reaction'][1][1] = 1
                # set this to be the first reaction (head)
                self.head[j] = np.array([self.current_index, 1, 0], dtype=np.int64)
            
            self.head[j][2] += 1
        
        # Since we are appending an element, the size of our array needs to increase by 1
        self.n += 1
        
    def delete_reaction(self, k):
        
        """
        Deletes the reaction from the list by its index.
        
        Parameters
        ----------
        k : `int64`
            Index of the reaction which to delete
        
        
        Notes
        -----
        Deletes an element from the array by a pop and swap mechanism.
        This is a dense dynamic array. As such, elements get swapped around by a pop and swap mechanism. Element k, which is trageted for deletion may no longer be at the place where it was originally inserted. Therefore, k does not refer to the element currently at index k in the dense array but needs to be looked up in the index array (self.index), which is an array with holes, preserving the location of its elements.
        Element k also needs to be deleted from the index array. The index array is a dynamic array with holes. Thereby, an element is removed from the index array by creating a hole that can be reused for the insertion of new elements. The index array also holds a doubly free linked list containing all reactions belonging to a certain particle. Therefore, the doubly free linked list is also updated.
        
        """
        
        if self.bimol == True:
            number_educts = 2
        else:
            number_educts = 1
        
        # We need to loop over both educts (in case of bimolecular reaction) and update the free linked list
        # of each educt, linking all reactions of the respective particle:
        for j in range(number_educts):
            # Get the reaction_id and educt_no of the next pointer
            # We need to assign these to the previous next pointer
            next_reaction_id = self.index[k]['next_reaction'][j][0]
            next_educt_no = self.index[k]['next_reaction'][j][1]
            # Therefore, also get the reaction_id and educt_no of the prev pointer
            prev_reaction_id = self.index[k]['prev_reaction'][j][0]
            prev_educt_no = self.index[k]['prev_reaction'][j][1]
            
            
            ip = self.Data[self.index[k]['id']]['educts_index'][j]
            
            # Check which reaction the head currently points to.
            first_reaction_idx = self.head[ip][0]
            # If the head happens to point to index k, which we just deleted, we need to set
            # the head to the reaction idx k pointed to:
            if first_reaction_idx == k:
                # However, only if reaction k did not point to itself (However, this should never happen anyway, right!?)! 
                if next_reaction_id != k:
                    self.head[ip][0] = next_reaction_id
                    self.head[ip][1] = next_educt_no
                    
                # else:
                    # print('completetly deleted from list')
            
            # Since we deleted k, the reaction that pointed to k (prev_reaction_id) now needs to skip k and instead point to the reaction the k pointed to.
            self.index[prev_reaction_id]['next_reaction'][prev_educt_no][0] = next_reaction_id
            self.index[prev_reaction_id]['next_reaction'][prev_educt_no][1] = next_educt_no
            # Similarly, k is no longer the previous reaction of the reaction that k pointed to. 
            # So, we also need to to set the previous pointer of the next reaction to that of k:
            self.index[next_reaction_id]['prev_reaction'][next_educt_no][0] = prev_reaction_id
            self.index[next_reaction_id]['prev_reaction'][next_educt_no][1] = prev_educt_no    
            
            # At last, we decrease the number of reaction particle i partcipates in by 1:
            self.head[ip][2] -= 1
        
        # Delete the reaction itself:
        # Get the id (id_swap) of the last item in the reactions list and swap it with k:
        id_swap = self.Data[self.n-1]['id']
        self.Data[self.index[k]['id']] = self.Data[id_swap]
        
        # Add to the freelist, which is stored in slot 0. !!!This is useless, as we do never
        # add any new reactions after a reaction has been deleted. We first add all reaction to the list. Then we evaluate all reactions until none is left!
        self.index[k]['next_hole'] = self.index.Data[0]['next_hole']
        self.index.Data[0]['next_hole'] = k    # !!!Why not k+1 ? -> because the values in 'next_hole' refer to the actual reaction index in the dense array. In the holes array, 0 is the head, which now points to reaction index k. Reaction index k is now available. When allcoating a 'slot', this only refers to getting an reaction id not yet taken. So maybe we should change the term 'slot here!? We could also just keep on incerasing the reaction id, for int64 we shouldnt get into trouble very soon! 
            
        self.index[id_swap]['id'] = self.index[k]['id']  
        
        self.n -= 1
        
    def delete_reaction_all(self, i):
        """
        Delete all reactions educt i (molecule or particle) is involved in.
        
        Parameters
        ----------
        i : `int64`
            Index of the educt
        
        """
        
        count = 0
        # Check if there even exist any reactions for particle i:
        if i in self.head:
            # Go to the head of the reaction linked list:
            head = self.head[i]
            # continue until head points to itself:
            while self.head[i][2]: # Equivalent to while self.head[i][2] > 0:
                self.delete_reaction(head[0])
                count += 1
                head = self.head[i]

        return count
                
                
    def get_random(self):
        """
        Return a random reaction from the list.
        
        Returns
        -------
        int64
            Index of the random reaction.
        
        """
        
        idx = np.random.randint(self.n)
        # print('idx: ', idx)
        
        # sorted_index = np.argsort(self.Data[0:self.n]['unique_educts_id'])
        # return self.Data[0:self.n][sorted_index]['id'][idx] 
    
        return self.Data[idx]['id'] # self.Data['id'][idx] # 
        
            
    def get(self, k):
        """Returns the data array of the reaction at index k, which contains the educt indices.
        
        
        Parameters
        ----------
        k : `int64`
            Index of the reaction whose educt indices to return
        
        Notes
        -----
        Return element at index k+1 (element 0 is reserved for the head of the free list).
        Any item will be returned, also holes, because checking whether an item is a hole at each access operation is too expensive.
        Also, there will be no test, whether the index is within bounds, as this can also reduce speed by up to 25%.
        To get a list of all items which are not holes, use the occupied attribute!
        
        Returns
        -------
        array_like
            Array containing the educt indices and other information.
        
        """
          
        return self.Data[self.index[k]['id']]


    def set(self, k, value):
        """
        Inserts the given values into the array at the position correspnding to the reaction of index k.
        
        Parameters
        ----------
        k : `int64`
            Index of the reaction
        value : `array_like`
            values to insert (np.dtype([('educts_index', (np.int64, (2,))), ('id', np.int64), ('unique_educts_id', np.float64)],  align=True))
        
        """
        
        self.Data[self.index[k]['id']] = value
        
    def reactions_left(self):
        print('((educt_1, educt_2), id): ', self.Data[0:self.n])

    def reactions_left_pointer(self, i):
        
        """Prints out all reactions left in the reactions list for educt of index i.
        
        Parameters
        ----------
        i : `int64`
            Educt index
        
        """
        
        if i in self.head:
            head = self.head[i]
            print('((educt_1, educt_2), id): ', self.Data[self.index[head[0]]['id']])
            next = self.index[head[0]]['next_reaction'][head[1]]
            
            while next[0]!=head[0]:
                print('((educt_1, educt_2), id): ', self.Data[self.index[next[0]]['id']])
                
                next = self.index[next[0]]['next_reaction'][next[1]]
                
#%%

# if __name__=='__main__':
    
    
    