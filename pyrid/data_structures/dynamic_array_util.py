# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numba as nb
import numpy as np
from numba.experimental import jitclass

spec_darray = [
    ('n', nb.int64),
    ('capacity', nb.int64),
    ('index_capacity', nb.int64),
    ('Data', nb.int64[:]),
    ('index', nb.int64[:]),
]


@jitclass(spec_darray)
class DenseArray(object):
    
    """
    Implementation of a dense dynamics array.
    
    Attributes
    ----------
    n : `int`
        Count actual elements in the array.
    capacity : `int`
        Total capacity of the array.
    index : `int64[:]`
        Array to keep track of element indices.
    index_capacity : `int`
        Total capacity of the index array.
    A : `int64[:]`
        Array containing the actual data.    
        
    Methods
    -------
    add(self, pid)
        Adds a particle index to the array.
    remove(self, k)
        Removes the element with index k from the array.
    _resize(self, new_cap)
        Resize internal data array to a new capacity (new_cap).
    _resize_index(self, new_cap)
        Resize the index array to a new capacity (new_cap).
    make_array(self, new_cap)
        Returns a new array with capacity new_cap.
    make_index(self, new_cap)
        Returns a new array with capacity new_cap.
    clear(self)
        Clears the array by setting the number of elements in the array to 0.

        
    Notes
    -----
    A dense dynamic array is a dynamic array (similar to lists in python or vectors in C++) where elements can be quickly deleted via a pop and swap mechanism. The problem with standard numpy arrays but also lsist and C++ vectors is that deletion of elements is very expensive. If we want to delete an element at index m of a numpy array of size n, numpy would create a new array that is one element smaller and copy all te data from the original array to the new array. ALso, if we want to increase the size of a numpy array by appending an element, again, a new array would need to be created and all data needs to be copied. This is extremely computationaly expensive. One way to create a dynamic array (and python lists work in that way) is, to not increase the array size each time an element is added but invcrease the array size by some multiplicative factor (usually 2). This consumes more memory but saves us from creating new arrays all the time. Now we simply need to keep track of the number of elements in the array (the length of the array) and the actual capacity which can be much larger.
    To delete elements from the array, one strait forward method is to just take the last element of the array and copy its data to wherever we want to delete an element (swapping). Next, we simply pop out the last element by decreasing the array length by 1. We call this type of array a 'dense array' because it keeps the array tightly packed. One issue with this method is that, if we later on need to find an element by its original index, we need to keep track of that. One can easily solve this issue by keeping a second array that saves for each index the current location in the dense array. 
    
    .. image:: ../../../Graphics/DenseArray.png
        :width: 30%
    
    |
    
    """

    def __init__(self):
        self.n = 0 # Count actual elements (Default is 0)
        self.capacity = 1#20000 # Default Capacity
        self.index_capacity = 1
        self.Data = self.make_array(self.capacity)
        self.index = self.make_index(self.capacity)

    def __getitem__(self, k):
        
        """
        Return element at index k.

        Parameters
        ----------
        k : `int64`
            Index
        
        Returns
        -------
        array like
            Data at index k
        """
        
        return self.Data[k]
    
    def __setitem__(self, k, value):
        
        """ 
        Set value of item at index k.

        Parameters
        ----------
        k : `int64`
            Index
        value : `array like`
            Value to which the element at index k is set to.
        """
        
        self.Data[k] = value
    
    def add(self, pid):
        
        """Adds a particle index to the array.
        
        Parameters
        ----------
        pid : `int64`
            particle index
        
        """
        
        if self.n == self.capacity:
            self._resize(2 * self.capacity)      

        if pid >= self.index_capacity:
            self._resize_index(2 * self.capacity)  
            
        self.index[pid] = self.n
        self.Data[self.n] = pid
        
        self.n += 1
        
    def remove(self, k):
        
        """Removes the element with index k from the array. Attention: k does not refer to the index of the array but the id of a specific element!
        
        Parameters
        ----------
        k : `int64`
            Index
        
        """
        
        # Get the id (id_swap) of the last item in the reactions list
        self.Data[self.index[k]] = self.Data[self.n-1]
        
        self.index[self.Data[self.n-1]] = self.index[k]
        
        self.n -= 1

    def _resize(self, new_cap):
        
        """Resize internal data array to a new capacity (new_cap).
        
        Parameters
        ----------
        new_cap : `int64`
            New array capacity.
        
        """

        Data_resized = self.make_array(new_cap) # New bigger array
          
        for k in range(self.n): # Reference all existing values
            Data_resized[k] = self.Data[k]
              
        self.Data = Data_resized # Call A the new bigger array
        self.capacity = new_cap # Reset the capacity
        
    def _resize_index(self, new_cap):
        
        """Resize the index array to a new capacity (new_cap).
        
        Parameters
        ----------
        new_cap : `int64`
            New array capacity.
          
        """

        index_new = self.make_index(new_cap) # New bigger array
          
        for k in range(self.n): # Reference all existing values
            index_new[k] = self.index[k]
              
        self.index = index_new # Call A the new bigger array
        self.index_capacity = new_cap # Reset the capacity
          
    def make_array(self, new_cap):
        """
        Returns a new array with capacity new_cap

        Parameters
        ----------
        new_cap : `int64`
            New array capacity.
        
        Returns
        -------
        int64[:]
            numpy zeros array of size new_cap.
        """
        
        return np.zeros(new_cap, dtype = np.int64)

    def make_index(self, new_cap):
        """
        Returns a new array with capacity new_cap

        Parameters
        ----------
        new_cap : `int64`
            New array capacity.
        
        Returns
        -------
        int64[:]
            numpy zeros array of size new_cap.
        """
        
        return np.zeros(new_cap, dtype = np.int64)
    
    def clear(self):
        """
        Clears the array by setting the number of elements in the array to 0.

        """

        self.n = 0
        
        
        
#%%


class HolesArray(object):
    
    """
    Implementation of a dynamics array "with holes".
    
    Attributes
    ----------
    n : `int`
        Count actual elements in the array.
    capacity : `int`
        Total capacity of the array.
    occupied : `obj`
        Dense array that keeps track of the occupied slots.
    Data : `array_like`
        Structured numpy array containing a linked list, connecting the holes in the array, and containing the actual data.
        `dtype = np.dtype([('next', np.uint64),('name', 'U20'), ... , align = True)`
    slot : `int64`
        saves the current slot when a new item is added. 
    item_t : `np.dtype`
        structured array data type that contains fields for the free linked list and the actual data. dtype = np.dtype([('next', np.uint64),('name', 'U20'), ... , align = True)

        
    Methods
    -------
    len(self)
        Returns the length of the array. The length is defined by the number of elements in the array not counting the holes.
    iterator(self)
        Iterates through all nonempty (no-hole) elements in the Data array. Note: This method is not supported by Numba at the time.
    insert(self, value)
        Inserts a new item into an empty slot (a hole) of the array or appends it if no empty slots are available.
    add_to_occupied(self, slot)
        Adds a slot index to the occupied indices list.
    _append(self, value)
        Appends an item to the end of the array. If the capacity of the array is reached, the array is resized.
    delete(self, i)
        Deletes an item from the array. It does so by creating a hole in the array.
    _resize(self, new_cap)
        Resizes the Data array to capacity new_cap.
    make_array(self, new_cap)
        Returns a new array with new_cap capacity.

        
    Notes
    -----
    A dynamic array with holes is a dynamic array (similar to lists in python or vectors in C++) where elements can be quickly deleted by creating 'holes' in the array. These holes are tracked via a free linked list. The array with holes has the benefit over 'dense arrays', that elements keep their original index, because they are not shifted/swapped at any point due to deletion of other elements. This makes accessing elements by index a bit faster compared to the dense array approach. However, iterating over the elements in the array becomes more complex, because we need to skip the holes. Therefore, we add a second array, which is a dense array, that saves the indices of all the occupie slots in the array with holes (alternatively we could add another linked list that connects all occupied slots). Now we can iterate over all elements in the holes array by iterating over the dense array. Keep in mind, however, that the order is not preserved in the dense array, since, whenever we delete an element from the holes array, we also need to delete this element from the dense array by the pop and swap mechanism. As such, this method does not work well if we need to iterate over a sorted array. 
    As with the dense dynamic array, the array size is increased by a multiplicative factor of 2 as soon as the capacity limit is reached.
    
    .. image:: ../../../Graphics/HolesArray.png
        :width: 40%
    
    |
    
    """

    def __init__(self, item_t):
        self.n = 1#0 # Count actual elements (Default is 0)
        self.capacity = 1 # Default Capacity
        self.item_t = item_t
        self.Data = self.make_array(self.capacity)
        self.Data[0]['next'] = 0
        self.occupied = DenseArray()
        self.slot = 0
        
    
    def len(self):
        """
        Returns the length of the array. The length is defined by the number of elements in the array not counting the holes. The actual size of the array can be accessed by the capacity attribute.
        """
        return self.n
      
    def __getitem__(self, k):
        
        """
        Return element at index k+1 (element 0 is reserved for the head of the free list).
        Any item will be returned, also holes, because checking whether an item is a hole at each access
        operation is too expensive.
        Also, there will be no test, whether the index is within bounds, as this can also reduce speed by up to 25%.
        To get a list of all items which are not holes, use the occupied attribute!
        
        Parameters
        ----------
        k : `int64`
            Index
        
        
        Returns
        -------
        array like
            Value of the structured array at index k (actually k+1)
        
        """
        
        # if not 0 <= k+1 <self.n:
        #     # Check it k index is in bounds of array
        #     raise IndexError('Index is out of bounds !')
            
        # if self.Data[k+1]['next']!=k+1:
        #     # raise IndexError('Item has been deleted')
        #     print('Warning: Item has been deleted')
          
        return self.Data[k+1] # Retrieve from the array at index k


    def __setitem__(self, k, value):
        
        """ 
        Sets the value of the array at index k+1. Index 0 is reserved for the head of the free list 
        which is used to find the holes in the array.
        
        Parameters
        ----------
        k : `int64`
            Index
        value : self.item_t
            value to insert at index k. The data type is whatever has been set for the np.dtype of the structured array that is the Data attribute. The data type is kept by the item_t attribute.
        
        """
        
        self.Data[k+1] = value
        
    def iterator(self):
        
        """ 
        Iterates through all nonempty (no-hole) elements in the Data array. Note: This method is not supported by Numba at the time.
        
        """

        i = 1
        while i<self.n:
            while self.Data[i]['next'] != i:
                i += 1
                
            yield self.Data[i]
            i += 1
                
        

    def insert(self, value):
        
        """ 
        Inserts a new item into an empty slot (a hole) of the array or appends it if no empty slots are available. 
        The method does so by accessing the next pointer of the head element. If the next pointer is 0, i.e. it points to the head element itself, no empty slots are available and the 
        new item is appended to the array by calling the _append method. If the next pointer is greater zero, the new item is added at the index the next pointer points to.
        
        Parameters
        ----------
        value : self.item_t
            value to insert at index k. The data type is whatever has been set for the np.dtype of the structured array that is the Data attribute. The data type is kept by the item_t attribute.
        
        
        """
        
        
        # value = np.zeros(1, dtype = self.item_t)[0]
        
        self.slot = self.Data[0]['next']
        self.Data[0]['next'] = self.Data[self.slot]['next']
        # If the freelist is empty, slot will be 0, because the header
        # item will point to itself.
        if self.slot>0:
            self.Data[self.slot] = value # Set self.slot index to element
            self.Data[self.slot]['next'] = self.slot
 
            self.occupied.add(nb.int64(self.slot-1))

        else:
            self._append(value)
            
            
    def add_to_occupied(self, slot):
        """ 
        Adds a slot index to the occupied indices list.
        
        Parameters
        ----------
        slot : `int64`
            Slot index.
        
        
        """

        self.occupied.add(nb.int64(slot))
        
    def _append(self, value):
        
        """
        Appends an item to the end of the array. If the capacity of the array is reached, the array is resized.
        
        Parameters
        ----------
        value : self.item_t
            value to insert at index k. The data type is whatever has been set for the np.dtype of the structured array that is the Data attribute. The data type is kept by the item_t attribute.
        
        
        """
        
        
        self.slot = self.n
        
        if self.n == self.capacity:
            # Double capacity if not enough room
            self._resize(2 * self.capacity) 
          
        self.Data[self.slot] = value # Set self.slot index to element
        self.Data[self.slot]['next'] = self.slot
        self.occupied.add(self.slot-1)
        self.n += 1
  
  
    def delete(self, i):
        
        """
        Deletes an item from the array. It does so by creating a hole in the array. The holes are kept track of by a free linked list. 
        A hole is created by removing the item from the list of of occupied indices and updating the free linked list:
        The head of the free linked list now points to the deleted item and the next pointer at the index of the deleted item points to the hole where the head originally pointed to.
        
        Parameters
        ----------
        i : `int64`
            Index
        parameter_2 : dtype
            Some Information
        
        """

        # if self.n==0:
        #     print("Array is empty deletion not Possible")
        #     return
        
        # if i not in self.occupied:
        #     raise IndexError("Index out of bound....deletion not possible") 
        
        # Add to the freelist, which is stored in slot 0.
        self.Data[i+1]['next'] = self.Data[0]['next']
        self.Data[0]['next'] = i+1
        self.occupied.remove(i)
        
        #TODO: Shrink at some point? -> difficult to do for a dynamic array with holes!
        

    def _resize(self, new_cap):
        
        """
        Resizes the Data array to capacity new_cap.
        
        Parameters
        ----------
        new_cap : `int64`
            Array capacity.
        
        
        """
        
          
        B = self.make_array(new_cap) # New bigger array
          
        for k in range(self.n): # Reference all existing values
            B[k] = self.Data[k]
              
        self.Data = B # Call A the new bigger array
        self.capacity = new_cap # Reset the capacity
          

    def make_array(self, new_cap):
        """
        Returns a new array with new_cap capacity

        Parameters
        ----------
        new_cap : `int64`
            Array capacity.

        """
        return np.zeros(new_cap, dtype = self.item_t)
    
    
#%%

item_t_index = np.dtype([('next_hole', np.int64), ('id', np.int64), ('next_reaction', np.int64, (2,2)), ('prev_reaction', np.int64, (2,2))],  align=True)


spec_holes_array = [
    ('n', nb.int64),
    ('capacity', nb.int64),
    ('item_t', nb.typeof(item_t_index)),
    ('Data', nb.typeof(np.zeros(1, dtype = item_t_index))),
    ('slot', nb.int64),
]


@jitclass(spec_holes_array)
class HolesArrayReact(object):
    
    """
    Implementation of a dynamics array "with holes" that is used for the index attribute in PyRIDs Reaction class. 
    In the Reaction class this array keeps track of the reaction indices :func:`pyrid.reactions.reactions_registry_util.Reaction`.
    
    Attributes
    ----------
    n : `int`
        Count actual elements in the array.
    capacity : `int`
        Total capacity of the array.
    occupied : `obj`
        Dense array that keeps track of the occupied slots.
    Data : `array_like`
        Structured numpy array containing a linked list, connecting the holes in the array, and containing the actual data.
    slot : `int64`
        saves the current slot when a new item is added. 
    item_t : `np.dtype`
        structured array data type that contains fields for the free linked list and the actual data.
    
    Methods
    -------
    allocate_slot()
        Allocates a new slot by finding the next hole in the array.
    _resize(new_cap)
        Increases the size of the array to capacity new_cap.
    make_array(new_cap)
        Returns a new array with new_cap capacity
    clear()
        clears the array by setting the number of currently inserted elements to 0.



    """
    

    def __init__(self):
        self.n = 0 # Count actual number of elements, including holes (Default is 0)
        self.capacity = 1 # Default Capacity
        self.item_t = item_t_index
        self.Data = self.make_array(self.capacity)
        
        
    def __getitem__(self, k):
        
        """
        Returns element at index k+1 (element 0 is reserved for the head of the free list).

        k : `int64`
            Index

        Returns
        -------
        array like
            Value of the structured array at index k (actually k+1)
        """
        
        return self.Data[k+1]
    
    def __setitem__(self, k, value):
        
        """ 
        Sets the value of the array at index k+1. Index 0 is reserved for the head of the free list 
        which is used to find the holes in the array.
        
        Parameters
        ----------
        k : `int64`
            Index
        value : self.item_t
            value to insert at index k. The data type is whatever has been set for the np.dtype of the structured array that is the Data attribute. The data type is kept by the item_t attribute.
        
        """
        
        self.Data[k+1] = value
        

    def allocate_slot(self):
        
        """Allocates a new slot by finding the next hole in the array. If no empty slot/hole is available the array is resized.

        Notes
        -----
        The allocate_slot method does only allocate a new slot but does not insert a new item / value at that slots´ location. It only returns the index of that allocated slot. 
        However, allocate_slot automatically assumes that the slot will be filled with Data after it has been called and therefore already updates 
        the free linked list and resizes the array (in case no holes are available and the array capacity has been reached).
        
        Returns
        -------
        int64
            Index of the next available slot in the array.
        
        """
        
        # First, we assume that there exist empty slots:
        self.slot = self.Data[0]['next_hole']
        # Since this hole is now occupied, update the head of the holes linked list,
        # i.e. take the next_hole that the current hole, which is no longer a hole
        # and therefore not part of the linked list anymore, points to and set it as the head (self.Data[0]).
        # Note: if self.slot = 0, the head points to itself, so the next step does basically nothing.
        self.Data[0]['next_hole'] = self.Data[self.slot]['next_hole']
        
        # If the freelist is empty, slot will be 0, because the header
        # item will point to itself, so we need to append a new slot. 
        # Otherwise, we return self.slot:
        if self.slot:
            return nb.int64(self.slot)
        
        # If slot==0, we first need to check if we reached the capacity of our array, before we append a new element:
        if self.n == self.capacity-1:
            # Double capacity if not enough room
            self._resize(2 * self.capacity)      
        
        # Since we are appending and element at the end of the array, the size of our array needs to increase by 1
        self.n += 1
        return nb.int64(self.n-1)

    # def __getitem__(self, k):
          
    #     return self.Data[k]


    # def __setitem__(self, k, value):

    #     self.Data[k] = value        

    # def delete(self, i):
        
    #     """Deletes an element from the array by creating a hole. Also updates a free linked list connecting all reactions of a particle.
        
    #     (Note: Not actually used in the reaction_registry class, which has its own, more complex deletion method. Just added here to have a delete method in the dense dynamic array (maybe usefull in future iterations of the code where this class is reused...could also be deleted, not sure yet.))
        
    #     Parameters
    #     ----------
    #     parameter_1 : dtype
    #         Some Information
    #     parameter_2 : dtype
    #         Some Information
        
    #     Raises
    #     ------
    #     NotImplementedError (just an example)
    #         Brief explanation of why/when this exception is raised
        
    #     Returns
    #     -------
    #     dtype
    #         Some information
        
    #     """
        
    #     # Add to the freelist, which is stored in slot 0.
    #     self.Data[i+1]['next_hole'] = self.Data[0]['next_hole']
    #     self.Data[0]['next_hole'] = i+1
        
    #     for j in range(2):
    #         # Get the reaction_id and educt_no of the next pointer
    #         # We need to assign these to the previous next pointer
    #         next_reaction_id = self.Data[i+1]['next_reaction'][j][0]
    #         next_educt_no = self.Data[i+1]['next_reaction'][j][1]
    #         # Therefore, also get the reaction_id and educt_no of the prev pointer
    #         prev_reaction_id = self.Data[i+1]['prev_reaction'][j][0]
    #         prev_educt_no = self.Data[i+1]['prev_reaction'][j][1]
            
    #         self.Data[prev_reaction_id+1]['next_reaction'][prev_educt_no][0] = next_reaction_id
    #         self.Data[prev_reaction_id+1]['next_reaction'][prev_educt_no][1] = next_educt_no
    #         # However, we also need to reassign the previous pointer of the next element
    #         self.Data[next_reaction_id+1]['prev_reaction'][next_educt_no][0] = prev_reaction_id
    #         self.Data[next_reaction_id+1]['prev_reaction'][next_educt_no][1] = prev_educt_no            

    def _resize(self, new_cap):
        
        """Increases the size of the array to capacity new_cap.
        
        Parameters
        ----------
        new_cap : `int`
            array size
        
        """
          
        Data_resized = self.make_array(new_cap) # New bigger array
          
        for k in range(self.n+1): # Reference all existing values
            Data_resized[k] = self.Data[k]
              
        self.Data = Data_resized # Call A the new bigger array
        self.capacity = new_cap # Reset the capacity
          
    def make_array(self, new_cap):
        
        """Returns a new array with new_cap capacity
        
        Parameters
        ----------
        new_cap : `int`
            array size

        
        Returns
        -------
        array_like
            array with dtype = self.item_t
        
        """
        
        return np.zeros(new_cap, dtype = self.item_t)

    def clear(self):
        
        """clears the array by setting the number of currently inserted elements to 0.
        
        """
        
        # Clearing the array is as simple as:
        self.n = 0

        # Which means, setting the actual number of elements as well as free slots in the array to 0
        # The capacity of the actual array will however stay the same unless we reinitialize self.Data
        # by calling make_array(self, new_cap)


#%%

key_head = nb.int64
value_head = nb.int64[:]

class DenseArrayReact(object):
    
    """
    Implementation of a dense dynamics array that is used as a parent class for PyRIDs Reaction class :func:`pyrid.reactions.reactions_registry_util.Reaction`.
    
    Attributes
    ----------
    n : `nb.int64`
        Number of reactions in list.
    n_success : `nb.int64`
        Number of successfull reactions.
    n_total : `nb.int64`
        Number of total reactions.
    capacity : `nb.int64`
        Current capacity of dynamic array.
    A : `array_like`
        Struct array containing reaction data reaction data.
        
        `dtype = np.dtype([('educts_index', (np.int64, (2,))), ('id', np.uint64), ('unique_educts_id', np.float64)],  align=True)`
    item_t : `numpy dtype`
        dtype of reaction struct array A (np.dtype([('educts_index', (np.int64, (2,))), ('id', np.uint64), ('unique_educts_id', np.float64)],  align=True))
    index : `holes_array_react.class_type.instance_type`
        Index saved in dynamic array with holes.
    slot : `nb.uint64`
        Current/last assigned slot of dynamic array (index).
    head : `nb.types.DictType(nb.int64, nb.int64[:])`
        Head of free linked list containing all reactions of an educt.
        
    
    Methods
    -------
    _resize(new_cap)
        Increases the size of the array to capacity new_cap.
    make_array(new_cap)
        Creates a new array of size new_cap.
    clear()
        Clears the array by setting the number of currently inserted elements to 0.

    
    """
    

    def __init__(self, item_t_main):
        self.n = 0 # Count actual elements (Default is 0)
        self.capacity = 1 # Default Capacity
        self.item_t = item_t_main
        self.Data = self.make_array(self.capacity)
        self.index = HolesArrayReact()
        self.dummy = np.zeros(1, dtype = self.item_t)
        
        self.head = nb.typed.Dict.empty(key_head, value_head)


    def __getitem__(self, k):
        
        """
        Return element at index k.
        """
        
        return self.Data[k]
    
    def __setitem__(self, k, value):
        
        """ 
        Set value of item at index k.
        """
        
        self.Data[k] = value
        
        
    # def __getitem__(self, k):
    #     """
    #     Return element at index k+1 (element 0 is reserved for the head of the free list).
    #     Any item will be returned, also holes, because checking whether an item is a hole at each access operation is too expensive.
    #     Also, there will be no test, whether the index is within bounds, as this can also reduce speed by up to 25%.
    #     To get a list of all items which are not holes, use the occupied attribute!
    #     """
          
    #     return self.Data[self.index[k]['id']]


    # def __setitem__(self, k, value):
    #     """
    #     The k+1 element is returned, because element 0 is reserved for the head of the free list 
    #     which is used to find the holes in teh array.
    #     """
        
    #     self.Data[self.index[k]['id']] = value
        

        
    # def delete(self, k):
        
    #     """Deletes an element from the array by swap and pop.
    #     This is a dense dynamic array. As such, elements get swapped around by a pop and swap mechanism. Element k, which we like to delete amy not be at the place where it was originally inserted. Therefore, k does not refer to the element currently at index k in the dense array but needs to be looked up in the index array (self.index).
        
    #     (Note: Not actually used in the reaction_registry class, which has its own, more complex deletion method. Just added here to have a delete method in the dense dynamic array (maybe usefull in future iterations of the code where this class is reused...could also be deleted, not sure yet.))
        
    #     Parameters
    #     ----------
    #     parameter_1 : dtype
    #         Some Information
    #     parameter_2 : dtype
    #         Some Information
        
    #     Raises
    #     ------
    #     NotImplementedError (just an example)
    #         Brief explanation of why/when this exception is raised
        
    #     Returns
    #     -------
    #     dtype
    #         Some information
        

    #     """
        
    #     # We dont do a swap here but simpy set value at k to value at n-1. 
        
    #     # Get the id (id_swap) of the last item in the reactions list
    #     id_swap = self.Data[self.n-1]['id']
    #     self.Data[self.index.Data[k+1]['id']] = self.Data[id_swap]
        
    #     self.index.delete(k)
        
    #     self.index.Data[id_swap+1]['id'] = self.index.Data[k+1]['id']  
        
    #     self.n -= 1

    def _resize(self, new_cap):
        
        """Increases the size of the array to capacity new_cap.
        
        Parameters
        ----------
        new_cap : `int`
            array size
        
        """
        
        # self.index._resize(new_cap)
        
        Data_resized = self.make_array(new_cap) # New bigger array
          
        for k in range(self.n): # Reference all existing values
            Data_resized[k] = self.Data[k]
              
        self.Data = Data_resized # Call A the new bigger array
        self.capacity = new_cap # Reset the capacity
          
    def make_array(self, new_cap):
        
        """Returns a new array with new_cap capacity
        
        Parameters
        ----------
        new_cap : `int`
            array size

        
        Returns
        -------
        array_like
            array with dtype = self.item_t
        
        """
        
        return np.zeros(new_cap, dtype = self.item_t)

    def clear(self):
        """clears the array by setting the number of currently inserted elements to 0.
        
        """
        
        self.n = 0
        self.index.clear()