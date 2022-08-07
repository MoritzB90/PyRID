===============
Data structures
===============

In PyRID, molecules and particles constantly enter or leave the system due to reactions and other events. Therefore, we need a data structure that can efficiently handle this constant change in the number of objects we need to keep track of in our simulation. The same holds true for the molecular reactions occurring at each time step. These need to be listed and evaluated efficiently. Fortunately, variants of dynamic array data structures are tailored for such tasks, of which we use two kinds, the tightly packed dynamic array and the dynamic array with holes.

**The tightly packed dynamic array (dense array)**

A tightly packed dynamic array is a dynamic array (similar to lists in python or vectors in C++) where elements can be quickly deleted via a pop and swap mechanism. The problem with standard numpy arrays but also lists and C++ vectors is that deletion of elements is very expensive. For example, if we want to delete an element at index m of a numpy array of size n, numpy would create a new array that is one element smaller and copies all the data from the original array to the new array. Also, if we want to increase the size of a numpy array by appending an element, again, a new array will need to be created, and all data needs to be copied. This is extremely computationally expensive. One way to create a dynamic array (and python lists work in that way) is to not increase the array size each time an element is added but increase the array size by some multiplicative factor (usually 2). This consumes more memory but saves us from creating new arrays all the time. Now we simply need to keep track of the number of elements in the array (the length of the array) and the actual capacity, which can be much larger. One straightforward method to delete elements from the array is just to take the last element of the array and copy its data to wherever we want to delete an element (swapping). Next, we pop out the last element by decreasing the array length by 1. We call this type of array a ‘tightly packed array’ because it keeps the array tightly packed. One issue with this method is that elements move around. Thereby, to find an element by its original insertion index, we need to keep track of where elements move. One can easily solve this issue by keeping a second array that saves for each index the current location in the tightly packed array.


**The dynamic array with holes**

To store molecules and particles, we use a dynamic array with holes. A dynamic array with holes is an array where elements can be quickly deleted by creating ‘holes’ in the array. These holes are tracked via a free linked list. The array with holes has the benefit over the ‘tightly packed array’ that elements keep their original index because they are not shifted/swapped at any point due to deletion of other elements. This makes accessing elements by index a bit faster compared to the other approach. However, iterating over the elements in the array becomes more complex because we need to skip the holes. Therefore, we add a second array, which is a tightly packed array, that saves the indices of all the occupied slots in the array with holes (alternatively, we could add another linked list that connects all occupied slots). We can then iterate over all elements in the holes array by iterating over the tightly packed array. Keep in mind, however, that the order is not preserved in the tightly packed array, since, whenever we delete an element from the holes array, we also need to delete this element from the dense array by the pop and swap mechanism. As such, this method does not work well if we need to iterate over a sorted array. In that case, one should use a free linked list approach for iteration. As with the tightly packed dynamic array, the array size is increased by a multiplicative factor of 2 as soon as the capacity limit is reached.

.. figure:: Figures/DynamicArrays.png
    :width: 50%
    :name: fig:DynamicArray_Molecules
    
    **Dynamic arrays**


**Dynamic arrays used for reaction handling**

The data structure we need to organize the reactions is a little bit more complex than a simple dense, dynamic array or one with holes, as is used to keep track of all the rigid body molecules and particles in the system. The Reaction class is a combination of different dynamic arrays and a hash table. Let me motivate this: Our data structure needs to be able to do four things as efficient as possible:

#. Add reactions,
#. Delete single reactions,
#. Delete all reactions a certain particle participates in,
#. Return a random reaction from the list.

We need to be able to delete a single reaction whenever this reaction is not successful. We need to delete all reactions of a particle whenever a reaction was successful because, in this case, the particle is no longer available since it either got deleted or changed its type (except in the case where the particle participates as an enzyme). We need to be able to request a random reaction from the list because processing the reactions in the order they occur in the list would introduce a bias\footnote{Reactions of particles with a low index are added to the reaction list first, because particle distances are evaluated in the order particles occur in the corresponding list (which is, at least in the beginning, in ascending order, when the indices of particles have not yet been swapped around a lot). Thereby particle one would always have a higher chance of having a successful reaction when competing with other particles.}. Points 1. and 2. could be easily established with a simple dynamic array. However, point 3 is a bit more complicated but can be solved with a doubly free linked list embedded into a dynamic array with holes. This doubly linked list connects all reactions of a particle. To find the starting point (the head) of a linked list within the array for a certain particle, we save the head in a hash table (dictionary). A doubly linked list is necessary because we need to be able to delete a single reaction (of index k) from the linked list (point 2). As such, we need to be able to reconnect the linked list’s ‘chain’. Therefore, we need to know the element in the linked list that pointed to k (prev) in addition to the element/reaction k points to (next). Another problem that needs to be solved is that a reaction can be linked to at maximum two educts. Therefore, each next and prev pointer needs to be 4-dimensional: We need one integer for each educt to save the next (prev) reaction index and another integer {0,1} to keep track of whether in the next (prev) reaction, the particle is the first or the second educt, because this may change from reaction to reaction! Since the dynamic array, the doubly linked list is embedded, in has holes, picking a random reaction from the list becomes another issue. This can, however, easily be solved by adding another dynamic array (tightly packed), which keeps the indices of all the reactions that are left in a tightly packed format. Picking a random reaction is then as easy as drawing a uniformly distributed random integer between 0 and n, where n is the length of the dense array. A very nice introduction/overview to the kind of data structures used here has been written by Niklas Gray (see https://ourmachinery.com/post/data-structures-part-1-bulk-data/)


.. figure:: Figures/Reactions_DynamicArray.png
    :width: 50%
    :name: fig:DynamicArray_Reactions
    
    **Data structure for reactions handling**

