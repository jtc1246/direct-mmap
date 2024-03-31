'''
The cpp module for directIO memory-mapped numpy array.
'''


from typing import Tuple

def say_hello(name: str) -> None: 
    '''
    Print your name, just for test.
    '''
    ...

def new_array(path: str, length: int) -> int:
    '''
    Create a memory-mapped numpy array.
    
    1. path: file path
    2. length: the length of array in bytes
    
    Return -1 if file not exists, -2 if file size too small, else return the array id.
    '''
    ...

def set_np_array(array_id: int, np_ptr: int, bytes_range: Tuple[int, ...]) -> None:
    '''
    Set the value in numpy array,
    
    according to the memory-mapped array and bytes range in the tuple,
    
    ranges don't need to be in order or non-overlapped
    
    return -1 if file not exists, -2 if file size too small, else return 0.
    
    1. array_id: the return value of new_array
    2. np_ptr: the pointer of numpy array, in python int
    3. bytes_range: the range of bytes to set. Length of it should be multiple of 2, 
    format like this: (1st's start, 1st's end, 2nd's start, 2nd's end, ...). 
    End should be actual end + 1.
    '''
    ...

def close_array(array_id: int) -> int:
    '''
    Close the memory-mapped array.
    
    array_id: the return value of new_array
    '''
    ...
