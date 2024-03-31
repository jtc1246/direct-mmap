'''
Numpy memory-mapped array with direct I/O.

No sequential read caching, increase the speed of random read.
'''

from .main import direct_mmap

__all__ = ['direct_mmap']
