import numpy as np
import os
import ctypes
from typing import Tuple
if __name__ == '__main__':
    import cpp.memmap as memmap
else:
    from .cpp import memmap
from threading import Lock

# based on numpy 1.26.0, you may change, but I will not support for other versions
try:
    sizes = {
        'int8': 1,
        'int16': 2,
        'int32': 4,
        'int64': 8,
        'int_': 8,
        'int': 8,
        'uint8': 1,
        'uint16': 2,
        'uint32': 4,
        'uint64': 8,
        'uint': 8,
        'float16': 2,
        'float32': 4,
        'float64': 8,
        'float_': 8,
        'float': 8,
        'complex64': 8,
        'complex128': 16,
        'complex_': 16,
        'complex': 16,
        'bool': 1,
        'bool8': 1,
        'bool_': 1,
        np.int8: 1,
        np.int16: 2,
        np.int32: 4,
        np.int64: 8,
        np.uint8: 1,
        np.uint16: 2,
        np.uint32: 4,
        np.uint64: 8,
        np.float16: 2,
        np.float32: 4,
        np.float64: 8,
        np.complex64: 8,
        np.complex128: 16,
        np.bool_: 1,
        np.int_: 8,
        np.float_: 8,
        np.complex_: 16,
    }
except:
    raise ImportError('This package is designed for numpy 1.26.0, may not work on other versions. Numpy data types is changing very fast. You may change the avove types to run on your device.')


class direct_mmap:
    '''
    Numpy memory-mapped array with direct I/O.
    
    No sequential read caching, increase the speed of random read.
    '''
    def __init__(self, path: str, shape: Tuple[int, ...], dtype, offset: int = 0):
        '''
        1. path: file path
        2. shape: the shape of the array
        3. dtype: the data type of the array, can be str or numpy dtype
        4. offset: the offset to the start of file, in bytes
        '''
        self._path = path
        self._shape = tuple(shape)
        self._dtype = dtype
        self._offset = offset
        if (offset < 0):
            raise ValueError('offset should not be negative')
        if (dtype not in sizes):
            raise ValueError('dtype not supported')
        self._size = sizes[dtype]
        self._dtype_szie = self._size
        if (len(shape) == 0):
            raise ValueError('shape should not be empty')
        for i in shape:
            if (i <= 0):
                raise ValueError('shape should not be negative or zero')
            self._size *= i
        array_id = memmap.new_array(path, self._size + offset)
        if (array_id == -1):
            raise FileNotFoundError('Error in opening file, please ensure that file exists, and your disk supports direct I/O.')
        if (array_id == -2):
            raise ValueError('file size smaller than array size + offset')
        self._array_id = array_id
        self._dimentions = len(shape)
        self._lock = Lock()
        self._thread_num = 0
        self._closed = False

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def size(self):
        return self._size // self._dtype_szie  # to make the performance same as numpy

    @property
    def actual_size(self):
        return self._size

    @property
    def ndim(self):
        return self._dimentions

    @property
    def offset(self):
        return self._offset

    @property
    def filename(self):
        return self._path

    def close(self) -> None:
        '''
        Close the array. Requires that all the reading is finished.
        '''
        with self._lock:
            if (self._thread_num != 0):
                raise Exception('This array is still in use.')
            memmap.close_array(self._array_id)
            self._closed = True

    def __del__(self):
        self.close()

    def __getitem__(self, key) -> np.ndarray:
        '''
        Only designed for small returned size, will handle all the data selection in python,
        will be very slow if the returned size is large (more accurately, uncontinous segments too much)

        Some optimization suggestions:
        1. In last dimension, don't select part, just select all, and use numpy to select later
        '''
        with self._lock:
            if (self._closed):
                raise Exception('This array has been closed.')
            self._thread_num += 1
        try:
            result = self.__getitem_wrapper(key)
        except Exception as e:
            with self._lock:
                self._thread_num -= 1
            raise
        with self._lock:
            self._thread_num -= 1
        return result

    def __getitem_wrapper(self, key) -> np.ndarray:
        '''
        Only designed for small returned size, will handle all the data selection in python,
        will be very slow if the returned size is large (more accurately, uncontinous segments too much)

        Some optimization suggestions:
        1. In last dimension, don't select part, just select all, and use numpy to select later
        '''
        if (type(key) == np.ndarray and key.dtype == np.bool_):
            # currently not supported
            raise NotImplementedError('bool indexing is currently not supported')
        if (type(key) != tuple):
            key = (key,)
        if (len(key) > self._dimentions):
            raise ValueError('too many indices')
        if (len(key) == 0):
            raise ValueError('Indices should not be empty')
        for t in key:
            if (type(t) != slice and type(t) != int and type(t) != list and type(t) != tuple):
                try:
                    _ = t.__int__()
                except:
                    name = type(t).__name__
                    raise ValueError(f'Subscription type {name} not supported')
        new_selections = []
        keep_dim = []
        cnt = 0
        for t in key:
            if (type(t) == int):
                x = t
                if (x < 0):
                    x += self._shape[cnt]
                if (x < 0 or x >= self._shape[cnt]):
                    raise ValueError('index out of range')
                new_selections.append((x,))
                keep_dim.append(False)
            elif (type(t) == list and type(t[0]) == int):
                x = []
                for i in t:
                    if (i < 0):
                        i += self._shape[cnt]
                    if (i < 0 or i >= self._shape[cnt]):
                        raise ValueError('index out of range')
                    x.append(i)
                    if (len(x) == 0):
                        raise ValueError('No data selected')
                new_selections.append(tuple(x))
                keep_dim.append(True)
            elif (type(t) == tuple and type(t[0]) == int):
                x = []
                for i in t:
                    if (i < 0):
                        i += self._shape[cnt]
                    if (i < 0 or i >= self._shape[cnt]):
                        raise ValueError('index out of range')
                    x.append(i)
                    if (len(x) == 0):
                        raise ValueError('No data selected')
                new_selections.append(tuple(x))
                keep_dim.append(True)
            elif (type(t) == slice):
                start = t.start
                end = t.stop
                step = t.step
                if (type(start) != int and start != None and type(end) != int and end != None and type(step) != int and step != None):
                    raise ValueError('slice should be int')
                if (start == None):
                    start = 0
                if (end == None):
                    end = self._shape[cnt]
                if (step == None):
                    step = 1
                if (start < 0):
                    start += self._shape[cnt]
                if (end < 0):
                    end += self._shape[cnt]
                start = max(0, start)
                end = min(self._shape[cnt], end)
                x = tuple(range(start, end, step))
                if (len(x) == 0):
                    raise ValueError('No data selected')
                new_selections.append(x)
                keep_dim.append(True)
            elif ((type(t) == tuple or type(t) == list) and type(t[0]) == bool):
                if (len(t) != self._shape[cnt]):
                    raise ValueError('bool index should have the same length as corresponding dimentions')
                x = []
                for i in range(len(t)):
                    if (t[i]):
                        x.append(i)
                if (len(x) == 0):
                    raise ValueError('No data selected')
                new_selections.append(tuple(x))
                keep_dim.append(True)
            elif ((type(t) == tuple or type(t) == list)):
                raise ValueError('Subscription type not supported')
            else:
                x = t.__int__()
                if (x < 0):
                    x += self._shape[cnt]
                if (x < 0 or x >= self._shape[cnt]):
                    raise ValueError('index out of range')
                new_selections.append((x,))
                keep_dim.append(False)
            cnt += 1
        assert (len(new_selections) == len(keep_dim) and len(new_selections) == len(key)), 'Internal error'
        last_dim = new_selections[-1]
        previous = last_dim[0]
        continuous_num = 1
        new_last_dim = []
        for i in last_dim[1:]:
            if (i == previous + continuous_num):
                continuous_num += 1
                continue
            new_last_dim.append((previous, continuous_num))
            previous = i
            continuous_num = 1
        new_last_dim.append((previous, continuous_num))
        new_shape = []
        new_size = self._dtype_szie
        for i in range(0, len(new_selections)):
            if (keep_dim[i]):
                new_shape.append(len(new_selections[i]))
                new_size *= len(new_selections[i])
            else:
                assert (len(new_selections[i]) == 1), 'Internal error'
        for i in range(len(new_selections), self._dimentions):
            new_shape.append(self._shape[i])
            new_size *= self._shape[i]
        new_array = np.zeros(new_shape, dtype=self._dtype)
        array_ptr = new_array.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        array_ptr = ctypes.cast(array_ptr, ctypes.c_void_p).value
        # print(array_ptr)
        bytes_ranges = []
        base_segment_szie = self._size
        for i in range(0, len(new_selections)):
            base_segment_szie //= self._shape[i]
        for start, length in new_last_dim:
            bytes_ranges.append((start * base_segment_szie, (start + length) * base_segment_szie))
        for i in range(len(new_selections) - 2, -1, -1):
            base_segment_szie *= self._shape[i + 1]
            new_bytes_ranges = []
            for j in new_selections[i]:
                for start, end in bytes_ranges:
                    new_bytes_ranges.append((start + j * base_segment_szie, end + j * base_segment_szie))
            bytes_ranges = new_bytes_ranges
        bytes_ranges = tuple(element + self._offset for tup in bytes_ranges for element in tup)
        # print(bytes_ranges)
        result = memmap.set_np_array(self._array_id, array_ptr, bytes_ranges)
        if (result == -1):
            raise FileNotFoundError('Error in opening file again, please ensure that file is not deleted or changed permission.')
        if (result == -2):
            raise ValueError('File size become smaller than origin, please don\'t change the file.')
        if (len(new_shape) == 0):
            return new_array[()]
        return new_array


# =========================== BENCKMARKING ===========================
# Disk: Samsung PM983, 3.84 TB, PCIe
#       4K 64 threads random read about 500K IOPS, single thread sequential read 910 MB/s
# System: Ubuntu 22.04, in docker ubuntu 20.04, python 3.9.18, 64G RAM
# Connection: PCIe 3.0 x4, 4 GB/s
# drop cache: "sudo sync; sudo sysctl -w vm.drop_caches=3", before np.memmap testing
# File: 54 GB, each time (task) access 300 segments of 320 bytes

#   Results

#   +-------------------------------+-------------+---------------+
#   |                               |  np.memmap  |  direct_mmap  |
#   |-------------------------------+-------------+---------------|
#   |   1000 tasks, single thread   |    90.38s   |      36s      |
#   |-------------------------------+-------------+---------------|
#   |     1000 tasks, 64 threads    |    9.24s    |     0.98s     |
#   |-------------------------------+-------------+---------------|
#   |    10000 tasks, 64 threads    |    20.29s   |      9.4s     |
#   +-------------------------------+-------------+---------------+


# =================== CODE FOR DIRECT_MMAP ===================

# from random import randint
# import time
# from threading import Thread
# from queue import Queue
# from _thread import start_new_thread
# file = '/media/jtc/PROCESSING/v01_646cb9c14f_EOS_170640000_40_2_uint64_1702823891665_传其体图兑片晚非串是法示电分会面么其女并由网核觉但期终道乐亿实南.jnp'

# mmap = direct_mmap(file, (170640000, 40), 'uint64', offset=64)
# q = Queue()
# threads = []

# for i in range(0, 1000):
#     r = randint(10000, 170640000 - 400000)
#     q.put(r)


# def worker():
#     while True:
#         try:
#             id = q.get(block=False)
#             x = mmap[id:id + 300000:1000]
#         except:
#             return

# t = time.time()
# for i in range(0, 64):
#     th = Thread(target=worker)
#     th.start()
#     threads.append(th)

# for th in threads:
#     th.join()

# print(time.time() - t) # 64 threads, 0.98 s, 
# # 10000 tasks 9.4 s

# t = time.time()
# for i in range(0, 1000):
#     r = randint(10000, 170640000 - 400000)
#     x = mmap[r:r + 300000:1000]
#     if (i % 100 == 0):
#         print(i)
# print(time.time() - t) # single thread, 36 s


# =================== CODE FOR NP.MEMMAP ===================

# from random import randint
# import time
# from threading import Thread
# from queue import Queue
# from _thread import start_new_thread
# file = '/media/jtc/PROCESSING/v01_646cb9c14f_EOS_170640000_40_2_uint64_1702823891665_传其体图兑片晚非串是法示电分会面么其女并由网核觉但期终道乐亿实南.jnp'

# mmap = np.memmap(file, shape=(170640000, 40), dtype='uint64', offset=64)
# q = Queue()
# threads = []
# x=np.zeros((300,40),dtype='uint64')

# for i in range(0, 1000):
#     r = randint(10000, 170640000 - 400000)
#     q.put(r)


# def worker():
#     while True:
#         try:
#             id = q.get(block=False)
#             x[:] = mmap[id:id + 300000:1000] # or it will just create a view
#         except:
#             return

# t = time.time()
# for i in range(0, 64):
#     th = Thread(target=worker)
#     th.start()
#     threads.append(th)

# for th in threads:
#     th.join()

# print(time.time() - t) # 64 threads, 9.24s
# # 10000 tasks 20.29 s (due to cache)

# t = time.time()
# for i in range(0, 1000):
#     r = randint(10000, 170640000 - 400000)
#     x[:] = mmap[r:r + 300000:1000]
#     if (i % 100 == 0):
#         print(i)
# print(time.time() - t) # single thread, 90.38 s