# direct-mmap

## Introduction

Numpy memory-mapped array with direct I/O. No sequential read caching, increase the speed of random read. 

This package can increase speed a lot if you do random read in a memory-mapped array.

In general, when we read a file, the operating system will cache the read data in memory. For normal usage, this can increase the speed of frequent read of same data. However, when we random access the data, it will do sequential prefetching (but these data will not be used), which causes a lot of unnecessary reads. This package can avoid this by using direct I/O.

## Installation

### Requirements

Linux, x86_64 (because direct I/O is only supported on Linux). Python version >= 3.9.

### Method 1: pip

```bash
pip install direct-mmap
```

### Method 2: build from source

Install python3-dev (change to your python version, like python3.9-dev) first. This can be done by `sudo add-apt-repository ppa:deadsnakes/ppa; sudo apt update; sudo apt install python3-dev` (change to your python version) on Ubuntu.

Then run the following command:

```bash
git clone git@github.com:jtc1246/direct-mmap.git
cd direct-mmap/direct_mmap/cpp/
```
Then change the Makefile, `python3.9-config --includes` and `python3.9-config --ldflags` to your python version.

Then run the following command:

```bash
make
cd ../..
python setup.py install
```

## Usage

```python
from direct_mmap import direct_mmap

path = './test.npy'
mmap = direct_mmap(path, (10000,200), 'uint64', offset=0)
a = mmap[10:1000:10, 15:30]
```

The subscription method is generally similar to numpy, but currently it doesn't support using a numpy bool array. This will return a numpy array, not a view.

Currently all the shape and choice of data is handled in python, so it would be very slow if there is too much data segments (i.e. too much uncontinuous data parts). So for suggestion, you can remove the last dimension from subscription, and use numpy to select later.

## Performance

I run a benchmark with direct-mmap and np.memmap. The performance of direct-mmap is much better than np.memmap.

Each task is following:

```python
mmap = direct_mmap(file, (170640000, 40), 'uint64', offset=64) # direct_mmap
mmap = np.memmap(file, shape=(170640000, 40), dtype='uint64', offset=64) # np.memmap
mmap[id:id + 300000:1000]
```
This file is about 54 GB. There are 3 test cases. The first one is read 1000 times in single thread, the second is read 1000 times in 64 threads, the third is read 10000 times in 64 threads.

**Results:**

|                               |  np.memmap  |  direct_mmap  |
| --- | --- | --- |
|   1000 tasks, single thread   |    90.38s   |      36s      |
|     1000 tasks, 64 threads    |    9.24s    |     0.98s     |
|    10000 tasks, 64 threads    |    20.29s   |      9.4s     |

Environment is following:

1. Disk: Samsung PM983, 3.84 TB, PCIe, 4K 64 threads random read about 500K IOPS, single thread sequential read 910 MB/s
2. System: Ubuntu 22.04, in docker ubuntu 20.04, python 3.9.18, 64G RAM
3. Connection: PCIe 3.0 x4, 4 GB/s
4. Drop cache: "sudo sync; sudo sysctl -w vm.drop_caches=3", before np.memmap testing

The testing code is in direct_mmap/main.py at last.
