#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <Python.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <unordered_map>
#include <vector>
typedef uint64_t uint64;
using std::unordered_map;
using std::vector;

// The principle of error handling in this file:
// Everthing that can be checked in python, is not needed to check here
// Only handle those can't be checked in python, like file existence and size

#define PAGE_SIZE 4096
#define INIT_CAPACITY 100

uint64 min(uint64 a, uint64 b);
uint64 max(uint64 a, uint64 b);

struct np_array {
    uint64 length;
    int fd;
    char* name; // may be used to open again in multi-threading
};

struct mapping {
    uint64 page_loc;
    uint64 np_loc;
    uint64 length;
};

typedef struct np_array np_array;
typedef struct mapping mapping;
typedef vector<mapping> page;  // page info is a vector of mapping

np_array** all_arrays = NULL;
uint64 array_num = 0;
uint64 array_capacity = 0;
uint64 head_ptr = 0;  // the next available position in all_arrays

int threads = 0;

void init() {
    // printf("Doing init\n");
    all_arrays = (np_array**)malloc(sizeof(np_array*) * INIT_CAPACITY);
    array_capacity = INIT_CAPACITY;
    for (uint64 i = 0; i < array_capacity; i++) {
        all_arrays[i] = NULL;
    }
    head_ptr = 0;
    array_num = 0;
}

void double_capacity() {
    np_array** new_arrays = (np_array**)malloc(sizeof(np_array*) * array_capacity * 2);
    memcpy(new_arrays, all_arrays, sizeof(np_array*) * array_capacity);
    for (uint64 i = array_capacity; i < array_capacity * 2; i++) {
        new_arrays[i] = NULL;
    }
    free(all_arrays);
    all_arrays = new_arrays;
    array_capacity *= 2;
}

uint64 get_next_loc() {
    if (head_ptr == array_capacity) {
        double_capacity();
    }
    head_ptr++;
    return (head_ptr - 1);
}

void* directio_read(int fd, uint64 page_id, int* bytes_read) {
    // allocate memory in this function, need to free at outer
    void* buffer;
    posix_memalign(&buffer, PAGE_SIZE, PAGE_SIZE);
    lseek(fd, page_id * PAGE_SIZE, SEEK_SET);
    *bytes_read = read(fd, buffer, PAGE_SIZE);
    return buffer;
}

static PyObject* say_hello(PyObject* self, PyObject* args) {
    const char* name;
    if (!PyArg_ParseTuple(args, "s", &name))
        return NULL;
    printf("Hello %s!\n", name);
    Py_RETURN_NONE;
}

static PyObject* new_array(PyObject* self, PyObject* args) {
    // create a memory-mapped numpy array
    // only handle the file reading part in C,
    // don't consider array shape and dtype, even offset
    char* path;
    uint64 length;
    // get 2 parameters, one is file path, one is length
    if (!PyArg_ParseTuple(args, "sl", &path, &length)) {
        return NULL;
    }
    // printf("Path: %s\n", path);
    // printf("Length: %lu\n", length);
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd == -1) {
        // file not exists
        return Py_BuildValue("O", PyLong_FromLong(-1));
    }
    uint64 file_size = lseek(fd, 0, SEEK_END);
    if (length > file_size) {
        close(fd);
        // length is larger than file size
        return Py_BuildValue("O", PyLong_FromLong(-2));
    }
    np_array* array = (np_array*)malloc(sizeof(np_array));
    array->length = length;
    array->fd = fd;
    array->name = strdup(path);
    uint64 loc = get_next_loc();
    all_arrays[loc] = array;
    return Py_BuildValue("O", PyLong_FromLong(loc));
}

static PyObject* set_np_array(PyObject* self, PyObject* args) {
    // set the value in numpy array,
    // according to the memory-mapped array and bytes range in the tuple
    // ranges don't need to be in order or non-overlapped
    // return -1 if file not exists, -2 if file size too small, else return 0.

    uint64 id;
    uint64 np_ptr_tmp;  // tmp store the unit64 value of pointer
    void* np_ptr;
    PyObject* tuple;
    PyObject* item;  // for tuple
    uint64 value;  // for tuple
    Py_ssize_t tuple_size, i;  // for tuple

    // get the parameters
    // one is the id of the array (returned value in new_array)
    // one is the pointer of the numpy array
    // one is the tuple of starting and ending bytes
    if (!PyArg_ParseTuple(args, "KKO!",
                          &id,
                          &np_ptr_tmp,  // pass in pointer through uint64
                          &PyTuple_Type, &tuple)) {  // tuple
        return NULL;
    }
    np_ptr = (void*)np_ptr_tmp;  // convert uint64 into pointer

    // Handle the tuple
    tuple_size = PyTuple_Size(tuple);
    uint64* ranges = (uint64*)malloc(sizeof(uint64) * tuple_size);
    for (i = 0; i < tuple_size; i++) {
        item = PyTuple_GetItem(tuple, i);  // get the item from tuple, will not increase the reference count
        value = PyLong_AsUnsignedLongLong(item);  // convert python int into uint64
        ranges[i] = value;
    }

    Py_BEGIN_ALLOW_THREADS
    // multi-threading
    uint64 length = all_arrays[id]->length;
    char* name = all_arrays[id]->name;
    int fd = open(name, O_RDONLY | O_DIRECT);
    if (fd == -1) {
        // file not exists
        return Py_BuildValue("O", PyLong_FromLong(-1));
    }
    uint64 file_size = lseek(fd, 0, SEEK_END);
    if (length > file_size) {
        close(fd);
        // length is larger than file size
        return Py_BuildValue("O", PyLong_FromLong(-2));
    }

    // generate the page info
    unordered_map<uint64, page> pages;
    uint64 range_num = tuple_size / 2;
    uint64 current_np_loc = 0;
    for (uint64 i = 0; i < range_num; i++) {
        uint64 start = ranges[2 * i];
        uint64 end = ranges[2 * i + 1];
        uint64 page_start = start / PAGE_SIZE;
        uint64 page_end = (end + PAGE_SIZE - 1) / PAGE_SIZE;
        for (uint64 j = page_start; j < page_end; j++) {
            mapping m;
            uint64 file_start = max(j * PAGE_SIZE, start);
            uint64 file_end = min((j + 1) * PAGE_SIZE, end);
            m.page_loc = file_start - j * PAGE_SIZE;
            m.np_loc = current_np_loc;
            m.length = file_end - file_start;
            current_np_loc += m.length;
            pages[j].push_back(m);
        }
    }

    // start to read file
    // int fd = all_arrays[id]->fd;
    // uint64 length = all_arrays[id]->length;
    for (auto it = pages.begin(); it != pages.end(); it++) {
        uint64 page_id = it->first;
        page p = it->second;
        int bytes_read;
        void* buffer = directio_read(fd, page_id, &bytes_read);
        if (page_id < length / PAGE_SIZE && bytes_read != PAGE_SIZE) {
            printf("Error in directio reading.\n");
            fprintf(stderr, "Error in directio reading.\n");
            // assert(0);
        }
        if (page_id == length / PAGE_SIZE && bytes_read < length % PAGE_SIZE) {
            printf("Error in directio reading.\n");
            fprintf(stderr, "Error in directio reading.\n");
            // assert(0);
        }
        for (uint64 i = 0; i < p.size(); i++) {
            mapping m = p[i];
            memcpy(np_ptr + m.np_loc, buffer + m.page_loc, m.length);
        }
        free(buffer);
    }
    free(ranges);
    Py_END_ALLOW_THREADS
    return Py_BuildValue("O", PyLong_FromLong(0));
}

static PyObject* close_array(PyObject* self, PyObject* args) {
    // close a memory-mapped numpy array
    // need to pass in the returned value of new_array
    uint64 loc;
    // get the parameter
    if (!PyArg_ParseTuple(args, "l", &loc)) {
        return NULL;
    }
    // printf("Close array at %lu\n", loc);
    int fd = all_arrays[loc]->fd;
    free(all_arrays[loc]->name);
    close(fd);
    free(all_arrays[loc]);
    all_arrays[loc] = NULL;
    return Py_BuildValue("O", PyLong_FromLong(0));
}

static PyMethodDef Methods[] = {
    {"say_hello", say_hello, METH_VARARGS, "Greet someone."},
    {"set_np_array", set_np_array, METH_VARARGS, "Set the value in numpy array."},
    {"new_array", new_array, METH_VARARGS, "Create a memmory-mapped numpy array."},
    {"close_array", close_array, METH_VARARGS, "Close an array."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "memmap", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1, /* size of per-interpreter state of the module,
           or -1 if the module keeps state in global variables. */
    Methods};

PyMODINIT_FUNC PyInit_memmap(void) {
    init();
    return PyModule_Create(&module);
}

// ====================== OTHERS START HERE ======================
uint64 min(uint64 a, uint64 b) {
    return a < b ? a : b;
}

uint64 max(uint64 a, uint64 b) {
    return a > b ? a : b;
}