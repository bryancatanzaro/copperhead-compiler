#pragma once
#include <stddef.h>
#include <cstdlib>
#include <iostream>

class host_alloc {
  public:
    void* allocate(size_t s) const {
        return malloc(s);
    }
    void deallocate(void* p) const {
        return free(p);
    }
};

#ifdef CUDA_SUPPORT
#include <cuda_runtime.h>

class cuda_alloc {
  public:
    void* allocate(size_t s) const {
        void* r;
        cudaMalloc(&r, s);
        return r;
    }
    void deallocate(void* p) const {
        cudaFree(p);
        return;
    }
};
#endif

//XXX Add memory pool allocators here
