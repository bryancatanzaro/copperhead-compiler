#include "cudata.h"
#include <cuda_runtime.h>
#include <string.h>
#include <cassert>

typedef std::pair<void*, ssize_t> seq;
using std::make_pair;

cuarray::cuarray() {
    l_d = NULL;
    r_d = NULL;
    s = 0;
    e = 0;
    n = 0;
    clean_local = true;
    clean_remote = true;
}

cuarray::~cuarray() {
    if (l_d != NULL) {
        free(l_d);
        l_d = NULL;
    }
    if (r_d != NULL) {
        cudaFree(r_d);
        r_d = NULL;
    }
}

cuarray::cuarray(ssize_t _n, CUTYPE _t, bool host) {
    clean_local = host;
    clean_remote = !host;
    n = _n;
    t = _t;
    if (t == CUBOOL) {
        e = sizeof(bool);
    } else if (t == CUINT32) {
        e = sizeof(int);
    } else if (t == CUINT64) {
        e = sizeof(long);
    } else if (t == CUFLOAT32) {
        e = sizeof(float);
    } else if (t == CUFLOAT64) {
        e = sizeof(double);
    } else {
        e = 0;
    }
        
    s = e * n;
    if (clean_local) {
        l_d = malloc(s);
    } else {
        cudaMalloc(&r_d, s);
    }
}

cuarray::cuarray(ssize_t _n, bool* l) {
    clean_local = true;
    clean_remote = false;
    n = _n;
    e = sizeof(bool);
    s = e * n;
    l_d = malloc(s);
    memcpy(l_d, l, s);
    t = CUBOOL;
}

cuarray::cuarray(ssize_t _n, int* l) {
    clean_local = true;
    clean_remote = false;
    n = _n;
    e = sizeof(int);
    s = e * n;
    l_d = malloc(s);
    memcpy(l_d, l, s);
    t = CUINT32;
}

cuarray::cuarray(ssize_t _n, long* l) {
    clean_local = true;
    clean_remote = false;
    n = _n;
    e = sizeof(long);
    s = e * n;
    l_d = malloc(s);
    memcpy(l_d, l, s);
    t = CUINT64;
}

cuarray::cuarray(ssize_t _n, float* l) {
    clean_local = true;
    clean_remote = false;
    n = _n;
    e = sizeof(float);
    s = e * n;
    l_d = malloc(s);
    memcpy(l_d, l, s);
    t = CUFLOAT32;
}

cuarray::cuarray(ssize_t _n, double* l) {
    clean_local = true;
    clean_remote = false;
    n = _n;
    e = sizeof(float);
    s = e * n;
    l_d = malloc(s);
    memcpy(l_d, l, s);
    t = CUFLOAT64;
}


void cuarray::retrieve() {
    //Lazy data movement
    if (!clean_local) {
        assert(r_d != NULL);
        //Lazy allocation
        if (l_d == NULL) {
            l_d = malloc(s);
        }
        cudaMemcpy(l_d, r_d, s, cudaMemcpyDeviceToHost);
        clean_local = true;
    }
}

void cuarray::exile() {
    //Lazy data movement
    if (!clean_remote) {
        assert(l_d != NULL);
        //Lazy allocation
        if (r_d == NULL) {
            cudaMalloc(&r_d, s);
        }
        cudaMemcpy(r_d, l_d, s, cudaMemcpyHostToDevice);
        clean_remote = true;
    }
}

seq cuarray::get_local_r() {
    retrieve();
    return make_pair(l_d, n);
}

seq cuarray::get_local_w() {
    retrieve();
    clean_remote = false;
    return make_pair(l_d, n);
}

seq cuarray::get_remote_r() {
    exile();
    return make_pair(r_d, n);
}

seq cuarray::get_remote_w() {
    exile();
    clean_local = false;
    return make_pair(r_d, n);
}

