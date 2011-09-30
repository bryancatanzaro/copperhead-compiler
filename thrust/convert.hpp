#pragma once
#include "../cudata/cudata.h"
#include <thrust/device_ptr.h>

template<typename T>
thrust::device_ptr<T> extract_device_begin(stored_sequence<T> &x) {
    return thrust::device_ptr<T>(x.data);
}

template<typename T>
thrust::device_ptr<T> extract_device_end(stored_sequence<T> &x) {
    return thrust::device_ptr<T>(x.data + x.length);
}



