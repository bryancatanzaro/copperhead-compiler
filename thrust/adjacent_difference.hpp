#pragma once
#include "../cudata/cudata.h"
#include "convert.h"

#include <thrust/adjacent_difference.h>

template<typename T>
sp_cuarray_var adjacent_difference(stored_sequence<T>& x) {
    sp_cuarray_var result_ary = make_remote(x.size());
    stored_sequence<T> result = get_remote_w<T>(result_ary);
    thrust::adjacent_difference(extract_device_begin(x),
                                extract_device_end(x),
                                extract_device_begin(result));
    return result_ary;
}
