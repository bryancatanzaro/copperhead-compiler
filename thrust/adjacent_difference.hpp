#pragma once
#include "../cudata/cudata.h"
#include "convert.hpp"

#include <thrust/adjacent_difference.h>

template<typename F, typename T>
sp_cuarray_var adjacent_difference(F fn, stored_sequence<T>& x) {
    sp_cuarray_var result_ary = make_remote<T>(x.size());
    stored_sequence<T> result = get_remote_w<T>(result_ary);
    thrust::adjacent_difference(extract_device_begin(x),
                                extract_device_end(x),
                                extract_device_begin(result),
                                fn);
    return result_ary;
}
