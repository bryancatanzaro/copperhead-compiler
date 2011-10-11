#pragma once
#include "../cudata/cudata.h"
#include <thrust/copy.h>

template<typename Seq>
sp_cuarray_var phase_boundary_independent(const Seq& in) {
    typedef typename Seq::value_type T;
    sp_cuarray_var result_ary = make_remote<T>(in.size());
    stored_sequence<T> result = get_remote_w<T>(result_ary);
    thrust::copy(in.begin(), in.end(), result.begin());
    return result_ary;
}
