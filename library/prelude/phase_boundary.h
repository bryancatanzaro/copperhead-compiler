#pragma once
#include "cudata.h"
#include <thrust/copy.h>

template<typename Seq>
boost::shared_ptr<cuarray<typename Seq::value_type> >
phase_boundary(const Seq& in) {
    typedef typename Seq::value_type T;
    boost::shared_ptr<cuarray<T> > result_ary = make_remote<T>(in.size());
    stored_sequence<T> result = get_remote_w<T>(result_ary);
    thrust::copy(in.begin(), in.end(), result.begin());
    return result_ary;
}
