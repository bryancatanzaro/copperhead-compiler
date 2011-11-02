#pragma once
#include "../cudata/cudata.h"

template<typename SeqX,
         typename SeqI>
boost::shared_ptr<cuarray<typename SeqX::value_type> > permute(
    SeqX& x,
    SeqI& i) {
    typedef typename SeqX::value_type T;
    boost::shared_ptr<cuarray<T> > result_ary = make_remote<T>(x.size());
    stored_sequence<T> result = get_remote_w<T>(result_ary);
    thrust::scatter(x.begin(),
                    x.end(),
                    i.begin(),
                    result.begin());
    return result_ary;
}
