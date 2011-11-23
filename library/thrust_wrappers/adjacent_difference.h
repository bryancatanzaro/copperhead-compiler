#pragma once

#include <thrust/adjacent_difference.h>

//XXX Need to look at F::result_type instead of Seq::value_type
template<typename F, typename Seq>
boost::shared_ptr<cuarray<typename Seq::value_type> >
adjacent_difference(const F& fn, Seq& x) {
    typedef typename Seq::value_type T;
    boost::shared_ptr<cuarray<T> > result_ary = make_remote<T>(x.size());
    stored_sequence<T> result = get_remote_w<T>(result_ary);
    thrust::adjacent_difference(x.begin(),
                                x.end(),
                                result.begin(),
                                fn);
    return result_ary;
}
