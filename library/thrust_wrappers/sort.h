#pragma once

#include <thrust/sort.h>

#include "../prelude/functors.h"

template<typename F, typename Seq>
boost::shared_ptr<cuarray<typename Seq::value_type> >
sort(const F& fn, Seq& x) {
    typedef typename Seq::value_type T;

    //Copy for value semantics (since thrust sort is "in-place")
    boost::shared_ptr<cuarray<T> > result_ary = make_remote<T>(x.size());
    stored_sequence<T> result = get_remote_w<T>(result_ary);
    thrust::copy(x.begin(),
                 x.end(),
                 result.begin());

    thrust::sort(result.begin(),
                 result.end(),
                 fn);
    return result_ary;
}


template<typename Seq>
boost::shared_ptr<cuarray<typename Seq::value_type> >
sort(const fn_cmp_lt<typename Seq::value_type>& fn, Seq& x) {
    typedef typename Seq::value_type T;

    //Copy for value semantics (since thrust sort is "in-place")
    boost::shared_ptr<cuarray<T> > result_ary = make_remote<T>(x.size());
    stored_sequence<T> result = get_remote_w<T>(result_ary);
    thrust::copy(x.begin(),
                 x.end(),
                 result.begin());

    thrust::sort(result.begin(),
                 result.end(),
                 thrust::less<T>());
    return result_ary;
}

template<typename Seq>
boost::shared_ptr<cuarray<typename Seq::value_type> >
sort(const fn_cmp_gt<typename Seq::value_type>& fn, Seq& x) {
    typedef typename Seq::value_type T;

    //Copy for value semantics (since thrust sort is "in-place")
    boost::shared_ptr<cuarray<T> > result_ary = make_remote<T>(x.size());
    stored_sequence<T> result = get_remote_w<T>(result_ary);
    thrust::copy(x.begin(),
                 x.end(),
                 result.begin());

    thrust::sort(result.begin(),
                 result.end(),
                 thrust::greater<T>());
    return result_ary;
}
