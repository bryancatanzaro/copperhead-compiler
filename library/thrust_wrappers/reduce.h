#pragma once

#include <thrust/reduce.h>

template<typename F, typename Seq>
typename F::result_type
reduce(const F& fn, Seq& x, const typename F::result_type& p) {
    return thrust::reduce(x.begin(), x.end(), p, fn);
}

template<typename Seq>
typename Seq::value_type
sum(Seq& x) {
    return thrust::reduce(x.begin(), x.end());
}
