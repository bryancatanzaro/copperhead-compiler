#pragma once

#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/reverse_iterator.h>

template<typename F, typename Seq>
sp_cuarray
scan(const F& fn, Seq& x) {
    typedef typename F::result_type T;
    sp_cuarray result_ary = make_remote<T>(x.size());
    stored_sequence<T> result = get_remote_w<T>(result_ary);
    thrust::inclusive_scan(x.begin(),
                           x.end(),
                           result.begin(),
                           fn);
    return result_ary;
}

template<typename F, typename Seq>
sp_cuarray
rscan(const F& fn, Seq& x) {
    typedef typename F::result_type T;
    typedef typename thrust::reverse_iterator<typename Seq::iterator_type> iterator_type;
    iterator_type drbegin(x.end());
    iterator_type drend(x.begin());
    sp_cuarray result_ary = make_remote<T>(x.size());
    stored_sequence<T> result = get_remote_w<T>(result_ary);
    thrust::reverse_iterator<thrust::device_ptr<T> > orbegin(result.end());
    thrust::inclusive_scan(drbegin, drend, orbegin, fn);
    return result_ary;
}

// template<typename F, typename T>
// sp_cuarray_var exscan(F& fn,
//                       stored_sequence<T>& x) {
//     sp_cuarray_var result_ary = make_remote<T>(x.size());
//     stored_sequence<T> result = get_remote_w<T>(result_ary);
//     thrust::exclusive_scan(extract_device_begin(x),
//                            extract_device_end(x),
//                            extract_device_begin(result),
//                            fn);
//     return result_ary;
// }


// template<typename F, typename T>
// sp_cuarray_var exrscan(F& fn,
//                      stored_sequence<T>& x) {
//     typename thrust::device_vector<T>::reverse_iterator drbegin(
//         extract_device_end(x));
//     typename thrust::device_vector<T>::reverse_iterator drend(
//         extract_device_begin(x));
//     sp_cuarray_var result_ary = make_remote<T>(x.size());
//     stored_sequence<T> result = get_remote_w<T>(result_ary);
//     typename thrust::device_vector<T>::reverse_iterator orbegin(
//         extract_device_end(result));
//     thrust::exclusive_scan(drbegin, drend, orbegin, fn);
// }
