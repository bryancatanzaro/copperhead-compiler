#pragma once
#include <prelude/sequences/zipped_sequence.h>

namespace copperhead {
namespace detail {
/*! Compute the stored sequence type for an input sequence
  (which may not be stored in memory at all)
*/
template<typename Tag, typename T>
struct stored_sequence {
    typedef sequence<Tag, T> type;
};

template<typename Tag,
         typename T0,
         typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename T6,
         typename T7,
         typename T8,
         typename T9>
struct stored_sequence<Tag,
                       thrust::tuple<T0,
                                     T1,
                                     T2,
                                     T3,
                                     T4,
                                     T5,
                                     T6,
                                     T7,
                                     T8,
                                     T9> > {
    typedef thrust::tuple<T0, T1, T2, T3, T4,
                          T5, T6, T7, T8, T9> sub_type;
    typedef zipped_sequence<
        thrust::tuple<
            typename stored_sequence<Tag, T0>::type,
            typename stored_sequence<Tag, T1>::type,
            typename stored_sequence<Tag, T2>::type,
            typename stored_sequence<Tag, T3>::type,
            typename stored_sequence<Tag, T4>::type,
            typename stored_sequence<Tag, T5>::type,
            typename stored_sequence<Tag, T6>::type,
            typename stored_sequence<Tag, T7>::type,
            typename stored_sequence<Tag, T8>::type,
            typename stored_sequence<Tag, T9>::type > > type;
};

template<typename Tag>
struct stored_sequence<Tag,
                       thrust::null_type> {
    typedef thrust::null_type type;
};

}
}
