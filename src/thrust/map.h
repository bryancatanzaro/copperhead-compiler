#pragma once
#include "../cudata/cudata.h"

#include <thrust/transform.h>
#include <thrust/detail/type_traits.h>
#include "transformed_sequence.h"
#include "zipped_sequence.h"


template<typename F,
         typename Seq0>
transformed_sequence<F, thrust::tuple<Seq0> >
map1(const F& fn,
    Seq0& x0) {
    return transformed_sequence<F, thrust::tuple<Seq0> >(fn, thrust::make_tuple(x0));
}

template<typename F,
         typename Seq0,
         typename Seq1>
transformed_sequence<F, thrust::tuple<Seq0, Seq1> >
map2(const F& fn,
    Seq0& x0,
    Seq1& x1) {
    return transformed_sequence<F, thrust::tuple<Seq0, Seq1> >(fn, thrust::make_tuple(x0, x1));
}

template<typename F,
         typename Seq0,
         typename Seq1,
         typename Seq2>
transformed_sequence<F, thrust::tuple<Seq0, Seq1, Seq2> >
map3(const F& fn,
    Seq0& x0,
    Seq1& x1,
    Seq2& x2) {
    return transformed_sequence<F, thrust::tuple<Seq0, Seq1, Seq2> >(fn, thrust::make_tuple(x0, x1, x2));
}

template<typename F,
         typename Seq0,
         typename Seq1,
         typename Seq2,
         typename Seq3>
transformed_sequence<F, thrust::tuple<Seq0, Seq1, Seq2, Seq3> >
map4(const F& fn,
    Seq0& x0,
    Seq1& x1,
    Seq2& x2,
    Seq3& x3) {
    return transformed_sequence<F, thrust::tuple<Seq0, Seq1, Seq2, Seq3> >(fn, thrust::make_tuple(x0, x1, x2, x3));
}
