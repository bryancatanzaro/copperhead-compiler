#pragma once
#include "shifted_sequence.h"

template<typename Seq>
shifted_sequence<Seq> shift(const Seq& src,
                            const typename Seq::index_type& amount,
                            const typename Seq::value_type& boundary) {
    return shifted_sequence<Seq>(src, amount, boundary);
}
