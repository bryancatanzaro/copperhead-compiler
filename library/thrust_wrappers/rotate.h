#pragma once
#include "rotated_sequence.h"

template<typename Seq>
rotated_sequence<Seq> rotate(const Seq& src,
                             const typename Seq::index_type& amount) {
    return rotated_sequence<Seq>(src, amount);
}
