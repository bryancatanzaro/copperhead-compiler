#pragma once

#include "index_sequence.h"

template<typename Seq0>
index_sequence indices(Seq0& x0) {
    return index_sequence(x0.size());
}
