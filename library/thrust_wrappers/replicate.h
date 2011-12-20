#pragma once
#include "constant_sequence.h"

template<typename I, typename T>
constant_sequence<T> replicate(const T& val,
                               const I& amount) {
    return constant_sequence<T>(val, amount);
}
