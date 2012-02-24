#pragma once
#include "make_cuarray.hpp"

/* This file allows the creation of views of cuarray objects,
   even when cuarray objects can't be instantiated by the compiler.

   nvcc cannot yet instantiate cuarray, for two reasons:
   * parts of cuarray use c++11 move semantics
   * parts of cuarray use c++11 std::shared_ptr

   Since the view objects created from cuarray objects can be
   instantiated by all compilers, they can be used in nvcc-compiled
   code.
   
*/

template<typename S>
S make_sequence(sp_cuarray& in, bool local=true, bool write=true);

