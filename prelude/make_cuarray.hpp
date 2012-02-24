#pragma once
#include "sequence.hpp"

//XXX Use std::shared_ptr once nvcc can pass it through
#define BOOST_SP_USE_SPINLOCK
#include <boost/shared_ptr.hpp>

/* This file allows construction of cuarray objects
   even when cuarray can't be instantiated by the compiler.

   nvcc cannot yet instantiate cuarray, for two reasons:
   * parts of cuarray use c++11 move semantics
   * parts of cuarray use c++11 std::shared_ptr

   Separating the interface for cuarray in this manner
   allows code compiled by nvcc to construct cuarray objects
   without needing to instantiate them directly.
*/

//Forward declaration
class cuarray;

typedef boost::shared_ptr<cuarray> sp_cuarray;

template<typename T>
sp_cuarray make_cuarray(size_t s);


