#pragma once


//These constructors are isolated to avoid showing C++11 stuff to nvcc
namespace copperhead {

//forward declaration
struct type_holder;

namespace detail {

type_holder* make_type_holder();
void add_type(type_holder*, int);
void add_type(type_holder*, long);
void add_type(type_holder*, float);
void add_type(type_holder*, double);
void add_type(type_holder*, bool);

void begin(type_holder*);
void end_sequence(type_holder*);
void end_tuple(type_holder*);
void finalize_type(type_holder*);

}

}
