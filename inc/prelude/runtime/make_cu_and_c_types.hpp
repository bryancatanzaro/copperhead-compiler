#pragma once


//These constructors are isolated to avoid showing C++11 stuff to nvcc
namespace copperhead {

//forward declaration
struct cu_and_c_types;

cu_and_c_types* make_type_holder(int);
cu_and_c_types* make_type_holder(long);
cu_and_c_types* make_type_holder(float);
cu_and_c_types* make_type_holder(double);
cu_and_c_types* make_type_holder(bool);

}
