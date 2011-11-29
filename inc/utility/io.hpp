#pragma once
#include <string>


namespace backend {
/*!
  \addtogroup utilities
  @{
*/

//! Prints an iterable object
/*! A flexible method of printing an iterable object.
  
  \param o Stream to print to.
  \param i Iterable object. Must have a .begin() and a .end() method
  which return the appropriate iterators.
  \param sep Separator string which will be placed in between all
  elements of the Iterable.  For example, ",".
  \param el Prefix for each element.
  \param open String to open printing with. For example, "(".
  \param close String to clos printing with. For example, ")".
*/
template<typename O, typename I>
void print_iterable(O& o, I& i,
                    std::string sep=std::string(","),
                    std::string el=std::string(""),
                    std::string open=std::string(""),
                    std::string close=std::string(""))
{
    o << open;
    for(auto it = i.begin();
        it != i.end();
        it++) {
        o << el << *it;
        if (std::next(it) != i.end())
            o << sep;
    }
    o << close;
}
/*!
  @}
*/


}

