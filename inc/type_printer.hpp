#pragma once

#include <stack>
#include "type.hpp"
#include "monotype.hpp"
#include "polytype.hpp"
#include "ctype.hpp"

namespace backend
{
//! Prints a detailed representation of a \ref backend::type_t "type_t" object
/*! This is modeled after the Python repr() method for producing
*strings: it is intended to be complete and detailed rather than
*human readable and elegant. 
*/
class repr_type_printer
    : public boost::static_visitor<>
{
public:
    //! Constructor
/*! 
  
  \param os The stream to be printed to
*/
    repr_type_printer(std::ostream &os);
    
    void operator()(const monotype_t &mt);
    
    void operator()(const polytype_t &pt);
    
    std::ostream &m_os;

    void sep() const;
    
protected:
    void open() const;
    
    void close() const;
    
};

namespace ctype {
//! Prints C++ implementation types
/*! This printer is intended to print C++ implementation types
*directly, with syntax that complies with C++ itself. 
*/

class ctype_printer
    : public boost::static_visitor<>
{
private:
    std::stack<bool> m_need_space;
public:
    //! Constructor
/*! 
  \param os Stream to be printed to
*/
    ctype_printer(std::ostream &os);
    
    void operator()(const monotype_t &mt);
    
    void operator()(const sequence_t &st);
    
    void operator()(const cuarray_t &ct);
    
    void operator()(const polytype_t &pt);
        
    std::ostream &m_os;

    void sep() const;
    
protected:
    void open() const;
    
    void close() const;
};
}
}
