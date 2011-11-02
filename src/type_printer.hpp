#pragma once

#include <stack>
#include "type.hpp"
#include "monotype.hpp"
#include "polytype.hpp"
#include "ctype.hpp"

namespace backend
{

class repr_type_printer
    : public boost::static_visitor<>
{
public:
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
class ctype_printer
    : public boost::static_visitor<>
{
private:
    std::stack<bool> m_need_space;
public:
    ctype_printer(std::ostream &os);
    
    void operator()(const monotype_t &mt);
    
    void operator()(const sequence_t &st);
    
    void operator()(const cuarray_t &ct);
    
    void operator()(const polytype_t &pt);
    
    void operator()(const templated_t &tt);
    
    std::ostream &m_os;

    void sep() const;
    
protected:
    void open() const;
    
    void close() const;
};
}
}
