#pragma once

#include "../backend.hpp"
#include <vector>
#include <memory>
#include <ostream>
#include <boost/iterator/indirect_iterator.hpp>

#include "type.hpp"
#include "monotype.hpp"
namespace backend {



class expression
    : public node
{
protected:
    std::shared_ptr<type_t> m_type;

    template<typename Derived>
    expression(Derived &self)
        : node(self), m_type(void_mt)
        {}
public:
    const type_t& type(void) const {
        return *m_type;
    }
    void set_type(const std::shared_ptr<type_t>& in) {
        m_type = in;
    }
};

class literal
    : public expression
{
protected:
    template<typename Derived>
    literal(Derived &self)
        : expression(self)
        {}

};

class number
    : public literal
{
public:
    number(const std::string &val)
        : literal(*this),
          m_val(val)
        {}
    inline const std::string val() const {
        return m_val;
    }
private:
    std::string m_val;
};

class name
    : public literal
{   
public:
    name(const std::string &val)
        : literal(*this),
          m_val(val)
        {}
    inline const std::string id() const {
        return m_val;
    }
private:
    std::string m_val;
};

class tuple
    : public expression
{
public:
    tuple(std::vector<std::shared_ptr<expression> > &&values)
        : expression(*this),
          m_values(std::move(values))
        {}
private:
    std::vector<std::shared_ptr<expression> > m_values;
public:
    typedef decltype(boost::make_indirect_iterator(m_values.cbegin())) const_iterator;
    const_iterator begin() const {
        return boost::make_indirect_iterator(m_values.cbegin());
    }

    const_iterator end() const {
        return boost::make_indirect_iterator(m_values.cend());
    }
};

class apply
    : public expression
{
private:
    std::shared_ptr<name> m_fn;
    std::shared_ptr<tuple> m_args;
public:
    apply(const std::shared_ptr<name> &fn,
          const std::shared_ptr<tuple> &args)
        : expression(*this),
          m_fn(fn), m_args(args)
        {}
    inline const name &fn(void) const {
        return *m_fn;
    }
    inline const tuple &args(void) const {
        return *m_args;
    }
};

class lambda
    : public expression
{
private:
    std::shared_ptr<tuple> m_args;
    std::shared_ptr<expression> m_body;
public:
    lambda(const std::shared_ptr<tuple> &args,
           const std::shared_ptr<expression> &body)
        : expression(*this),
          m_args(args), m_body(body)
        {}
    inline const tuple &args(void) const {
        return *m_args;
    }
    inline const expression &body(void) const {
        return *m_body;
    }
};

class closure
    : public expression
{
public:
    closure()
        : expression(*this)
        {}
};

class conditional
    : public expression
{
public:
    conditional()
        : expression(*this)
        {}
};

}
    

