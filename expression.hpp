#pragma once

#include "node.hpp"
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
    std::shared_ptr<type_t> m_ctype;
    template<typename Derived>
    expression(Derived &self,
               std::shared_ptr<type_t> type =
               std::shared_ptr<type_t>(new void_mt()),
               std::shared_ptr<type_t> ctype =
               std::shared_ptr<type_t>(new void_mt()))
        : node(self), m_type(type), m_ctype(ctype)
        {}
public:
    const type_t& type(void) const {
        return *m_type;
    }
    const type_t& ctype(void) const {
        return *m_ctype;
    }
};

class literal
    : public expression
{
protected:
    template<typename Derived>
    literal(Derived &self,
            std::shared_ptr<type_t> type =
            std::shared_ptr<type_t>(new void_mt()),
            std::shared_ptr<type_t> ctype =
            std::shared_ptr<type_t>(new void_mt()))
        : expression(self, type, ctype)
        {}

};

class number
    : public literal
{
public:
    number(const std::string &val,
           std::shared_ptr<type_t> type =
           std::shared_ptr<type_t>(new void_mt()),
           std::shared_ptr<type_t> ctype =
           std::shared_ptr<type_t>(new void_mt()))
        : literal(*this, type, ctype),
          m_val(val)
        {}
    inline const std::string val() const {
        return m_val;
    }
protected:
    std::string m_val;
};

class name
    : public literal
{   
public:
    name(const std::string &val,
         std::shared_ptr<type_t> type =
         std::shared_ptr<type_t>(new void_mt()),
         std::shared_ptr<type_t> ctype =
         std::shared_ptr<type_t>(new void_mt()))
        : literal(*this, type, ctype),
          m_val(val)
        {}
    inline const std::string id() const {
        return m_val;
    }
protected:
    std::string m_val;
};

class tuple
    : public expression
{
public:
    tuple(std::vector<std::shared_ptr<expression> > &&values,
          std::shared_ptr<type_t> type =
          std::shared_ptr<type_t>(new void_mt()),
          std::shared_ptr<type_t> ctype =
          std::shared_ptr<type_t>(new void_mt()))
        : expression(*this, type, ctype),
          m_values(std::move(values))
        {}
protected:
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
protected:
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
protected:
    std::shared_ptr<tuple> m_args;
    std::shared_ptr<expression> m_body;
public:
    lambda(const std::shared_ptr<tuple> &args,
           const std::shared_ptr<expression> &body,
           std::shared_ptr<type_t> type =
           std::shared_ptr<type_t>(new void_mt()),
           std::shared_ptr<type_t> ctype =
           std::shared_ptr<type_t>(new void_mt()))
        : expression(*this, type, ctype),
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
    

