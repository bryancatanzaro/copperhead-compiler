#pragma once

#include "node.hpp"
#include <vector>
#include <memory>
#include <ostream>
#include <boost/iterator/indirect_iterator.hpp>

#include "type.hpp"
#include "monotype.hpp"
#include "ctype.hpp"
namespace backend {



class expression
    : public node
{
protected:
    std::shared_ptr<type_t> m_type;
    std::shared_ptr<ctype::type_t> m_ctype;
    template<typename Derived>
    expression(Derived &self,
               std::shared_ptr<type_t> type =
               std::shared_ptr<type_t>(new void_mt()),
               std::shared_ptr<ctype::type_t> ctype =
               std::shared_ptr<ctype::type_t>(new ctype::void_mt()))
        : node(self), m_type(type), m_ctype(ctype)
        {}
public:
    const type_t& type(void) const {
        return *m_type;
    }
    const ctype::type_t& ctype(void) const {
        return *m_ctype;
    }
};

class literal
    : public expression
{
protected:
    const std::string m_val;
public:
    template<typename Derived>
    literal(Derived &self,
            const std::string& val,
            std::shared_ptr<type_t> type =
            std::shared_ptr<type_t>(new void_mt()),
            std::shared_ptr<ctype::type_t> ctype =
            std::shared_ptr<ctype::type_t>(new ctype::void_mt()))
        : expression(self, type, ctype), m_val(val)
        {}
    literal(const std::string& val,
            std::shared_ptr<type_t> type =
            std::shared_ptr<type_t>(new void_mt()),
            std::shared_ptr<ctype::type_t> ctype =
            std::shared_ptr<ctype::type_t>(new ctype::void_mt()))
        : expression(*this, type, ctype), m_val(val)
        {}
    const std::string& id(void) const {
        return m_val;
    }

};

class name
    : public literal
{   
public:
    name(const std::string &val,
         std::shared_ptr<type_t> type =
         std::shared_ptr<type_t>(new void_mt()),
         std::shared_ptr<ctype::type_t> ctype =
         std::shared_ptr<ctype::type_t>(new ctype::void_mt()))
        : literal(*this, val, type, ctype)
        {}
    template<typename Derived>
    name(Derived& self, const std::string &val,
         std::shared_ptr<type_t> type,
         std::shared_ptr<ctype::type_t> ctype) :
        literal(self, val, type, ctype) {}
};

class tuple
    : public expression
{
public:
    tuple(std::vector<std::shared_ptr<expression> > &&values,
          std::shared_ptr<type_t> type =
          std::shared_ptr<type_t>(new void_mt()),
          std::shared_ptr<ctype::type_t> ctype =
          std::shared_ptr<ctype::type_t>(new ctype::void_mt()))
        : expression(*this, type, ctype),
          m_values(std::move(values))
        {}
protected:
    const std::vector<std::shared_ptr<expression> > m_values;
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
    const std::shared_ptr<name> m_fn;
    const std::shared_ptr<tuple> m_args;
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
    const std::shared_ptr<tuple> m_args;
    const std::shared_ptr<expression> m_body;
public:
    lambda(const std::shared_ptr<tuple> &args,
           const std::shared_ptr<expression> &body,
           std::shared_ptr<type_t> type =
           std::shared_ptr<type_t>(new void_mt()),
           std::shared_ptr<ctype::type_t> ctype =
           std::shared_ptr<ctype::type_t>(new ctype::void_mt()))
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
    

