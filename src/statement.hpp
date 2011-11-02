#pragma once

#include "node.hpp"
#include "type.hpp"
#include "monotype.hpp"
#include "ctype.hpp"
#include <vector>
#include <memory>

namespace backend {

class statement
    : public node
{
protected:
    template<typename Derived>
    statement(Derived &self)
        : node(self)
        {}
};

class ret
    : public statement
{
public:
    ret(const std::shared_ptr<expression> &val)
        : statement(*this),
          m_val(val)
    {}
protected:
    const std::shared_ptr<expression> m_val;
public:
    inline const expression& val(void) const {
        return *m_val;
    }
};

class bind
    : public statement
{
public:
    bind(const std::shared_ptr<expression> &lhs,
         const std::shared_ptr<expression> &rhs)
        : statement(*this),
          m_lhs(lhs), m_rhs(rhs)
        {}
protected:
    const std::shared_ptr<expression> m_lhs;
    const std::shared_ptr<expression> m_rhs;

public:
    inline const expression& lhs(void) const {
        return *m_lhs;
    }
    inline const expression& rhs(void) const {
        return *m_rhs;
    }
};

class call
    : public statement
{
protected:
    const std::shared_ptr<apply> m_sub;
public:
    call(const std::shared_ptr<apply> &n)
        : statement(*this), m_sub(n) {}
    inline const apply& sub(void) const {
        return *m_sub;
    }
};
        

class procedure
    : public statement
{
public:
    procedure(const std::shared_ptr<name> &id,
              const std::shared_ptr<tuple> &args,
              const std::shared_ptr<suite> &stmts,
              const std::shared_ptr<type_t> type =
              std::shared_ptr<type_t>(new void_mt()),
              std::shared_ptr<ctype::type_t> ctype =
              std::shared_ptr<ctype::type_t>(new ctype::void_mt()),
              const std::string &place =
              "__device__")
        : statement(*this),
          m_id(id), m_args(args), m_stmts(stmts), m_type(type),
          m_ctype(ctype), m_place(place)
        {}
protected:
    const std::shared_ptr<name> m_id;
    const std::shared_ptr<tuple> m_args;
    const std::shared_ptr<suite> m_stmts;
    std::shared_ptr<type_t> m_type;
    std::shared_ptr<ctype::type_t> m_ctype;
    const std::string m_place;
public:
    inline const name& id(void) const {
        return *m_id;
    }

    inline const tuple& args(void) const {
        return *m_args;
    }

    inline const suite& stmts(void) const {
        return *m_stmts;
    }
    const type_t& type(void) const {
        return *m_type;
    }
    const ctype::type_t& ctype(void) const {
        return *m_ctype;
    }

    const std::string& place(void) const {
        return m_place;
    }
    
};

class conditional
    : public statement
{
protected:
    std::shared_ptr<expression> m_cond;
    std::shared_ptr<suite> m_then;
    std::shared_ptr<suite> m_orelse;
    
public:
    conditional(std::shared_ptr<expression> cond,
                std::shared_ptr<suite> then,
                std::shared_ptr<suite> orelse)
        : statement(*this), m_cond(cond),
          m_then(then), m_orelse(orelse)
        {}
    inline const expression& cond(void) const {
        return *m_cond;
    }
    inline const suite& then(void) const {
        return *m_then;
    }
    inline const suite& orelse(void) const {
        return *m_orelse;
    }
};



class suite 
    : public node
{
public:
    suite(std::vector<std::shared_ptr<statement> > &&stmts)
        : node(*this),
          m_stmts(std::move(stmts))
        {}
protected:
    const std::vector<std::shared_ptr<statement> > m_stmts;
public:
    typedef decltype(boost::make_indirect_iterator(m_stmts.cbegin())) const_iterator;
    const_iterator begin() const {
        return boost::make_indirect_iterator(m_stmts.cbegin());
    }

    const_iterator end() const {
        return boost::make_indirect_iterator(m_stmts.cend());
    }
};


}
