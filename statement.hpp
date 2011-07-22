#pragma once

#include "../backend.hpp"
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
    ret(const std::shared_ptr<expression> & val)
        : statement(*this),
          m_val(val)
        {}
private:
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
private:
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

class procedure
    : public statement
{
public:
    procedure(const std::shared_ptr<name> &id,
              const std::shared_ptr<tuple> &args,
              const std::shared_ptr<suite> &stmts)
        : statement(*this),
          m_id(id), m_args(args), m_stmts(stmts)
        {}
private:
    const std::shared_ptr<name> m_id;
    const std::shared_ptr<tuple> m_args;
    const std::shared_ptr<suite> m_stmts;

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
    
};

class suite 
    : public node
{
public:
    suite(std::vector<std::shared_ptr<statement> > &&stmts)
        : node(*this),
          m_stmts(std::move(stmts))
        {}
private:
    std::vector<std::shared_ptr<statement> > m_stmts;
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

