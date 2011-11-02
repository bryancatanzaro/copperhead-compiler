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
    ret(const std::shared_ptr<expression> &val);
protected:
    const std::shared_ptr<expression> m_val;
public:
    const expression& val(void) const;
};

class bind
    : public statement
{
public:
    bind(const std::shared_ptr<expression> &lhs,
         const std::shared_ptr<expression> &rhs);
protected:
    const std::shared_ptr<expression> m_lhs;
    const std::shared_ptr<expression> m_rhs;

public:
    const expression& lhs(void) const;
    const expression& rhs(void) const;
};

class call
    : public statement
{
protected:
    const std::shared_ptr<apply> m_sub;
public:
    call(const std::shared_ptr<apply> &n);
    const apply& sub(void) const;
};
        

class procedure
    : public statement
{
public:
    procedure(const std::shared_ptr<name> &id,
              const std::shared_ptr<tuple> &args,
              const std::shared_ptr<suite> &stmts,
              const std::shared_ptr<type_t> &type =
              std::shared_ptr<type_t>(new void_mt()),
              const std::shared_ptr<ctype::type_t> &ctype =
              std::shared_ptr<ctype::type_t>(new ctype::void_mt()),
              const std::string &place =
              "__device__");
protected:
    const std::shared_ptr<name> m_id;
    const std::shared_ptr<tuple> m_args;
    const std::shared_ptr<suite> m_stmts;
    std::shared_ptr<type_t> m_type;
    std::shared_ptr<ctype::type_t> m_ctype;
    const std::string m_place;
public:
    const name& id(void) const;

    const tuple& args(void) const;

    const suite& stmts(void) const;
    
    const type_t& type(void) const;
    
    const ctype::type_t& ctype(void) const;

    const std::string& place(void) const;
    
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
                std::shared_ptr<suite> orelse);
    
    const expression& cond(void) const;
    
    const suite& then(void) const;
    
    const suite& orelse(void) const;
    
};



class suite 
    : public node
{
public:
    suite(std::vector<std::shared_ptr<statement> > &&stmts);
protected:
    const std::vector<std::shared_ptr<statement> > m_stmts;
public:
    typedef decltype(boost::make_indirect_iterator(m_stmts.cbegin())) const_iterator;
    const_iterator begin() const;

    const_iterator end() const;
};


}

