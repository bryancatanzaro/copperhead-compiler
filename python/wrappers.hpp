#pragma once
#include "expression.hpp"
#include "statement.hpp"
#include "cppnode.hpp"

namespace backend {
class apply_wrap
    : public apply
{
public:
    apply_wrap(const std::shared_ptr<name> &fn,
               const std::shared_ptr<tuple> &args)
        : apply(fn, args)
    {}
    inline const std::shared_ptr<name> p_fn(void) const {
        return m_fn;
    }
    inline const std::shared_ptr<tuple> p_args(void) const {
        return m_args;
    }
};

class lambda_wrap
    : public lambda
{
public:
    lambda_wrap(const std::shared_ptr<tuple> &args,
                const std::shared_ptr<expression> &body)
        : lambda(args, body)
    {}
    inline const std::shared_ptr<tuple> p_args(void) const {
        return m_args;
    }
    inline const std::shared_ptr<expression> p_body(void) const {
        return m_body;
    }
};

class tuple_wrap
    : public backend::tuple
{
public:
    tuple_wrap(std::vector<std::shared_ptr<expression> > &&values)
        : backend::tuple(std::move(values))
    {}
    tuple_wrap()
        : tuple({})
    {}
    typedef decltype(m_values.cbegin()) const_ptr_iterator;
    const_ptr_iterator p_begin() const {
        return m_values.cbegin();
    }
    const_ptr_iterator p_end() const {
        return m_values.cend();
    }
};

class ret_wrap
    : public ret
{
public:
    ret_wrap(const std::shared_ptr<expression> &val)
        : ret(val)
    {}
    inline std::shared_ptr<expression> p_val(void) const {
        return m_val;
    }
};

class bind_wrap
    : public bind
{
public:
    bind_wrap(const std::shared_ptr<expression> &lhs,
              const std::shared_ptr<expression> &rhs)
        : bind(lhs, rhs)
    {}
    inline const std::shared_ptr<expression> p_lhs(void) const {
        return m_lhs;
    }
    inline const std::shared_ptr<expression> p_rhs(void) const {
        return m_rhs;
    }
};

class procedure_wrap
    : public procedure
{
public:
    procedure_wrap(const std::shared_ptr<name> &id,
                   const std::shared_ptr<tuple> &args,
                   const std::shared_ptr<suite> &stmts)
        : procedure(id, args, stmts)
    {}
    inline const std::shared_ptr<name> p_id(void) const {
        return m_id;
    }
    inline const std::shared_ptr<tuple> p_args(void) const {
        return m_args;
    }
    inline const std::shared_ptr<suite> p_stmts(void) const {
        return m_stmts;
    }
};    

class suite_wrap
    : public suite
{
public:
    suite_wrap(std::vector<std::shared_ptr<statement> > &&stmts)
        : suite(std::move(stmts))
    {}
    suite_wrap()
        : suite({})
    {}
    typedef decltype(m_stmts.cbegin()) const_ptr_iterator;
    const_ptr_iterator p_begin() const {
        return m_stmts.cbegin();
    }

    const_ptr_iterator p_end() const {
        return m_stmts.cend();
    }
};

class structure_wrap
    : public structure
{
public:
    structure_wrap(const std::shared_ptr<name> &name,
                   const std::shared_ptr<suite> &stmts)
        : structure(name, stmts)
    {}

    inline const std::shared_ptr<name> p_id(void) const {
        return m_id;
    }

    inline const std::shared_ptr<suite> p_stmts(void) const {
        return m_stmts;
    }

};

}


