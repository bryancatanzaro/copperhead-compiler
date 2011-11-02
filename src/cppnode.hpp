#pragma once

#include "node.hpp"
#include "expression.hpp"
#include "statement.hpp"
#include <vector>
#include <memory>

namespace backend {

class structure
    : public statement
{
protected:
    std::shared_ptr<name> m_id;
    std::shared_ptr<suite> m_stmts;
public:
    structure(const std::shared_ptr<name> &name,
              const std::shared_ptr<suite> &stmts);
    const name& id(void) const;
    const suite& stmts(void) const;
};

class templated_name
    : public name
{
protected:
    std::shared_ptr<ctype::tuple_t> m_template_types;
public:
    templated_name(const std::string &id,
                   const std::shared_ptr<ctype::tuple_t> &template_types,
                   const std::shared_ptr<type_t>& type =
                   std::shared_ptr<type_t>(new void_mt()),
                   const std::shared_ptr<ctype::type_t>& ctype =
                   std::shared_ptr<ctype::type_t>(new ctype::void_mt()));
    const ctype::tuple_t& template_types() const;
};

class include
    : public statement
{
protected:
    const std::shared_ptr<literal> m_id;
    const char m_open;
    const char m_close;
public:
    include(const std::shared_ptr<literal> &id,
            const char open = '\"',
            const char close = '\"');
    const literal& id() const;
    const char& open() const;
    const char& close() const;
};

class typedefn
    : public statement
{
protected:
    const std::shared_ptr<ctype::type_t> m_origin;
    const std::shared_ptr<ctype::type_t> m_rename;
public:
    typedefn(const std::shared_ptr<ctype::type_t> origin,
             const std::shared_ptr<ctype::type_t> rename);
    const ctype::type_t& origin() const;
    const ctype::type_t& rename() const;
};

}
