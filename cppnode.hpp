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
              const std::shared_ptr<suite> &stmts)
        : statement(*this),
          m_id(name),
          m_stmts(stmts)
        {}
    inline const name& id(void) const {
        return *m_id;
    }
    inline const suite& stmts(void) const {
        return *m_stmts;
    }
};

class templated_name
    : public name
{
protected:
    std::shared_ptr<ctype::tuple_t> m_template_types;
public:
    templated_name(const std::string &id,
                   const std::shared_ptr<ctype::tuple_t> &template_types,
                   const std::shared_ptr<type_t> type =
                   std::shared_ptr<type_t>(new void_mt()),
                   const std::shared_ptr<ctype::type_t> ctype =
                   std::shared_ptr<ctype::type_t>(new ctype::void_mt()))
        : name(*this, id, type, ctype),
          m_template_types(template_types) {}
    const ctype::tuple_t& template_types() const {
        return *m_template_types;
    }
};

class include
    : public statement
{
protected:
    const std::shared_ptr<name> m_id;
    const char m_open;
    const char m_close;
public:
    include(const std::shared_ptr<name> &id,
            const char open = '\"',
            const char close = '\"') : statement(*this),
                                       m_id(id),
                                       m_open(open),
                                       m_close(close) {}
    const name& id() const {
        return *m_id;
    }
    const char& open() const {
        return m_open;
    }
    const char& close() const {
        return m_close;
    }
};

}
