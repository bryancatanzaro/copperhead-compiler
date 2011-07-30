#pragma once

#include "../backend.hpp"
#include "statement.hpp"
#include <vector>
#include <memory>

namespace backend {

class structure
    : public statement
{
private:
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

}
