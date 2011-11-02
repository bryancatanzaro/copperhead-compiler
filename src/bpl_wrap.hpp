#pragma once
#include "node.hpp"
#include "type.hpp"
#include "ctype.hpp"
#include "copier.hpp"
#include "utility/isinstance.hpp"
#include "utility/markers.hpp"
#include "utility/snippets.hpp"
#include "py_printer.hpp"
#include "import/library.hpp"

namespace backend {


class bpl_wrap
    : public copier
{
private:
    const std::string& m_entry_point;
    std::vector<std::shared_ptr<include> > m_includes;
    bool m_outer;
public:
    bpl_wrap(const std::string& entry_point,
             const registry& reg);
    using copier::operator();
    result_type operator()(const suite& n);
};
}
