#pragma once
#include "node.hpp"
#include "type.hpp"
#include "ctype.hpp"
#include "utility/isinstance.hpp"
#include "utility/markers.hpp"
#include "utility/snippets.hpp"
#include "py_printer.hpp"
#include "copier.hpp"

namespace backend {


class wrap
    : public copier
{
private:
    const std::string& m_entry_point;
    bool m_wrapping;
    std::shared_ptr<procedure> m_wrapper;
public:
    wrap(const std::string& entry_point);
    
    using copier::operator();

    result_type operator()(const procedure &n);

    result_type operator()(const ret& n);
    
    result_type operator()(const suite&n);
};

}
