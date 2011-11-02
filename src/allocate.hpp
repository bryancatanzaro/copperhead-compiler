#pragma once
#include "node.hpp"
#include "type.hpp"
#include "ctype.hpp"
#include "utility/isinstance.hpp"
#include "utility/markers.hpp"
#include "utility/snippets.hpp"
#include "py_printer.hpp"
#include "cuda_printer.hpp"
#include "copier.hpp"

namespace backend {


class allocate
    : public copier
{
private:
    const std::string& m_entry_point;
    bool m_in_entry;
    std::vector<std::shared_ptr<statement> > m_allocations;
public:
    allocate(const std::string& entry_point);
    using copier::operator();
    result_type operator()(const procedure &n);
    result_type operator()(const bind &n);
};

}
