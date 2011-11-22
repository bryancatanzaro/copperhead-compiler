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
    std::shared_ptr<suite> m_host;
    std::shared_ptr<suite> m_device;
    //Declaration for the entry point function, grabbed from wrapper
    std::shared_ptr<procedure> m_wrap_decl;
public:
    bpl_wrap(const std::string& entry_point,
             const registry& reg);
    using copier::operator();
    result_type operator()(const suite& n);
    result_type operator()(const procedure& n);
    std::shared_ptr<suite> p_host_code() const {
        return m_host;
    }
    std::shared_ptr<suite> p_device_code() const {
        return m_device;
    }
};
}
