#pragma once
#include <cassert>
#include <sstream>

#include "import/library.hpp"
#include "environment.hpp"
#include "py_printer.hpp"
#include "type_printer.hpp"
#include "utility/markers.hpp"
#include "utility/isinstance.hpp"
#include "utility/up_get.hpp"

namespace backend
{


class cuda_printer
    : public py_printer
{
private:
    const std::string& entry;
    environment<std::string> declared;
    ctype::ctype_printer tp;
    bool m_in_rhs;
public:
    cuda_printer(const std::string &entry_point,
                 const registry& globals,
                 std::ostream &os);
    
    using backend::py_printer::operator();

    void operator()(const backend::name &n);

    void operator()(const templated_name &n);
    
    void operator()(const literal &n);

    void operator()(const tuple &n);

    void operator()(const apply &n);
    
    void operator()(const closure &n);
    
    void operator()(const conditional &n);
    
    void operator()(const ret &n);
    
    void operator()(const bind &n);
    
    void operator()(const call &n);
    
    void operator()(const procedure &n);
    
    void operator()(const suite &n);

    void operator()(const structure &n);
    
    void operator()(const include &n);
    
    void operator()(const typedefn &n);
    
    void operator()(const std::string &s);
    
    template<typename T>
    void operator()(const std::vector<T> &v) {
        detail::list(*this, v);
    }
};


}
