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
    bool m_in_struct;
    //! Prints template declarations
    /*! \tparam I Iterator type to template types
      \param begin Iterator to beginning of template types.
      \param end Iterator to end of template types.
    */
    template<typename I>
    void print_template_decl(const I& begin, const I& end) {
        m_os << "template<";
        for(auto i = begin;
            i != end;
            i++) {
            m_os << "typename ";
            boost::apply_visitor(tp, *i);
            if (std::next(i) != end) {
                m_os << ", ";
            }
        }
        m_os << ">" << std::endl;
    }
    //! Print return type of procedure
/*! 
  
  \param mt Monotype of procedure
  \param n The procedure node itself
*/
    void print_proc_return(const ctype::monotype_t& mt,
                           const procedure& n);

        
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

    void operator()(const namespace_block &n);
    
    void operator()(const std::string &s);
    
    template<typename T>
    void operator()(const std::vector<T> &v) {
        detail::list(*this, v);
    }
};


}
