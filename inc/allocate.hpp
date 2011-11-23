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

/*!
  \file   allocate.hpp
  \brief  The declaration of the \p allocate rewrite pass.
  
  
*/


namespace backend {

/*! \p allocate is a rewrite pass that inserts memory allocation for
  temporary variables and results.
*/
class allocate
    : public copier
{
private:
    const std::string& m_entry_point;
    bool m_in_entry;
    std::vector<std::shared_ptr<statement> > m_allocations;
public:

/*!   
  \param entry_point The name of the entry point procedure 
*/
    allocate(const std::string& entry_point);
    using copier::operator();

//! Rewrite rule for \p procedure nodes

    result_type operator()(const procedure &n);

//! Rewrite rule for \p bind nodes
    result_type operator()(const bind &n);
};

}
