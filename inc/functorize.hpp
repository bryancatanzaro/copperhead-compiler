#pragma once
#include <string>
#include <set>
#include <sstream>
#include "rewriter.hpp"
#include "utility/isinstance.hpp"
#include "utility/markers.hpp"
#include "import/library.hpp"
#include "type_convert.hpp"

#include "type_printer.hpp"


namespace backend {

namespace detail {

class type_corresponder
    : public boost::static_visitor<> {
private:
    typedef std::map<std::string,
                     std::shared_ptr<type_t> > type_map;

    std::shared_ptr<type_t> m_working;
    type_map& m_corresponded;
    template<typename Type>
    static inline std::shared_ptr<type_t> get_type_ptr(const Type &n) {
        return std::const_pointer_cast<type_t>(n.shared_from_this());
    }
public:
    typedef std::map<std::string, std::shared_ptr<type_t> >::const_iterator iterator;
    type_corresponder(const std::shared_ptr<type_t>& input,
                      type_map& corresponded);
    
    iterator begin() const;
    
    iterator end() const;
    
    void operator()(const monotype_t &n);

    void operator()(const polytype_t &n);

    void operator()(const sequence_t &n);

    void operator()(const tuple_t &n);

    void operator()(const fn_t &n);

    template<typename N>
    void operator()(const N &n) {
        //Don't harvest correspondences from other types
    }

    
    
};
}


/*! \p A compiler pass to create function objects for all procedures
 *  except the entry point.
 */
class functorize
    : public rewriter
{
private:
    const std::string& m_entry_point;
    std::vector<result_type> m_additionals;
    std::set<std::string> m_fns;
    const registry& m_reg;


    typedef std::map<std::string,
                     std::shared_ptr<type_t> > type_map;

    type_map m_type_map;

    void make_type_map(const apply& n);

    std::shared_ptr<expression> instantiate_fn(const name& n,
                                               std::shared_ptr<type_t> p_t);
public:
    /*! \param entry_point The name of the entry point procedure
        \param reg The registry of functions the compiler knows about
     */
    functorize(const std::string& entry_point,
               const registry& reg);
    
    using rewriter::operator();

    result_type operator()(const apply &n);
    
    result_type operator()(const suite &n);
    
    result_type operator()(const procedure &n);
    
};

}
