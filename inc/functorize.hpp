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
//! Matches two types
/*! This class is used to match two Copperhead types, to discover how
  they are related. This can be used for type propagation
*/
class type_corresponder
    : public boost::static_visitor<> {
private:
    typedef std::map<std::string,
                     std::shared_ptr<type_t> > type_map;

    std::shared_ptr<type_t> m_working;
    type_map& m_corresponded;
public:
    //! Constructor
/*! 
  
  \param input Input type to be matched
  \param corresponded Type map that will be amended with correspondence
  \return 
*/
    type_corresponder(const std::shared_ptr<type_t>& input,
                      type_map& corresponded);

    //! Harvest correspondence from a monotype_t
    void operator()(const monotype_t &n);
    //! Harvest correspondence from a polytype_t
    void operator()(const polytype_t &n);
    //! Harvest correspondence from a sequence_t
    void operator()(const sequence_t &n);
    //! Harvest correspondence from a tuple_t
    void operator()(const tuple_t &n);
    //! Harvest correspondence from a fn_t
    void operator()(const fn_t &n);
    //! Don't harvest correspondences from any other types
    template<typename N>
    void operator()(const N &n) {
    }

    
    
};
}

/*! 
\addtogroup rewriters
@{
 */


//! A rewriter which instantiates function objects for all procedures
/*!
  This rewriter creates functor object wrappers for all procedures
  except the entry point.

  It also instantiates functor objects when appropriate. For polymorphic
  functors, this requires type propagation to discover what the
  type variables in the polytype of the functor object should be.
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
    //! Constructor
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
/*! 
  @}
 */

}
