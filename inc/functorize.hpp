/*
 *   Copyright 2012      NVIDIA Corporation
 * 
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 * 
 *       http://www.apache.org/licenses/LICENSE-2.0
 * 
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 * 
 */
#pragma once
#include <string>
#include <set>
#include <sstream>
#include "rewriter.hpp"
#include "utility/isinstance.hpp"
#include "utility/markers.hpp"
#include "utility/initializers.hpp"
#include "import/library.hpp"
#include "type_convert.hpp"

#include "type_printer.hpp"
#include "environment.hpp"

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
                     std::shared_ptr<const type_t> > type_map;

    std::shared_ptr<const type_t> m_working;
    type_map& m_corresponded;

public:
    //! Constructor
/*!
\param input Input type to be matched
\param corresponded Type map that will be amended with correspondence
\return
*/
    type_corresponder(const std::shared_ptr<const type_t>& input,
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
    : public rewriter<functorize>
{
private:
    const std::string& m_entry_point;
    std::vector<result_type> m_additionals;
    const registry& m_reg;
    std::map<std::string,
             std::shared_ptr<const type_t> > m_fns;

    std::shared_ptr<const expression> instantiate_fn(const name& n,
                                                     const type_t& t);

     typedef std::map<std::string,
                      std::shared_ptr<const type_t> > type_map;
public:
    //! Constructor
    /*! \param entry_point The name of the entry point procedure
        \param reg The registry of functions the compiler knows about
     */
    functorize(const std::string& entry_point,
               const registry& reg);
    
    using rewriter<functorize>::operator();

    result_type operator()(const apply &n);
    
    result_type operator()(const suite &n);
    
    result_type operator()(const procedure &n);
    
};
/*! 
  @}
 */

}
