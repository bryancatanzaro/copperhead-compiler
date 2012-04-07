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

#include <boost/variant.hpp>
#include <functional>
#include <memory>

namespace backend {

class monotype_t;
class polytype_t;
class sequence_t;
class tuple_t;
class fn_t;

namespace detail {
typedef boost::variant<
    monotype_t &,
    polytype_t &,
    sequence_t &,
    tuple_t &,
    fn_t &
    > type_base;

struct make_type_base_visitor
    : boost::static_visitor<type_base>
{
    make_type_base_visitor(void *p);
    template<typename Derived>
    type_base operator()(const Derived &) const {
        // use of std::ref disambiguates variant's copy constructor dispatch
        return type_base(std::ref(*reinterpret_cast<Derived*>(ptr)));
    }
    void *ptr;
};

type_base make_type_base(void *ptr, const type_base &other); 

}

/*! 
  \addtogroup types
  @{
 */

//! Parent for all Copperhead type objects
/*! Not intended to be instantiated directly 
*/
class type_t
    : public detail::type_base,
      public std::enable_shared_from_this<type_t>
{
protected:
    typedef detail::type_base super_t;
    //! Constructor from subclass
/*!   
  \param self Reference to subclass object under construction
*/
    template<typename Derived>
    type_t(Derived &self)
        : super_t(std::ref(self)) //use of std::ref disambiguates variant's copy constructor dispatch
        {}
public:

//! Copy constructor
/*! 
  \param other type to copy from
*/
    type_t(const type_t &other);
//! Get the pointer to this type object
    std::shared_ptr<const type_t> ptr() const;
};

/*!
  @}
*/

}
