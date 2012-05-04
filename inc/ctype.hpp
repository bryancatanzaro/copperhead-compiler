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
#include <vector>
#include <boost/variant.hpp>
#include <boost/iterator/indirect_iterator.hpp>
#include <memory>
#include <iostream>

namespace backend {

namespace ctype {

class monotype_t;
class polytype_t;
class sequence_t;
class tuple_t;
class fn_t;
class cuarray_t;
class zipped_sequence_t;

namespace detail {
typedef boost::variant<
    monotype_t &,
    polytype_t &,
    sequence_t &,
    tuple_t &,
    fn_t &,
    cuarray_t &,
    zipped_sequence_t &
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
  \addtogroup ctypes
  @{
 */

//! Parent type for all C++ implementation types
/*! Not intended to be instantiated directly 
  
  \param self 
  
  \return 
*/
class type_t
    : public detail::type_base,
      public std::enable_shared_from_this<type_t>
{
protected:
    typedef detail::type_base super_t;
    //! Derived constructor
/*! To be called by derived object during construction.
  
  \param self Reference to derived object being constructed.
*/
    template<typename Derived>
    type_t(Derived &self)
        : super_t(std::ref(self)) //use of std::ref disambiguates variant's copy constructor dispatch
        {}
public:
    //! Copy constructor
    type_t(const type_t &other);
    //! Get pointer holding this type_t object
    std::shared_ptr<const type_t> ptr() const;
};

//! Monomorphic type
/*! Can be used standalone or as a parent class */
class monotype_t :
    public type_t
{
protected:
    const std::string m_name;
    const std::vector<std::shared_ptr<const type_t> > m_params;
public:
    //! Basic onstructor
/*! This constructor is used for simple monotypes with no subtypes.
  \param name The name of type.
*/
    monotype_t(const std::string &name);
    
    //! Derived constructor
/*! To be called during construction of derived object
  
  \param self Reference to derived object under construction.
  \param name Name of type.
  
*/
    template<typename Derived>
    monotype_t(Derived &self,
               const std::string &name)
        : type_t(self),
          m_name(name)
        {}

    //! Derived constructor
/*! This constructor is instantiated by subclasses during
 *  construction. It accepts a list of subtypes that define this
 *  type. For example, Seq(Seq(Int)) is a nested type, and
 *  constructing it requires defining subtypes.
 
  
  \param self Reference to derived object being constructed.
  \param name The name of the type.
  \param params List of subtypes.
*/
    template<typename Derived>
    monotype_t(Derived& self,
               const std::string &name,
               std::vector<std::shared_ptr<const type_t> > &&params)
        : type_t(self), m_name(name), m_params(std::move(params)) {}
    
    //! Gets name of type.
    const std::string& name(void) const;
    //! Type of iterator to subtypes in this monotype
    typedef decltype(boost::make_indirect_iterator(m_params.cbegin())) const_iterator;
    //! Iterator to the beginning of the subtypes contained in this monotype
    const_iterator begin() const;
    //! Iterator to the end of the subtypes contained in this monotype
    const_iterator end() const;
    //! Number of subtypes in this monotype. Returns 0 for non-nested types.
    int size() const;
    //! Get pointer holding this type_t object
    std::shared_ptr<const monotype_t> ptr() const;
};

extern std::shared_ptr<const monotype_t> int32_mt;
extern std::shared_ptr<const monotype_t> int64_mt;
extern std::shared_ptr<const monotype_t> uint32_mt;
extern std::shared_ptr<const monotype_t> uint64_mt;
extern std::shared_ptr<const monotype_t> float32_mt;
extern std::shared_ptr<const monotype_t> float64_mt;
extern std::shared_ptr<const monotype_t> bool_mt;
extern std::shared_ptr<const monotype_t> void_mt;

//! Sequence type.
/* Can be used directly or as a parent class */
class sequence_t :
        public monotype_t
{
public:
    //! Basic constructor
/*!   
  \param sub Type of element of the Sequence
*/
    sequence_t(const std::shared_ptr<const type_t> &sub);
    //! Derived constructor
/*! To be called during the construction of a derived object
  
  \param self Reference to the derived object under construction
  \param name Name of the Sequence-like type.
  \param sub Type of the element of the Sequence-like type.
  
*/
    template<typename Derived>
    sequence_t(Derived &self,
               const std::string& name,
               const std::shared_ptr<const type_t>& sub); // :
        // monotype_t(self,
        //            name,
        //            utility::make_vector<std::shared_ptr<const type_t> >(sub)) {}
    //! Gets the type of the element of the Sequence
    const type_t& sub() const;
    //! Gets a pointer to the type of the element of the Sequence
    std::shared_ptr<const type_t> p_sub() const;
    //! Get pointer holding this type_t object
    std::shared_ptr<const sequence_t> ptr() const;
};

//! cuarray_t functions as a sequence_t, but prints differently
class cuarray_t :
        public sequence_t
{
public:
    //! Basic constructor
/*!   
  \param sub Type of element of the Sequence
*/
    cuarray_t(const std::shared_ptr<const type_t> &sub);
    //! Get pointer holding this type_t object
    std::shared_ptr<const cuarray_t> ptr() const;
};

//! zipped_sequence_t functions as a sequence_t, but prints differently
class zipped_sequence_t :
        public sequence_t
{
public:
    //! Basic constructor
/*!   
  \param sub Type of element of the Sequence
*/
    zipped_sequence_t(const std::shared_ptr<const tuple_t> &sub);
    //! Get pointer holding this type_t object
    std::shared_ptr<const zipped_sequence_t> ptr() const;
};

//! Tuple type.
class tuple_t :
        public monotype_t
{
public:
    //! Constructor
/*! 
  \param sub A vector of types contained in this tuple.
*/
    tuple_t(std::vector<std::shared_ptr<const type_t> > && sub);
    //! Get pointer holding this type_t object
    std::shared_ptr<const tuple_t> ptr() const;
};

//! Function type
class fn_t :
        public monotype_t
{
public:
    //! Constructor
/*! 
  \param args Tuple of argument types.
  \param result Result type.
*/
    fn_t(const std::shared_ptr<const tuple_t> args,
         const std::shared_ptr<const type_t> result);
    //! Gets the tuple of argument types.
    const tuple_t& args() const;
    //! Gets the result type.
    const type_t& result() const;
    //! Get pointer holding this type_t object
    std::shared_ptr<const fn_t> ptr() const;
};

//! Polymorphic type
/*! This is translated into C++ templated types.  Note that since C++
   templated types can (and often are) nested, this type is somewhat
   different from its analogue Copperhead
   type \ref backend::polytype_t.  Specifically, instead of accepting
   a vector of monotypes, it accepts
   both \ref backend::ctype::monotype_t and backend::ctype::polytype_t
   objects as type variables.
   
   This reflects the less restrictive type system of C++ compared to
   Copperhead.  */
class polytype_t
    : public type_t {
private:
    const std::vector<std::shared_ptr<const type_t> > m_vars;
    const std::shared_ptr<const monotype_t> m_monotype;

public:
    //! Constructor
/*! 
  
  \param vars Variables to instantiate type with.  Note that in
  contrast to the Copperhead \ref backend::polytype_t, the
  C++ \ref backend::ctype::polytype_t can be nested, which is why this
  is a \p vector of \ref backend::ctype::type_t rather than
  a \p vector of \ref backend::ctype::monotype_t.

  \param monotype Base type
  
  \return 
*/
    polytype_t(std::vector<std::shared_ptr<const type_t> >&& vars,
               const std::shared_ptr<const monotype_t>& monotype);
//! Gets base type
    const monotype_t& monotype() const;
    //! Gets \p std::shared_ptr to base type
    std::shared_ptr<const monotype_t> p_monotype() const;
    //! Type of iterator to type variables
    typedef decltype(boost::make_indirect_iterator(m_vars.cbegin())) const_iterator;
    //! Iterator to beginning of type variables
    const_iterator begin() const;
    //! Iterator to end of type variables
    const_iterator end() const;
    //! Get pointer holding this type_t object
    std::shared_ptr<const polytype_t> ptr() const;
};

/*
  @}
*/

}
}
