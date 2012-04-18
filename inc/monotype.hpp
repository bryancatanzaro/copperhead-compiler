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
#include <boost/iterator/indirect_iterator.hpp>
#include "type.hpp"
#include <iostream>
#include <string>

namespace backend {

/*!
  \addtogroup types
  @{
*/

//! Monomorphic type
/*! This class is used both directly and as a parent class to derived
 * monotypes.
 */

class monotype_t :
    public type_t
{
protected:
    const std::string m_name;
    std::vector<std::shared_ptr<const type_t> > m_params;
   
public:
    //! Basic constructor
/*! This constructor is used for simple monotypes with no subtypes.
  
  \param name The name of the type
*/
    monotype_t(const std::string &name);
//! Derived constructor
/*! This constructor is instantiated by subclasses during construction.
  
  \param self Reference to derived object being constructed.
  \param name The name of the type.
*/
    template<typename Derived>
    monotype_t(Derived& self,
               const std::string &name) :
        type_t(self), m_name(name) {}
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
//! Gets the name of the type
/*! 
  \return The name of the type. 
*/
    //XXX Should this be id() to be consistent?
    const std::string& name(void) const;
//! Iterator type over the subtypes contained in this monotype
    typedef decltype(boost::make_indirect_iterator(m_params.cbegin())) const_iterator;
    //! Iterator to the beginning of the subtypes contained in this monotype
    const_iterator begin() const;
    //! Iterator to the end of the subtypes contained in this monotype
    const_iterator end() const;
    //! Number of subtypes in this monotype. Returns 0 for non-nested types.
    int size() const;
    //! Get the pointer to this type object
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
/*!   
  \param sub The type of each element of the Sequence
*/
class sequence_t :
        public monotype_t
{
public:
    sequence_t(const std::shared_ptr<const type_t> &sub);
    //! Gets the type of each element of the Sequence
    const type_t& sub() const;
    //! Get the pointer to this type object
    std::shared_ptr<const sequence_t> ptr() const;

};

//! Tuple type.
/*! This is used extensively to represent any sequence of types
*/
class tuple_t :
        public monotype_t
{
public:
    //! Constructor
/*! 
  
  \param sub A \p vector of pointers to types which are contained in
  this tuple.
*/
    tuple_t(std::vector<std::shared_ptr<const type_t> > && sub);
    //! Derived constructor
/*! 
  
  \param self A reference to the derived object being constructed
  \param name The name of the type being constructed
  \param sub A \p vector of pointers to types which are contained in
  this tuple-like type.
*/
    template<typename Derived>
    inline tuple_t(Derived& self,
                   const std::string& name,
                   std::vector<std::shared_ptr<const type_t> > && sub)
        : monotype_t(self, name, std::move(sub))
        {}
    //! Get the pointer to this type object
    std::shared_ptr<const tuple_t> ptr() const;

};


//! Function type
class fn_t :
        public monotype_t
{
public:
    //! Constructor
    /*! 
  
  \param args A tuple of types of the arguments
  \param result The type of the result
  
  \return 
*/
    fn_t(const std::shared_ptr<const tuple_t> args,
                const std::shared_ptr<const type_t> result);
    //! Gets the argument types
    const tuple_t& args() const;
    //! Gets the result type
    const type_t& result() const;
    //! Get the pointer to this type object
    std::shared_ptr<const fn_t> ptr() const;

};

/*!
  @}
*/

}
