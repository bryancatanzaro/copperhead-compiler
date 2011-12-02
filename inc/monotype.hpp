#pragma once
#include <vector>
#include <boost/iterator/indirect_iterator.hpp>
#include "type.hpp"
#include <iostream>
#include <string>
#include "utility/initializers.hpp"

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
    std::vector<std::shared_ptr<type_t> > m_params;
   
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
               std::vector<std::shared_ptr<type_t> > &&params)
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

};

extern std::shared_ptr<monotype_t> int32_mt;
extern std::shared_ptr<monotype_t> int64_mt;
extern std::shared_ptr<monotype_t> uint32_mt;
extern std::shared_ptr<monotype_t> uint64_mt;
extern std::shared_ptr<monotype_t> float32_mt;
extern std::shared_ptr<monotype_t> float64_mt;
extern std::shared_ptr<monotype_t> bool_mt;
extern std::shared_ptr<monotype_t> void_mt;

//! Sequence type.
/*!   
  \param sub The type of each element of the Sequence
*/
class sequence_t :
        public monotype_t
{
public:
    sequence_t(const std::shared_ptr<type_t> &sub);
    //! Gets the type of each element of the Sequence
    const type_t& sub() const;
    //! Gets a \p shared_ptr to the type of each element of the Sequence
    std::shared_ptr<type_t> p_sub() const;
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
    tuple_t(std::vector<std::shared_ptr<type_t> > && sub);
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
                   std::vector<std::shared_ptr<type_t> > && sub)
        : monotype_t(self, name, std::move(sub))
        {}
    //! An iterator to the pointers of types contained in this tuple
    typedef decltype(m_params.cbegin()) const_ptr_iterator;
    //! An iterator to the pointer of the first type of the tuple
    const_ptr_iterator p_begin() const;
    //! An iterator to the pointer of the last type of the tuple
    const_ptr_iterator p_end() const;
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
    fn_t(const std::shared_ptr<tuple_t> args,
                const std::shared_ptr<type_t> result);
    //! Gets the argument types
    const tuple_t& args() const;
    //! Gets the result type
    const type_t& result() const;
    //! Gets a pointer to the argument types
    std::shared_ptr<tuple_t> p_args() const;
    //! Gets a pointer to the result types
    std::shared_ptr<type_t> p_result() const;
};

/*!
  @}
*/

}
