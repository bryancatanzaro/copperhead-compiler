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

#include "node.hpp"
#include <vector>
#include <memory>
#include <ostream>
#include <boost/iterator/indirect_iterator.hpp>

#include "type.hpp"
#include "monotype.hpp"
#include "ctype.hpp"

/*!
  \file   expression.hpp
  \brief  Contains the declarations for all AST expression nodes.

*/


namespace backend {


/*! 
  \addtogroup nodes
  @{
 */


//! The super class for all AST expression nodes.
/*! Every expression bears a type and a ctype.  The type is a Copperhead
  type describing the interface the expression must have. The ctype
  is used during implementation to describe the C++ type used to
  implement the expression node.
*/
class expression
    : public node
{
protected:
    //! The Copperhead type of the expression
    std::shared_ptr<const type_t> m_type;
    //! The C++ implementation type of the expression
    std::shared_ptr<const ctype::type_t> m_ctype;
/*! 
  The constructor is only intended to be called by subtypes.
  \param self A reference to the subtype object being constructed,
  necessary for disambiguating the underlying \p variant.
  \param type The Copperhead type of the expression, defaults to Void
  \param ctype The C++ type of the expression, defaults to void
  
*/
    template<typename Derived>
    expression(Derived &self,
               const std::shared_ptr<const type_t>& type = void_mt,
               const std::shared_ptr<const ctype::type_t>& ctype = ctype::void_mt)
        : node(self), m_type(type), m_ctype(ctype)
        {}
public:
    
    //! Gets the Copperhead type of the expression
    const type_t& type(void) const;
    //! Gets the C++ implementation type of the expression
    const ctype::type_t& ctype(void) const;
    /*! When we need to get at the pointer holding an expression */
    std::shared_ptr<const expression> ptr() const;

};

//! The parent class for all Literal expressions
class literal
    : public expression
{
protected:
    const std::string m_val;
public:
/*! This constructor is called from subclasses.
  
  \param self Reference to child being constructed.
  \param val Value of the literal.
  \param type Copperhead type.
  \param ctype C++ implementation type.
*/
    template<typename Derived>
    literal(Derived &self,
            const std::string& val,
            const std::shared_ptr<const type_t>& type = void_mt,
            const std::shared_ptr<const ctype::type_t>& ctype = ctype::void_mt)
        : expression(self, type, ctype), m_val(val)
        {}
    //! A constructor for standalone Literal expressions.
/*! \p literal can also be used on its own, with no subclass. This
  constructor facilitates this.
  
  \param val Value of the literal.
  \param type Copperhead type.
  \param ctype C++ implementation type.
*/
    literal(const std::string& val,
            const std::shared_ptr<const type_t>& type = void_mt,
            const std::shared_ptr<const ctype::type_t>& ctype = ctype::void_mt);

    //! Get the value of this literal
    const std::string& id(void) const;
    /*! When we need to get at the pointer holding a literal */
    std::shared_ptr<const literal> ptr() const;

};
//! AST node for identifiers.
class name
    : public literal
{   
public:
/*!   
  \param val The identifier text.
  \param type The Copperhead type.
  \param ctype The C++ implementation type.
*/
    name(const std::string &val,
         const std::shared_ptr<const type_t>& type = void_mt,
         const std::shared_ptr<const ctype::type_t>& ctype = ctype::void_mt);
//! Constructor for subclasses.
/*! 
  \param self Reference to subclass being constructed.
  \param val The identifier text.
  \param type The Copperhead type.
  \param ctype The C++ implementation type.
  
  \return 
*/
    template<typename Derived>
    name(Derived& self, const std::string &val,
         const std::shared_ptr<const type_t>& type,
         const std::shared_ptr<const ctype::type_t>& ctype) :
        literal(self, val, type, ctype) {}
    
    /*! When we need to get at the pointer holding a name */
    std::shared_ptr<const name> ptr() const;

};
//! AST node for tuples.
/*! We use tuples to represent any sequence of expressions in the
  program text.
*/

class tuple
    : public expression
{
public:
/*! 
  \param values The sequence of expressions. 
  \param type The Copperhead type for this tuple.
  \param ctype The C++ implementation type for this tuple.
  
  \return 
*/
    tuple(std::vector<std::shared_ptr<const expression> > &&values,
          const std::shared_ptr<const type_t>& type = void_mt,
          const std::shared_ptr<const ctype::type_t>& ctype = ctype::void_mt);
protected:
    const std::vector<std::shared_ptr<const expression> > m_values;
public:
    //! A constant iterator to the expressions held by the tuple
    typedef decltype(boost::make_indirect_iterator(m_values.cbegin())) const_iterator;
    //! The iterator to the beginning of the expressions held by the tuple
    const_iterator begin() const;
    //! The iterator to the end of the expressions held by the tuple
    const_iterator end() const;
        
    //! The arity of this tuple.
/*! 
  \return How many expressions are held by this tuple.
*/
    int arity() const;
    /*! When we need to get at the pointer holding a tuple */
    std::shared_ptr<const tuple> ptr() const;

};
//! AST node for applying a function to a sequence of arguments
class apply
    : public expression
{
protected:
    const std::shared_ptr<const name> m_fn;
    const std::shared_ptr<const tuple> m_args;
public:
/*! 
  \param fn The function to be applied.
  \param args The arguments given to the function.
*/

    apply(const std::shared_ptr<const name> &fn,
          const std::shared_ptr<const tuple> &args);
    //! Get the function to be applied
    const name &fn(void) const;
    //! Get the arguments given to the function
    const tuple &args(void) const;
    /*! When we need to get at the pointer holding a apply */
    std::shared_ptr<const apply> ptr() const;

};

//! AST node for \p lambda expressions
/*! Anonymous function definition
 */  
class lambda
    : public expression
{
protected:
    const std::shared_ptr<const tuple> m_args;
    const std::shared_ptr<const expression> m_body;
public:
    //! Constructor
/*!
  \param args Formal arguments of the \p lambda
  \param body Expression to be evaluated in the \p lambda
  \param type Copperhead type.
  \param ctype C++ implementation type.
*/

    lambda(const std::shared_ptr<const tuple> &args,
           const std::shared_ptr<const expression> &body,
           const std::shared_ptr<const type_t>& type = void_mt,
           const std::shared_ptr<const ctype::type_t>& ctype = ctype::void_mt);
    //! Get the formal arguments of the \p lambda
    const tuple &args(void) const;
    //! Get the expression to be evaluated in the \p lambda
    const expression &body(void) const;
    /*! When we need to get at the pointer holding a lambda */
    std::shared_ptr<const lambda> ptr() const;

};

//! AST node for closures
/*! This node makes closures explicit. It encloses a sequence of
  identifiers, and an expression where these identifiers are used.
*/
class closure
    : public expression
{
protected:
    const std::shared_ptr<const tuple> m_args;
    const std::shared_ptr<const expression> m_body;
    
public:
/*!   
  \param args Identifiers being closed over.
  \param body Expression where identifiers are used
  \param type Copperhead type.
  \param ctype C++ implementation type.
*/
    closure(const std::shared_ptr<const tuple> &args,
            const std::shared_ptr<const expression> &body,
            const std::shared_ptr<const type_t>& type = void_mt,
            const std::shared_ptr<const ctype::type_t>& ctype = ctype::void_mt);
    const tuple &args(void) const;
    const expression &body(void) const;
    /*! When we need to get at the pointer holding a closure */
    std::shared_ptr<const closure> ptr() const;

};

//! AST node for subscripting
class subscript
    : public expression
{
protected:
    const std::shared_ptr<const name> m_src;
    const std::shared_ptr<const expression> m_idx;
    
public:
/*!   
  \param src Identifier being subscripted
  \param idx Subscript index
  \param type Copperhead type.
  \param ctype C++ implementation type.
*/
    subscript(const std::shared_ptr<const name> &src,
              const std::shared_ptr<const expression> &idx,
              const std::shared_ptr<const type_t>& type = void_mt,
              const std::shared_ptr<const ctype::type_t>& ctype = ctype::void_mt);
    const name &src(void) const;
    const expression &idx(void) const;
    /*! When we need to get at the pointer holding a subscript */
    std::shared_ptr<const subscript> ptr() const;

};

/*! 
  @}
 */


}
    

