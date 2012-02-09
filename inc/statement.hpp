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
#include "type.hpp"
#include "monotype.hpp"
#include "ctype.hpp"
#include <vector>
#include <memory>

namespace backend {

/*! 
  \addtogroup nodes
  @{
 */

//! Parent class for all statement AST nodes.
class statement
    : public node
{
protected:
    template<typename Derived>
    statement(Derived &self)
        : node(self)
        {}
};

//! AST node for the Return statement.
/*! "return $val" 
*/
class ret
    : public statement
{
public:
/*!
  \param val Expression to be returned.  
*/
    ret(const std::shared_ptr<expression> &val);
protected:
    const std::shared_ptr<expression> m_val;
public:
    const expression& val(void) const;
    std::shared_ptr<expression> p_val(void) const;
};
//! AST node for the Bind statement.
/*! "$lhs = $rhs"
 */
class bind
    : public statement
{
public:
    /*!  
      \param lhs Left hand side
      \param rhs Right hand side
    */
    bind(const std::shared_ptr<expression> &lhs,
         const std::shared_ptr<expression> &rhs);
protected:
    const std::shared_ptr<expression> m_lhs;
    const std::shared_ptr<expression> m_rhs;

public:
    const expression& lhs(void) const;
    const expression& rhs(void) const;
    std::shared_ptr<expression> p_lhs(void) const;
    std::shared_ptr<expression> p_rhs(void) const;
};

//! AST node for the Call statement.
/*! This statement calls a function, but does not bind the result
  of the function call to any variable. This is not used in front-end
  code (where nothing can be mutated), but only internally by the
  compiler.
*/
class call
    : public statement
{
protected:
    const std::shared_ptr<apply> m_sub;
public:
    /*!  
      \param n The Apply expression which is called by this statement.
    */

    call(const std::shared_ptr<apply> &n);
    const apply& sub(void) const;
    std::shared_ptr<apply> p_sub(void) const;
};
        
//! AST node for the Procedure statement.
/*! This defines a procedure.
 */
class procedure
    : public statement
{
public:
   /*!  
     \param id The name of the procedure.
     \param args The formal arguments of the procedure.
     \param stmts The code body of the procedure.
     \param type The Copperhead type of the procedure.
     \param ctype The C++ implementation type of the procedure.
     \param place Optional decorator for the procedure. E.g. "__device__",
     "extern static", etc.
   */
    procedure(const std::shared_ptr<name> &id,
              const std::shared_ptr<tuple> &args,
              const std::shared_ptr<suite> &stmts,
              const std::shared_ptr<type_t> &type = void_mt,
              const std::shared_ptr<ctype::type_t> &ctype = ctype::void_mt,
              const std::string &place =
              "__device__");
protected:
    const std::shared_ptr<name> m_id;
    const std::shared_ptr<tuple> m_args;
    const std::shared_ptr<suite> m_stmts;
    std::shared_ptr<type_t> m_type;
    std::shared_ptr<ctype::type_t> m_ctype;
    const std::string m_place;
public:
    const name& id(void) const;
    const tuple& args(void) const;
    const suite& stmts(void) const;
    std::shared_ptr<name> p_id(void) const;
    std::shared_ptr<tuple> p_args(void) const;
    std::shared_ptr<suite> p_stmts(void) const;
    const type_t& type(void) const;
    const ctype::type_t& ctype(void) const;
    std::shared_ptr<type_t> p_type(void) const;
    std::shared_ptr<ctype::type_t> p_ctype(void) const;
    const std::string& place(void) const;

};
//! AST node for conditional statements
class conditional
    : public statement
{
protected:
    std::shared_ptr<expression> m_cond;
    std::shared_ptr<suite> m_then;
    std::shared_ptr<suite> m_orelse;
    
public:
/*! 
  
  \param cond An expression which decides which code block is executed.
  \param then Code block which is executed if \p cond evaluates to \p true .
  \param orelse Code block which is executed if \p cond evaluates to \p false .
*/
    conditional(std::shared_ptr<expression> cond,
                std::shared_ptr<suite> then,
                std::shared_ptr<suite> orelse);
    
    const expression& cond(void) const;
    const suite& then(void) const;
    const suite& orelse(void) const;
    std::shared_ptr<expression> p_cond(void) const;
    std::shared_ptr<suite> p_then(void) const;
    std::shared_ptr<suite> p_orelse(void) const;
};


//! AST node representing a sequence of \p statement nodes.
class suite 
    : public node
{
public:
/*! 
  
  \param stmts The \p statement nodes held by this node.
*/
    suite(std::vector<std::shared_ptr<statement> > &&stmts);
protected:
    const std::vector<std::shared_ptr<statement> > m_stmts;
public:
    typedef decltype(boost::make_indirect_iterator(m_stmts.cbegin())) const_iterator;
    const_iterator begin() const;
    const_iterator end() const;
    typedef decltype(m_stmts.cbegin()) const_ptr_iterator;
    const_ptr_iterator p_begin() const;
    const_ptr_iterator p_end() const;
    int size() const;
};

/*! 
  @}
 */


}

