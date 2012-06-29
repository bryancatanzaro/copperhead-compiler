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
#include <stack>
#include "node.hpp"
#include "expression.hpp"
#include "statement.hpp"
#include "cppnode.hpp"
#include "type.hpp"
#include "ctype.hpp"

#include "type_printer.hpp"
#include "utility/isinstance.hpp"

/*!
  \file   rewriter.hpp
  \brief  Contains the definition of the rewriter visitor
  
  
*/

namespace backend {


/*! \addtogroup rewriters
  @{
 */


//! Parent visitor class for all AST rewrites.
/*! 
  \p rewriter traverses an AST and copies it to produce a new AST.
  Its rewrite methods should be redefined by
  rewrite passes which selectively transform the AST.  Uses the
  Curiously Recurring Template Pattern in lieu of virtual methods.
  
  Importantly, \p rewriter does not actually copy AST nodes.
  Instead, as the rewrite proceeds,
  \p rewriter checks at each AST node to see if the rewritten node
  is identical to the source node. If so, \p rewriter will return
  a shared_ptr to the source node, rather than allocating a new
  node which is just a copy of the source.  This saves memory and time,
  since most AST rewrites leave a majority of the nodes untouched.

  This optimization works because AST nodes are immutable.
 */

template<typename Derived>
class rewriter
    : public boost::static_visitor<std::shared_ptr<const node> >
{
private:
    Derived& get_sub();
    
protected:
    /*! Used for bookkeeping to discover straight copies */
    std::stack<bool> m_matches;
    //! Start the check to see if we are doing a straight copy
    /*! To check whether we are doing a straight copy, we have to
      examine all subfields of an AST node. This function begins
      the process of examining all subfields of an AST node.
      It enables recursive matching, so that multiple potential
      matches can be evaluated simultaneously during recursion
      on the AST.
    */
    void start_match();
    
    //! Update the current match under consideration
/*! 
  \param t A shared_ptr to the new node after rewriting.
  \param u The source node.
*/
    template<typename T, typename U>
    inline void update_match(const std::shared_ptr<T>& t, const U& u) {
        m_matches.top() = m_matches.top() && (t == u.ptr());
    }
    
    //! Did we find a match, meaning we have a straight copy? 
/*! 
  \return \p true if we have found a match,
  \p false if the match is not perfect.
*/
    bool is_match();

public:
    result_type operator()(const literal& n);

    result_type operator()(const name &n);
    
    result_type operator()(const tuple &n);
    
    result_type operator()(const apply &n);
    
    result_type operator()(const lambda &n);
    
    result_type operator()(const closure &n);

    result_type operator()(const subscript &n);
        
    result_type operator()(const conditional &n);
    
    result_type operator()(const ret &n);
    
    result_type operator()(const bind &n);
    
    result_type operator()(const call &n);
    
    result_type operator()(const procedure &n);
    
    result_type operator()(const suite &n);
    
    result_type operator()(const structure &n);
    
    result_type operator()(const templated_name &n);
    
    result_type operator()(const include &n);
    
    result_type operator()(const typedefn &n);

    result_type operator()(const namespace_block &n);

    result_type operator()(const while_block &n);
};
/*! 
  @}
 */

}

#include "rewriter.inl"
