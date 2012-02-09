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
  Its rewrite methods are virtual and should be redefined by
  rewrite passes which selectively transform the AST.
  
  Importantly, \p rewriter does not actually copy AST nodes.
  Instead, as the rewrite proceeds,
  \p rewriter checks at each AST node to see if the rewritten node
  is identical to the source node. If so, \p rewriter will return
  a shared_ptr to the source node, rather than allocating a new
  node which is just a copy of the source.  This saves memory and time,
  since most AST rewrites leave a majority of the nodes untouched.

  This optimization works because AST nodes are immutable.
 */

class rewriter
    : public boost::static_visitor<std::shared_ptr<node> >
{
protected:
    //! Get the \p shared_ptr holding an AST \p node
/*! 
  If a rewrite pass knows that no rewriting will happen at
  deeper levels of the AST, it can grab the shared_ptr
  holding the AST node directly and return it during the copy.

  \param n Assumed to be a \p node held by a \p shared_ptr .
  
  \return The shared_ptr that holds \n
*/
    //If you know you're not going to do any rewriting at deeper
    //levels of the AST, just grab the pointer from the node
    template<typename Node>
    static inline std::shared_ptr<Node> get_node_ptr(const Node &n) {
        return std::static_pointer_cast<Node>(
            std::const_pointer_cast<node>(n.shared_from_this()));
    }
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
        m_matches.top() = m_matches.top() && (t == get_node_ptr(u));
    }
    
    //! Did we find a match, meaning we have a straight copy? 
/*! 
  \return \p true if we have found a match,
  \p false if the match is not perfect.
*/
    bool is_match();

public:
    virtual result_type operator()(const literal& n);

    virtual result_type operator()(const name &n);
    
    virtual result_type operator()(const tuple &n);
    
    virtual result_type operator()(const apply &n);
    
    virtual result_type operator()(const lambda &n);
    
    virtual result_type operator()(const closure &n);
        
    virtual result_type operator()(const conditional &n);
    
    virtual result_type operator()(const ret &n);
    
    virtual result_type operator()(const bind &n);
    
    virtual result_type operator()(const call &n);
    
    virtual result_type operator()(const procedure &n);
    
    virtual result_type operator()(const suite &n);
    
    virtual result_type operator()(const structure &n);
    
    virtual result_type operator()(const templated_name &n);
    
    virtual result_type operator()(const include &n);
    
    virtual result_type operator()(const typedefn &n);

    virtual result_type operator()(const namespace_block &n);
    
};
/*! 
  @}
 */

}
