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

namespace backend {

class copier
    : public no_op_visitor<std::shared_ptr<node> >
{
protected:
    //If you know you're not going to do any rewriting at deeper
    //levels of the AST, just grab the pointer from the node
    template<typename Node>
    static inline result_type get_node_ptr(const Node &n) {
        return std::const_pointer_cast<node>(n.shared_from_this());
    }
    std::stack<bool> m_matches;
    void start_match();
    
    template<typename T, typename U>
    inline void update_match(const T& t, const U& u) {
        m_matches.top() = m_matches.top() && (t == get_node_ptr(u));
    }
    bool is_match();

public:
    using backend::no_op_visitor<std::shared_ptr<node> >::operator();
  
    
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
    
};

}
