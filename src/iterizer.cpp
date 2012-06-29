#include "iterizer.hpp"

using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using std::vector;
using std::move;
using std::string;
using std::tuple;

namespace backend {

namespace detail {

class analyze_recursion
    : public rewriter<is_tail_recursive> {
private:
    string proc_name;
    bool in_else_branch;
    bool nested;
    bool recursive;
    bool sense;
    std::shared_ptr<const expression> pred;

public:
    analyze_recursion() : in_else_branch(false), nested(false),
                          recursive(false), sense(false),
                          pred(0) {}

    
    using rewriter<is_tail_recursive>::operator();

    result_type operator()(const conditional& c) {
        
        pred = c.cond().ptr();
        
    }
    
    result_type operator()(const procedure& p) {
        proc_name = p.name().id();
        return rewriter<is_tail_recursive>::operator()(p);
    }

};


iterizer::result_type iterizer::operator()(const procedure& p) {

    
    return p.ptr();
}

}
