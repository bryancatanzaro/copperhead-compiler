#include "prune.hpp"

using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using std::vector;
using std::deque;
using std::move;
using std::string;

namespace backend {

prune::result_type prune::operator()(const suite& n) {
    deque<shared_ptr<const statement> > stmts;
    auto i = n.end();
    do {
        i--;
        shared_ptr<const statement> rewritten =
            static_pointer_cast<const statement>(
                boost::apply_visitor(*this, *i));
        if (rewritten != shared_ptr<const statement>()) {
            stmts.push_front(rewritten);
        }
    } while (i != n.begin());
    return make_shared<const suite>(
        vector<shared_ptr<const statement> >(stmts.begin(), stmts.end()));
}

prune::result_type prune::operator()(const name& n) {
    m_used.insert(n.id());
    return n.ptr();
}

prune::result_type prune::operator()(const bind& n) {
    shared_ptr<const expression> rhs =
        static_pointer_cast<const expression>(
            boost::apply_visitor(*this, n.rhs()));
    assert(detail::isinstance<name>(n.lhs()));
    const name& lhs = boost::get<const name&>(n.lhs());
    if (m_used.exists(lhs.id())) {
        return n.ptr();
    } else {
        return prune::result_type();
    }
}

prune::result_type prune::operator()(const procedure &n) {
    m_used.begin_scope();
    auto rewritten = rewriter<prune>::operator()(n);
    m_used.end_scope();
    return rewritten;
}

}
