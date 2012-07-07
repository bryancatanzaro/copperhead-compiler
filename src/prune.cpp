#include "prune.hpp"

#include <algorithm>

using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using std::vector;
using std::reverse;
using std::move;
using std::string;

namespace backend {

namespace detail {
class protector :
        public rewriter<protector> {
private:
    environment<string> m_declared;
    environment<string> m_modified;
public:
    using rewriter<protector>::operator();

    result_type operator()(const procedure &p) {
        for(auto i = p.args().begin();
            i != p.args().end();
            i++) {
            assert(detail::isinstance<name>(*i));
            const name& arg_name = boost::get<const name&>(*i);
            m_declared.insert(arg_name.id());
        }
        return rewriter<protector>::operator()(p);
    }

    result_type operator()(const bind &b) {
        assert(detail::isinstance<name>(b.lhs()));
        const name& lhs = boost::get<const name&>(b.lhs());
        if (m_declared.exists(lhs.id())) {
            m_modified.insert(lhs.id());
        } else {
            m_declared.insert(lhs.id());
        }
        return b.ptr();
    }
    
    const environment<string>& modified() const {
        return m_modified;
    }
    
};
}


prune::result_type prune::operator()(const suite& n) {
    vector<shared_ptr<const statement> > stmts;
    auto i = n.end();
    do {
        i--;
        shared_ptr<const statement> rewritten =
            static_pointer_cast<const statement>(
                boost::apply_visitor(*this, *i));
        if (rewritten != shared_ptr<const statement>()) {
            stmts.push_back(rewritten);
        }
    } while (i != n.begin());
    reverse(stmts.begin(), stmts.end());
    return make_shared<const suite>(move(stmts));
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
    if (m_used.exists(lhs.id()) ||
        m_protected.exists(lhs.id())) {
        return n.ptr();
    } else {
        return prune::result_type();
    }
}

prune::result_type prune::operator()(const procedure &n) {
    detail::protector p;
    boost::apply_visitor(p, n);
    m_protected = p.modified();
    m_used.begin_scope();
    auto rewritten = rewriter<prune>::operator()(n);
    m_used.end_scope();
    return rewritten;
}

}
