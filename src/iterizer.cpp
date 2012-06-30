#include "iterizer.hpp"
#include "utility/initializers.hpp"
#include "utility/snippets.hpp"

using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using std::vector;
using std::move;
using std::string;
using std::tuple;
using backend::utility::make_vector;

namespace backend {

namespace detail {

class analyze_recursion
    : public rewriter<analyze_recursion> {
private:
    string m_proc_name;
    bool m_in_else_branch;
    bool m_nested;
    bool m_recursive;
    bool m_sense;
    std::shared_ptr<const expression> m_pred;
    std::shared_ptr<const name> recursion_result;
public:
    analyze_recursion() : m_in_else_branch(false), m_nested(false),
                          m_recursive(false), m_sense(false),
                          m_pred() {}

    
    using rewriter<analyze_recursion>::operator();

    result_type operator()(const bind& n) {
        if (detail::isinstance<apply>(n.rhs())) {
            const apply& rhs = boost::get<const apply&>(n.rhs());
            if (rhs.fn().id() != m_proc_name) {
                return n.ptr();
            }
            if (m_nested) {
                throw std::domain_error("Recursive function call found in non-tail position");
            }
            if (!detail::isinstance<name>(n.lhs())) {
                return n.ptr();
            }
            
            const name& lhs = boost::get<const name&>(n.lhs());
            recursion_result = lhs.ptr();
        }
        return n.ptr();
    }

    result_type operator()(const ret& r) {
        if (detail::isinstance<name>(r.val())) {
            const name& n = boost::get<const name&>(r.val());
            if (recursion_result != shared_ptr<const name>()) {
                if (m_pred == shared_ptr<const expression>()) {
                    throw std::domain_error(
                        "Tail recursive calls must be enclosed in a conditional.");
                }
                if (n.id() == recursion_result->id()) {
                    m_recursive = true;
                    m_sense = m_in_else_branch;
                } else {
                    throw std::domain_error("Recursive function call found in non-tail position");
                }
            }
        }
        return r.ptr();
    }
    
    result_type operator()(const conditional& c) {
        m_nested = m_nested | (m_pred != shared_ptr<const expression>());
        m_pred = c.cond().ptr();
        m_in_else_branch = false;
        recursion_result = shared_ptr<const name>();
        boost::apply_visitor(*this, c.then());
        m_in_else_branch = true;
        recursion_result = shared_ptr<const name>();
        boost::apply_visitor(*this, c.orelse());
        return c.ptr();
    }
    
    result_type operator()(const procedure& p) {
        m_proc_name = p.id().id();
        recursion_result = shared_ptr<const name>();
        m_recursive = false;
        return rewriter<analyze_recursion>::operator()(p);
    }

    bool recursive() const {
        return m_recursive;
    }
    bool sense() const {
        return m_sense;
    }
};

class normalize_sense
    : public rewriter<normalize_sense> {
private:
    bool m_sense;
public:
    normalize_sense(bool sense): m_sense(sense) {}
    using rewriter<normalize_sense>::operator();

    result_type operator()(const suite& s) {
        vector<shared_ptr<const statement> > stmts;
        bool found_nested = false;
        for(auto i = s.begin(); i != s.end(); i++) {
            result_type rewritten = boost::apply_visitor(*this, *i);
            if (detail::isinstance<suite>(*rewritten)) {
                found_nested = true;
                const suite& nested_suite = boost::get<const suite&>(*rewritten);
                for(auto j = nested_suite.begin();
                    j != nested_suite.end();
                    j++) {
                    stmts.push_back(j->ptr());
                }
            } else {
                stmts.push_back(i->ptr());
            }
        }
        if (!found_nested) {
            return s.ptr();
        } else {
            return make_shared<const suite>(move(stmts));
        }
    }

    result_type operator()(const conditional& c) {
        if (!m_sense) {
            return c.ptr();
        }
        vector<shared_ptr<const statement> > stmts;
        
        //We need to invert the sense of this conditional
        shared_ptr<const name> inverted_cond =
            make_shared<const name>(
                "inverted",
                bool_mt);
        shared_ptr<const expression> cond = c.cond().ptr();
        //Flatten conditional expression to avoid creating nested expressions
        if (!detail::isinstance<name>(c.cond())) {
            shared_ptr<const name> cond_name =
                make_shared<const name>(
                    "conditional",
                    bool_mt);
            stmts.push_back(
                make_shared<const bind>(
                    cond_name,
                    cond));
            cond = cond_name;
        }
        shared_ptr<const apply> make_inverted =
            make_shared<const apply>(
                make_shared<const name>("op_not"),
                make_shared<const tuple>(
                    make_vector<shared_ptr<const expression> >(cond)));
        shared_ptr<const bind> inverted_bind =
            make_shared<const bind>(inverted_cond, make_inverted);
        stmts.push_back(inverted_bind);
        shared_ptr<const conditional> inverted_conditional =
            make_shared<const conditional>(
                inverted_cond,
                c.orelse().ptr(),
                c.then().ptr());
        stmts.push_back(inverted_conditional);
        return make_shared<const suite>(move(stmts));
    }
            
    
};

class make_loop
    : public rewriter<make_loop> {
private:
    vector<shared_ptr<const statement> > m_pre;
    const procedure& m_enclosing;
public:

    make_loop(const procedure& enclosing) : m_enclosing(enclosing) {}
    using rewriter<make_loop>::operator();
    
    result_type operator()(const suite& s) {
        vector<shared_ptr<const statement> > stmts;
        bool found_nested = false;
        for(auto i = s.begin(); i != s.end(); i++) {
            result_type rewritten = boost::apply_visitor(*this, *i);
            if (detail::isinstance<suite>(*rewritten)) {
                found_nested = true;
                const suite& nested_suite = boost::get<const suite&>(*rewritten);
                for(auto j = nested_suite.begin();
                    j != nested_suite.end();
                    j++) {
                    stmts.push_back(j->ptr());
                }
            } else {
                stmts.push_back(i->ptr());
            }
        }
        if (!found_nested) {
            return s.ptr();
        } else {
            return make_shared<const suite>(move(stmts));
        }
                
    }

    result_type operator()(const bind& b) {
        m_pre.push_back(b.ptr());
        return b.ptr();
    }
    
    result_type operator()(const conditional& c) {
        vector<shared_ptr<const statement> > stmts;
        vector<shared_ptr<const statement> > while_stmts;
        std::cout << "conditional" << std::endl;
        for(auto j = c.then().begin();
            j != c.then().end();
            j++) {
            bool accounted = false;
            if (detail::isinstance<bind>(*j)) {
                const bind& b = boost::get<const bind&>(*j);
                if (detail::isinstance<apply>(b.rhs())) {
                    const apply& a = boost::get<const apply&>(b.rhs());
                    if (a.fn().id() == m_enclosing.id().id()) {
                        accounted = true;
                        for(auto k = a.args().begin(),
                                l = m_enclosing.args().begin();
                            k != a.args().end();
                            k++, l++) {
                            bool assigned = false;
                            //Formal parameters to procedures must be names
                            assert(detail::isinstance<name>(*l));
                            const name& formal_name = boost::get<const name&>(*l);
                            if (detail::isinstance<name>(*k)) {
                                const name& result_name = boost::get<const name&>(*k);
                                //If result name is the same as formal name, no assign
                                if (result_name.id() == formal_name.id()) {
                                    assigned = true;
                                }   
                            }
                            if (!assigned) {
                                while_stmts.push_back(
                                    make_shared<const bind>(
                                        formal_name.ptr(),
                                        k->ptr()));
                            }
                        }
                    }
                }
            } else if (detail::isinstance<ret>(*j)) {
                accounted = true;
            }
            if (!accounted) {
                while_stmts.push_back(j->ptr());
            }
        }
        while_stmts.insert(while_stmts.end(), m_pre.begin(), m_pre.end());
        stmts.push_back(
            make_shared<const while_block>(
                c.cond().ptr(),
                make_shared<const suite>(
                    move(while_stmts))));
        for(auto i = c.orelse().begin();
            i != c.orelse().end();
            i++) {
            stmts.push_back(i->ptr());
        }
        auto result = make_shared<const suite>(move(stmts));
        return result;
    }
};

}

iterizer::result_type iterizer::operator()(const procedure& p) {
    detail::analyze_recursion a;
    boost::apply_visitor(a, p);
    //Return if procedure is not tail recursive
    if (!a.recursive()) {
        return p.ptr();
    }
    
    detail::normalize_sense s(a.sense());
    auto normalized = boost::apply_visitor(s, p);

    detail::make_loop l(p);
    return boost::apply_visitor(l, *normalized);
        
}

}
