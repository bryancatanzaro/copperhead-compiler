#pragma once
#include <string>
#include <set>
#include "copier.hpp"
#include "utility/isinstance.hpp"
#include "utility/markers.hpp"
#include "import/library.hpp"

namespace backend {


/*! \p A compiler pass to create function objects for all procedures
 *  except the entry point.
 */
class functorize
    : public copier
{
private:
    const std::string& m_entry_point;
    std::vector<result_type> m_additionals;
    std::set<std::string> m_fns;
    const registry& m_reg;
public:
    /*! \param entry_point The name of the entry point procedure
        \param reg The registry of functions the compiler knows about
     */
    functorize(const std::string& entry_point,
               const registry& reg)
        : m_entry_point(entry_point), m_additionals({}),
          m_reg(reg) {
        for(auto i = reg.fns().cbegin();
            i != reg.fns().cend();
            i++) {
            auto id = i->first;
            std::string fn_name = std::get<0>(id);
            m_fns.insert(fn_name);
        }
               

    }
public:
    using copier::operator();

    result_type operator()(const apply &n) {
        std::vector<std::shared_ptr<expression> > n_arg_list;
        const tuple& n_args = n.args();
        for(auto n_arg = n_args.begin();
            n_arg != n_args.end();
            ++n_arg) {
            if (!(detail::isinstance<name>(*n_arg)))
                n_arg_list.push_back(std::static_pointer_cast<expression>(boost::apply_visitor(*this, *n_arg)));
            else {
                const name& n_name = boost::get<const name&>(*n_arg);
                const std::string id = n_name.id();
                auto found = m_fns.find(id);
                if (found == m_fns.end()) {
                    n_arg_list.push_back(
                        std::static_pointer_cast<expression>(
                            boost::apply_visitor(*this, *n_arg)));
                } else {
                    n_arg_list.push_back(
                        std::shared_ptr<literal>(
                            new literal(detail::fnize_id(id) + "()")));
                }
            }
        }
        auto n_fn = std::static_pointer_cast<name>(this->copier::operator()(n.fn()));
        auto new_args = std::shared_ptr<tuple>(new tuple(std::move(n_arg_list)));
        return std::shared_ptr<apply>(new apply(n_fn, new_args));
    }
    
    result_type operator()(const suite &n) {
        std::vector<std::shared_ptr<statement> > stmts;
        for(auto i = n.begin(); i != n.end(); i++) {
            auto p = std::static_pointer_cast<statement>(boost::apply_visitor(*this, *i));
            stmts.push_back(p);
            while(m_additionals.size() > 0) {
                auto p = std::static_pointer_cast<statement>(m_additionals.back());
                stmts.push_back(p);
                m_additionals.pop_back();
            }
        }
        return result_type(
            new suite(
                std::move(stmts)));
    }
    result_type operator()(const procedure &n) {
        auto n_proc = std::static_pointer_cast<procedure>(this->copier::operator()(n));
        if (n_proc->id().id() != m_entry_point) {
            const ctype::fn_t& n_t = boost::get<const ctype::fn_t&>(
                n.ctype());
            const ctype::type_t& r_t = n_t.result();
            std::shared_ptr<ctype::type_t> origin = get_ctype_ptr(r_t);
            std::shared_ptr<ctype::type_t> rename(
                new ctype::monotype_t("result_type"));
            std::shared_ptr<typedefn> res_defn(
                new typedefn(origin, rename));

            
            std::shared_ptr<tuple> forward_args = std::static_pointer_cast<tuple>(this->copier::operator()(n_proc->args()));
            std::shared_ptr<name> forward_name = std::static_pointer_cast<name>(this->copier::operator()(n_proc->id()));
            std::shared_ptr<apply> op_call(new apply(forward_name, forward_args));
            std::shared_ptr<ret> op_ret(new ret(op_call));
            std::vector<std::shared_ptr<statement> > op_body_stmts{op_ret};
            std::shared_ptr<suite> op_body(new suite(std::move(op_body_stmts)));
            auto op_args = std::static_pointer_cast<tuple>(this->copier::operator()(n.args()));
            std::shared_ptr<name> op_id(new name(std::string("operator()")));
            std::shared_ptr<procedure> op(
                new procedure(
                    op_id, op_args, op_body,
                    get_type_ptr(n.type()),
                    get_ctype_ptr(n.ctype())));
            std::shared_ptr<suite> st_body(new suite(std::vector<std::shared_ptr<statement> >{res_defn, op}));
            std::shared_ptr<name> st_id(new name(detail::fnize_id(n_proc->id().id())));
            std::shared_ptr<structure> st(new structure(st_id, st_body));
            m_additionals.push_back(st);
            m_fns.insert(n_proc->id().id());
        }
        return n_proc;

    }
    
};

}
