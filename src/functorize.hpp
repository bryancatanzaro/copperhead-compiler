#pragma once
#include <string>
#include <set>
#include <sstream>
#include "copier.hpp"
#include "utility/isinstance.hpp"
#include "utility/markers.hpp"
#include "import/library.hpp"
#include "type_convert.hpp"

#include "type_printer.hpp"


namespace backend {

namespace detail {
class type_corresponder
    : public boost::static_visitor<> {
private:
    typedef std::map<std::string,
                     std::shared_ptr<type_t> > type_map;

    std::shared_ptr<type_t> m_working;
    type_map& m_corresponded;
    template<typename Type>
    static inline std::shared_ptr<type_t> get_type_ptr(const Type &n) {
        return std::const_pointer_cast<type_t>(n.shared_from_this());
    }
public:
    typedef std::map<std::string, std::shared_ptr<type_t> >::const_iterator iterator;
    type_corresponder(const std::shared_ptr<type_t>& input,
                      type_map& corresponded)
        : m_working(input), m_corresponded(corresponded) {}

    iterator begin() const {
        return m_corresponded.begin();
    }
    iterator end() const {
        return m_corresponded.end();
    }
    
    void operator()(const monotype_t &n) {
        std::string id = n.name();
        m_corresponded.insert(std::make_pair(id, m_working));
    }

    void operator()(const polytype_t &n) {
        //Polytypes are not allowed to be nested;
        assert(false);
    }

    void operator()(const sequence_t &n) {
        //m_working must be a sequence_t or else the typechecking is wrong
        assert(detail::isinstance<sequence_t>(*m_working));
        const type_t& working_sub = std::static_pointer_cast<sequence_t>(m_working)->sub();
        m_working = get_type_ptr(working_sub);
        boost::apply_visitor(*this, n.sub());
    }

    void operator()(const tuple_t &n) {
        //m_working must be a tuple_t or else the typechecking is wrong
        assert(detail::isinstance<tuple_t>(*m_working));
        const tuple_t& working_tuple = boost::get<const tuple_t&>(*m_working);
        for(auto i = n.begin(),
                j = working_tuple.begin();
            i != n.end();
            ++i, ++j) {
            m_working = get_type_ptr(*j);
            boost::apply_visitor(*this, *i);
        }
        
    }

    void operator()(const fn_t &n) {
        if (detail::isinstance<polytype_t>(*m_working)) {
            //If the fn type is a polytype, we can't harvest any correspondence
            return;
        }
        assert(detail::isinstance<fn_t>(*m_working));
        const fn_t& working_fn = boost::get<const fn_t&>(*m_working);
        m_working = get_type_ptr(working_fn.args());
        boost::apply_visitor(*this, n.args());
        m_working = get_type_ptr(working_fn.result());
        boost::apply_visitor(*this, n.result());
    }

    template<typename N>
    void operator()(const N &n) {
        //Don't harvest correspondences from other types
    }

    
    
};
}


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


    typedef std::map<std::string,
                     std::shared_ptr<type_t> > type_map;

    type_map m_type_map;

    void make_type_map(const apply& n) {
        m_type_map.clear();
        const name& fn_name = n.fn();

        //If function name is not a polytype, the type map should be empty
        if (!detail::isinstance<polytype_t>(fn_name.type()))
            return;
        const polytype_t& fn_polytype = boost::get<const polytype_t&>(fn_name.type());
        //Polytype must contain a function type
        assert(detail::isinstance<fn_t>(fn_polytype.monotype()));
        const fn_t& fn_monotype = boost::get<const fn_t&>(fn_polytype.monotype());
        const tuple_t& fn_arg_t = fn_monotype.args();
        std::vector<std::shared_ptr<type_t> > arg_types;
        for(auto i = n.args().begin();
            i != n.args().end();
            i++) {
            arg_types.push_back(get_type_ptr(i->type()));
        }
        std::shared_ptr<tuple_t> arg_t =
            std::make_shared<tuple_t>(std::move(arg_types));
        detail::type_corresponder tc(arg_t, m_type_map);
        boost::apply_visitor(tc, fn_arg_t);
        
    }

    std::shared_ptr<expression> instantiate_fn(const name& n,
        const type_t& t) {
        std::string id = n.id();
        const type_t& n_t = n.type();
        if (!detail::isinstance<polytype_t>(n_t)) {
            //The function is monomorphic. Instantiate a functor.
            return std::make_shared<apply>(
                std::make_shared<name>(detail::fnize_id(id)),
                std::make_shared<tuple>(std::vector<std::shared_ptr<expression> >{}));
        }
        //Use already populated type map to instantiate the
        //Polymorphic functor with the types it needs in situ

        //The function type is polymorphic
        //Find the monomorphic type
        const polytype_t& n_pt = boost::get<const polytype_t&>(n_t);
        const type_t& n_mt = n_pt.monotype();
        
        //First, create a type map relating the types of the
        //polymorphic function being instantiated to the
        //types in the apply which is instantiating this function.
        type_map fn_to_apl;
        detail::type_corresponder tc(get_type_ptr(t), fn_to_apl);
        boost::apply_visitor(tc, n_mt);

        std::vector<std::shared_ptr<type_t> > instantiated_types;
        for(auto i = n_pt.begin();
            i != n_pt.end();
            i++) {
            std::string fn_t_name = i->name();
            //The name of this type should be in the type map
            assert(fn_to_apl.find(fn_t_name)!=fn_to_apl.end());
            std::shared_ptr<type_t> apl_t = fn_to_apl.find(fn_t_name)->second;
            //The value in the type map should be a monotype
            assert(detail::isinstance<monotype_t>(*apl_t));
            const monotype_t& apl_mt = boost::get<const monotype_t&>(*apl_t);
            //This monotype must exist in the apply type map
            assert(m_type_map.find(apl_mt.name())!=m_type_map.end());
            instantiated_types.push_back(
                m_type_map.find(apl_mt.name())->second);
        }
        std::vector<std::shared_ptr<ctype::type_t> > instantiated_ctypes;
        detail::cu_to_c ctc;
        for(auto i = instantiated_types.begin();
            i != instantiated_types.end();
            i++) {
            instantiated_ctypes.push_back(
                boost::apply_visitor(ctc, **i));
        }
        return std::make_shared<apply>(
            std::make_shared<templated_name>(detail::fnize_id(id),
                                             std::make_shared<ctype::tuple_t>(std::move(instantiated_ctypes)),
                                             get_type_ptr(n.type()),
                                             get_ctype_ptr(n.ctype())),
            std::make_shared<tuple>(std::vector<std::shared_ptr<expression> >{}));

    }
    
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
    using copier::operator();

    result_type operator()(const apply &n) {
        //If the function we're applying is polymorphic,
        //Figure out what types it's being instantiated with
        make_type_map(n);

        
        std::vector<std::shared_ptr<expression> > n_arg_list;
        const tuple& n_args = n.args();
        
        std::shared_ptr<fn_t> fn_type;
        if (detail::isinstance<fn_t>(n.fn().type())) {
            fn_type = std::static_pointer_cast<fn_t>(
                get_type_ptr(n.fn().type()));
        } else{
            //Must be a polytype_t(fn_t)
            assert(detail::isinstance<polytype_t>(n.fn().type()));
            const polytype_t& pt = boost::get<const polytype_t&>(n.fn().type());
            fn_type = std::static_pointer_cast<fn_t>(
                get_type_ptr(pt.monotype()));
        }
        const tuple_t& args_type = fn_type->args();
        auto arg_type = args_type.begin();
        for(auto n_arg = n_args.begin();
            n_arg != n_args.end();
            ++n_arg, ++arg_type) {
            if (!(detail::isinstance<name>(*n_arg)))
                //Fallback if we have something other than a name
                //XXX This might not be necessary when we bind
                //closure objects to identifiers in the program
                n_arg_list.push_back(
                    std::static_pointer_cast<expression>(
                        boost::apply_visitor(*this, *n_arg)));
            else {
                const name& n_name = boost::get<const name&>(*n_arg);
                const std::string id = n_name.id();
                auto found = m_fns.find(id);
                if (found == m_fns.end()) {
                    n_arg_list.push_back(
                        std::static_pointer_cast<expression>(
                            boost::apply_visitor(*this, *n_arg)));
                } else {
                    //We've found a function to instantiate
                    n_arg_list.push_back(
                        instantiate_fn(n_name, *arg_type));
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
            //Add result_type declaration
            assert(detail::isinstance<ctype::fn_t>(n.ctype()));
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
