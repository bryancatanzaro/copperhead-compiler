#include "ctype.hpp"
#include "utility/initializers.hpp"

using std::shared_ptr;
using std::static_pointer_cast;
using std::make_shared;
using std::string;
using std::vector;
using backend::utility::make_vector;


namespace backend {

namespace ctype {

namespace detail {
make_type_base_visitor::make_type_base_visitor(void *p)
    : ptr(p){}

type_base make_type_base(void *ptr, const type_base &other) {
    return boost::apply_visitor(make_type_base_visitor(ptr), other);
}

}

type_t::type_t(const type_t &other)
    : super_t(detail::make_type_base(this, other)) {}

std::shared_ptr<const type_t> type_t::ptr() const {
    return this->shared_from_this();
}

monotype_t::monotype_t(const string &name)
        : type_t(*this),
          m_name(name),
          m_params() {}
    
const string& monotype_t::name(void) const {
    return m_name;
}

monotype_t::const_iterator monotype_t::begin() const {
    return boost::make_indirect_iterator(m_params.cbegin());
}

monotype_t::const_iterator monotype_t::end() const {
    return boost::make_indirect_iterator(m_params.cend());
}

monotype_t::const_ptr_iterator monotype_t::p_begin() const {
    return m_params.cbegin();
}

monotype_t::const_ptr_iterator monotype_t::p_end() const {
    return m_params.cend();
}

int monotype_t::size() const {
    return m_params.size();
}


//! The instantiated type object representing the Int32 type
shared_ptr<monotype_t> int32_mt = make_shared<monotype_t>("int");
//! The instantiated type object representing the Int64 type
shared_ptr<monotype_t> int64_mt = make_shared<monotype_t>("long");
//! The instantiated type object representing the Uint32 type
shared_ptr<monotype_t> uint32_mt = make_shared<monotype_t>("unsigned int");
//! The instantiated type object representing the Uint64 type
shared_ptr<monotype_t> uint64_mt = make_shared<monotype_t>("unsigned long");
//! The instantiated type object representing the Float32 type
shared_ptr<monotype_t> float32_mt = make_shared<monotype_t>("float");
//! The instantiated type object representing the Float64 type
shared_ptr<monotype_t> float64_mt = make_shared<monotype_t>("double");
//! The instantiated type object representing the Bool type
shared_ptr<monotype_t> bool_mt = make_shared<monotype_t>("bool");
//! The instantiated type object representing the Void type
shared_ptr<monotype_t> void_mt = make_shared<monotype_t>("void");

sequence_t::sequence_t(const shared_ptr<type_t> &sub)
    : monotype_t(*this,
                 "sequence",
                 make_vector<shared_ptr<type_t> >(sub)) {}

template<typename Derived>
sequence_t::sequence_t(Derived& self,
                       const std::string& name,
                       const std::shared_ptr<type_t>& sub)
    : monotype_t(self,
                 name,
                 utility::make_vector<std::shared_ptr<type_t> >(sub)) {}

const type_t& sequence_t::sub() const {
    return *m_params[0];
}

shared_ptr<type_t> sequence_t::p_sub() const {
    return m_params[0];
}

tuple_t::tuple_t(vector<shared_ptr<type_t> > && sub)
    : monotype_t(*this,
                 "Tuple",
                 std::move(sub)) {}

fn_t::fn_t(const shared_ptr<tuple_t> args,
           const shared_ptr<type_t> result)
    : monotype_t(*this,
                 "Fn",
                 make_vector<shared_ptr<type_t> >(args)(result)) {}


const tuple_t& fn_t::args() const {
    return boost::get<const tuple_t&>(*m_params[0]);
}

const type_t& fn_t::result() const {
    return *m_params[1];
}

shared_ptr<tuple_t> fn_t::p_args() const {
    return static_pointer_cast<tuple_t>(m_params[0]);
}

shared_ptr<type_t> fn_t::p_result() const {
    return m_params[1];
}


cuarray_t::cuarray_t(const shared_ptr<type_t> sub) :
    sequence_t(*this, "sp_cuarray", sub) {}


polytype_t::polytype_t(vector<shared_ptr<type_t> > && vars,
                       shared_ptr<monotype_t> monotype)
    : type_t(*this), m_vars(std::move(vars)), m_monotype(monotype) {}

const monotype_t& polytype_t::monotype() const {
    return *m_monotype;
}

shared_ptr<monotype_t> polytype_t::p_monotype() const {
    return m_monotype;
}

polytype_t::const_iterator polytype_t::begin() const {
    return boost::make_indirect_iterator(m_vars.cbegin());
}

polytype_t::const_iterator polytype_t::end() const {
    return boost::make_indirect_iterator(m_vars.cend());
}

polytype_t::const_ptr_iterator polytype_t::p_begin() const {
    return m_vars.cbegin();
}

polytype_t::const_ptr_iterator polytype_t::p_end() const {
    return m_vars.cend();
}

}
}
