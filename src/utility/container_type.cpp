#include "utility/container_type.hpp"
#include "utility/isinstance.hpp"
#include <iostream>
using std::shared_ptr;
using std::vector;
using std::make_shared;

namespace backend {
namespace detail {

shared_ptr<const ctype::type_t> container_type(const ctype::type_t& t) {
    if (detail::isinstance<ctype::sequence_t>(t)) {
        const ctype::sequence_t seq = boost::get<const ctype::sequence_t&>(t);
        return make_shared<const ctype::cuarray_t>(
            seq.sub().ptr());
    } else if (!detail::isinstance<ctype::tuple_t>(t)) {
        return t.ptr();
    }
    bool match = true;
    const ctype::tuple_t& t_tuple = boost::get<const ctype::tuple_t&>(t);
    vector<shared_ptr<const ctype::type_t> > sub_types;
    for(auto i = t_tuple.begin(); i != t_tuple.end(); i++) {
        shared_ptr<const ctype::type_t> sub_container = container_type(*i);
        match = match && (sub_container == i->ptr());
        sub_types.push_back(container_type(*i));
    }
    if (match) {
        return t.ptr();
    }
    return make_shared<const ctype::tuple_t>(move(sub_types));
}

}
}
