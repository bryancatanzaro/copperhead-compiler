#include "node.hpp"
#include "repr_printer.hpp"

namespace backend {
namespace detail
{

make_node_base_visitor::make_node_base_visitor(void *p) : ptr(p) {}

node_base make_node_base(void *ptr, const node_base &other) {
    return boost::apply_visitor(make_node_base_visitor(ptr), other);
}

} // end detail


//copy constructor requires special handling
node::node(const node &other)
    : super_t(detail::make_node_base(this, other)) {}


std::ostream& operator<<(std::ostream& strm,
                         const node& n) {
    repr_printer rp(strm);
    boost::apply_visitor(rp, n);
    return strm;
}

std::shared_ptr<const node> node::ptr() const {
    return this->shared_from_this();
}


}


    
