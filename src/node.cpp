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
    : super_t(detail::make_node_base(this, other))
{
#ifdef DEBUG
    id = ++counter;
    std::cout << "Copying node[" << id << "] from ";
    detail::inspect(other);
    std::cout << std::endl;
#endif
}

#ifdef DEBUG
node::~node() {
    std::cout << "Destroying node[" << id << "]" << std::endl;
}
#endif

#ifdef DEBUG
int node::counter = 0;
#endif

std::ostream& operator<<(std::ostream& strm,
                         const node& n) {
    repr_printer rp(strm);
    boost::apply_visitor(rp, n);
    return strm;
}


}


    
