#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/implicit.hpp>
#include <boost/variant.hpp>
#include "node.hpp"
#include "expression.hpp"
#include "py_printer.hpp"
#include "repr_printer.hpp"

#include <sstream>
#include <iostream>

namespace backend {
template<class T>
T* get_pointer(std::shared_ptr<T> const &p) {
    return p.get();
}
template<class T, class P>
const std::string str(std::shared_ptr<T> &p) {
    std::ostringstream os;
    P pp(os);
    pp(*p);
    return os.str();
}

}
using namespace boost::python;
using namespace backend;

typedef std::vector<std::shared_ptr<node> > node_vector;


void test(std::shared_ptr<backend::node> in) {
    std::ostringstream os;
    repr_printer rp(os);
    boost::apply_visitor(rp, *in);
    std::cout << os.str() << std::endl;
}

BOOST_PYTHON_MODULE(variant) {
    class_<node_vector>("node_vector")
        .def(vector_indexing_suite<node_vector, true>());
    class_<node, boost::noncopyable>("Node", no_init);
    class_<expression, bases<node>, boost::noncopyable>("Expression", no_init);
    class_<name, std::shared_ptr<name>, bases<expression, node> >("Name", init<std::string>())
        .def("id", &name::id)
        .def("__str__", &backend::str<name, py_printer>)
        .def("__repr__", &backend::str<name, repr_printer>);  
    class_<number, std::shared_ptr<number>, bases<expression, node> >("Number", init<std::string>())
        .def("val", &number::val)
        .def("__str__", &backend::str<number, py_printer>)
        .def("__repr__", &backend::str<number, repr_printer>);
    implicitly_convertible<std::shared_ptr<backend::name>, std::shared_ptr<backend::node> >();
    def("test", test);
}
