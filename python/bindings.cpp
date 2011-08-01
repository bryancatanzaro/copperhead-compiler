#include <boost/python.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/make_constructor.hpp>
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

void test(std::shared_ptr<backend::node> in) {
    std::ostringstream os;
    repr_printer rp(os);
    boost::apply_visitor(rp, *in);
    std::cout << os.str() << std::endl;
}

static std::shared_ptr<backend::tuple> make_backend_tuple(list vals) {
    std::vector<std::shared_ptr<expression> > values;
    boost::python::ssize_t n = len(vals);
    for(boost::python::ssize_t i=0; i<n; i++) {
        object elem = vals[i];
        std::shared_ptr<expression> p_elem = extract<std::shared_ptr<expression> >(elem);
        values.push_back(p_elem);
    }
    return std::shared_ptr<backend::tuple>(new backend::tuple(values));
}


BOOST_PYTHON_MODULE(bindings) {
    class_<node, boost::noncopyable>("Node", no_init);
    class_<expression, bases<node>, boost::noncopyable>("Expression", no_init);
    //class_<literal, bases<node, expression>, boost::noncopyable>("Literal", no_init);
    class_<name, std::shared_ptr<name>, bases<expression, node> >("Name", init<std::string>())
        .def("id", &name::id)
        .def("__str__", &backend::str<name, py_printer>)
        .def("__repr__", &backend::str<name, repr_printer>);  
    class_<number, std::shared_ptr<number>, bases<expression, node> >("Number", init<std::string>())
        .def("val", &number::val)
        .def("__str__", &backend::str<number, py_printer>)
        .def("__repr__", &backend::str<number, repr_printer>);
    class_<backend::tuple, std::shared_ptr<backend::tuple>, bases<expression, node> >("Tuple")
        .def("__init__", make_constructor(make_backend_tuple))
        .def("__str__", &backend::str<backend::tuple, py_printer>)
        .def("__repr__", &backend::str<backend::tuple, repr_printer>);
    implicitly_convertible<std::shared_ptr<backend::name>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::name>, std::shared_ptr<backend::expression> >();
    //implicitly_convertible<std::shared_ptr<backend::name>, std::shared_ptr<backend::literal> >();
    implicitly_convertible<std::shared_ptr<backend::number>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::number>, std::shared_ptr<backend::expression> >();
    //implicitly_convertible<std::shared_ptr<backend::number>, std::shared_ptr<backend::literal> >();
    implicitly_convertible<std::shared_ptr<backend::tuple>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::tuple>, std::shared_ptr<backend::expression> >();
}
