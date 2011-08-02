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
template<class T, class P=py_printer>
const std::string str(std::shared_ptr<T> &p) {
    std::ostringstream os;
    P pp(os);
    pp(*p);
    return os.str();
}

template<class T>
const std::string repr(std::shared_ptr<T> &p) {
    return str<T, repr_printer>(p);
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

template<typename S, typename T>
static std::shared_ptr<T> make_from_list(list vals) {
    std::vector<std::shared_ptr<S> > values;
    boost::python::ssize_t n = len(vals);
    for(boost::python::ssize_t i=0; i<n; i++) {
        object elem = vals[i];
        std::shared_ptr<S> p_elem = extract<std::shared_ptr<S> >(elem);
        values.push_back(p_elem);
    }
    return std::shared_ptr<T>(new T(values));
}


BOOST_PYTHON_MODULE(bindings) {
    class_<node, boost::noncopyable>("Node", no_init);
    class_<expression, bases<node>, boost::noncopyable>("Expression", no_init);
    class_<literal, bases<expression, node>, boost::noncopyable>("Literal", no_init);
    class_<name, std::shared_ptr<name>, bases<literal, expression, node> >("Name", init<std::string>())
        .def("id", &name::id)
        .def("__str__", &backend::str<name>)
        .def("__repr__", &backend::repr<name>);  
    class_<number, std::shared_ptr<number>, bases<literal, expression, node> >("Number", init<std::string>())
        .def("val", &number::val)
        .def("__str__", &backend::str<number>)
        .def("__repr__", &backend::repr<number>);
    class_<backend::tuple, std::shared_ptr<backend::tuple>, bases<expression, node> >("Tuple")
        .def("__init__", make_constructor(make_from_list<expression, backend::tuple>))
        .def("__str__", &backend::str<backend::tuple>)
        .def("__repr__", &backend::repr<backend::tuple>);
    class_<apply, std::shared_ptr<apply>, bases<expression, node> >("Apply", init<std::shared_ptr<name>, std::shared_ptr<backend::tuple> >())
        //.def("fn", &apply::fn) //Need to figure out how to return things
        //.def("args", &apply::args)
        .def("__str__", &backend::str<apply>)
        .def("__repr__", &backend::repr<apply>);
    class_<lambda, std::shared_ptr<lambda>, bases<expression, node> >("Lambda", init<std::shared_ptr<backend::tuple>, std::shared_ptr<expression> >())
        .def("__str__", &backend::str<lambda>)
        .def("__repr__", &backend::repr<apply>);
    class_<statement, bases<node>, boost::noncopyable>("Statement", no_init);
    class_<ret, std::shared_ptr<ret>, bases<statement, node> >("Return", init<std::shared_ptr<expression> >())
        .def("__str__", &backend::str<ret>)
        .def("__repr__", &backend::repr<ret>);
    class_<bind, std::shared_ptr<bind>, bases<statement, node> >("Bind", init<std::shared_ptr<expression>, std::shared_ptr<expression> >())
        .def("__str__", &backend::str<bind>)
        .def("__repr__", &backend::repr<bind>);
    class_<procedure, std::shared_ptr<procedure>, bases<statement, node> >("Procedure", init<std::shared_ptr<name>, std::shared_ptr<backend::tuple>, std::shared_ptr<suite> >())
        .def("__str__", &backend::str<procedure>)
        .def("__repr__", &backend::repr<procedure>);
    class_<suite, std::shared_ptr<suite>, bases<node> >("Suite")
        .def("__init__", make_constructor(make_from_list<statement, suite>))
        .def("__str__", &backend::str<backend::suite>)
        .def("__repr__", &backend::repr<backend::suite>);
        

    implicitly_convertible<std::shared_ptr<backend::expression>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::literal>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::name>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::name>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::name>, std::shared_ptr<backend::literal> >();
    implicitly_convertible<std::shared_ptr<backend::number>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::number>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::number>, std::shared_ptr<backend::literal> >();
    implicitly_convertible<std::shared_ptr<backend::tuple>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::tuple>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::apply>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::apply>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::lambda>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::lambda>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::statement>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::ret>, std::shared_ptr<backend::statement> >();
    implicitly_convertible<std::shared_ptr<backend::ret>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::bind>, std::shared_ptr<backend::statement> >();
    implicitly_convertible<std::shared_ptr<backend::bind>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::procedure>, std::shared_ptr<backend::statement> >();
    implicitly_convertible<std::shared_ptr<backend::procedure>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::suite>, std::shared_ptr<backend::node> >();
}
