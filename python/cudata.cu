#include "../src/cudata/cudata.h"
    
BOOST_PYTHON_MODULE(cudata) {
    import_array();
    using namespace boost::python;
    class_<cuarray_var, boost::shared_ptr<cuarray_var> >("CuArray", no_init)
        .def("__init__", make_constructor(make_cuarray))
        .def("__repr__", repr_cuarray);
}
