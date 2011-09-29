#include "../cudata/cudata.h"

__global__ void test_kernel(stored_sequence<float> a) {
    if (threadIdx.x == 0) {
        a[1] = a[1] + 2.78;
    } else if (threadIdx.x == 1) {
        a[0] = a[0] + 3.14;
    }
}

void test(const boost::shared_ptr<cuarray_var>& in) {
    stored_sequence<float> seq = boost::get<cuarray<float> >(*in).get_remote_w();
    test_kernel<<<1, 2>>>(seq);
}
    
BOOST_PYTHON_MODULE(cudata) {
    import_array();
    using namespace boost::python;
    class_<cuarray_var, boost::shared_ptr<cuarray_var> >("CuArray", no_init)
        .def("__init__", make_constructor(make_cuarray))
        .def("__repr__", repr_cuarray);
}
