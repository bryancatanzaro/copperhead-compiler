cudata.so: cudata.cu
	nvcc -arch=sm_20 -I/usr/include/python2.7 -I../prelude -I/Users/catanzar/boost_1_47_0 -I/usr/lib/pymodules/python2.7/numpy/core/include -Xcompiler "-fPIC,-Os" -lboost_python -shared -o cudata.so cudata.cu

adj.so: adj.cu
	nvcc -arch=sm_20 -I/usr/include/python2.7 -I../prelude -I../cudata -I../thrust -I/home/bcatanzaro/thrust -I/Users/catanzar/boost_1_47_0 -I/usr/lib/pymodules/python2.7/numpy/core/include -Xcompiler "-fPIC,-Os" -lboost_python -shared -o adj.so adj.cu

scan_test.so: scan_test.cu
	nvcc -arch=sm_20 -I/usr/include/python2.7 -I../prelude -I../cudata -I../thrust -I/home/bcatanzaro/thrust -I/Users/catanzar/boost_1_47_0 -I/usr/lib/pymodules/python2.7/numpy/core/include -Xcompiler "-fPIC,-Os" -lboost_python -shared -o scan_test.so scan_test.cu


map_test.so: map_test.cu
	nvcc -arch=sm_20 -I/usr/include/python2.7 -I../prelude -I../cudata -I../thrust -I/home/bcatanzaro/thrust -I/Users/catanzar/boost_1_47_0 -I/usr/lib/pymodules/python2.7/numpy/core/include -Xcompiler "-fPIC,-Os" -lboost_python -shared -o map_test.so map_test.cu

axpy_test.so: axpy_test.cu
	nvcc -arch=sm_20 -I/usr/include/python2.7 -I../prelude -I../cudata -I../thrust -I/home/bcatanzaro/thrust -I/Users/catanzar/boost_1_47_0 -I/usr/lib/pymodules/python2.7/numpy/core/include -Xcompiler "-fPIC,-Os" -lboost_python -shared -o axpy_test.so axpy_test.cu

radix_test.so: radix_test.cu
	nvcc -arch=sm_20 -I/usr/include/python2.7 -I../prelude -I../cudata -I../thrust -I/home/bcatanzaro/thrust -I/Users/catanzar/boost_1_47_0 -I/usr/lib/pymodules/python2.7/numpy/core/include -Xcompiler "-fPIC,-Os" -lboost_python -shared -o radix_test.so radix_test.cu