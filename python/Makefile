cudata.so: cudata.cu
	nvcc -arch=sm_20 -I/usr/include/python2.7 -I../prelude -I/Users/catanzar/boost_1_47_0 -I/usr/lib/pymodules/python2.7/numpy/core/include -Xcompiler "-fPIC,-Os" -lboost_python -shared -o cudata.so cudata.cu

test.so: test.cu
	nvcc -arch=sm_20 -I/usr/include/python2.7 -I../prelude -I../cudata -I../thrust -I/home/bcatanzaro/thrust -I/Users/catanzar/boost_1_47_0 -I/usr/lib/pymodules/python2.7/numpy/core/include -Xcompiler "-fPIC,-Os" -lboost_python -shared -o test.so test.cu