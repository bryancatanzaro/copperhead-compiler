import numpy as np
import cudata

b = np.array([1.0,2.0], dtype=np.float32)
c = cudata.CuArray(b)
print(repr(c))
d = cudata.CuArray(b)
print(repr(d))
cudata.test(d)
