import os
import inspect
import fnmatch

# try to import an environment first
try:
    Import('env')
except:
    exec open(os.path.join('build', "build-env.py"))
    env = Environment()


#XXX Don't hard code this    
if platform == 'darwin':
    env.Append(CPPPATH = "/Users/catanzar/boost_1_47_0")
    env.Append(LIBPATH="/Users/catanzar/boost/boost_1_47_0/stage/lib")
    env.Append(LINKFLAGS="-F/System/Library/Frameworks/ -framework Python")

env.Append(CPPPATH = "/usr/include/python2.7")
env.Append(CPPPATH = "/usr/lib/pymodules/python2.7/numpy/core/include")    
env.Append(LIBS = ['boost_python'])

cppenv = env.Clone()
cudaenv = env.Clone()

cppenv.Append(CCFLAGS = "-std=c++0x -Wall -Os")

cudaenv.Append(CPPPATH = "../cudata")
cudaenv.Append(CPPPATH = "../prelude")
cudaenv.Append(NVCCFLAGS = '-arch=sm_20')
cudaenv.Append(LIBS = ['cudart'])


object_files = []
for root, dirnames, filenames in os.walk(os.path.join(os.curdir,
                                                      "src")):
  for filename in fnmatch.filter(filenames, '*.cpp'):
      object_files.append(os.path.join(root, filename))



objects = []
for x in object_files:
    head, tail = os.path.split(x)
    basename, extension = os.path.splitext(tail)
    target = os.path.join(os.path.join("build", "obj"), os.path.join(head, basename))
    cppenv.SharedObject(target=target, source=x)
    #XXX Don't hardcode this!
    objects.append(target + '.os')
    

for x in Glob('python/*.cpp'):
    basename, extension = os.path.splitext(str(x))
    cppenv.SharedLibrary(target=basename, source=[x]+objects, SHLIBPREFIX='', SHLIBSUFFIX='.so')

for x in Glob('python/*.cu'):
    basename, extension = os.path.splitext(str(x))
    cudaenv.SharedLibrary(target=basename, source=x, SHLIBPREFIX='', SHLIBSUFFIX='.so')