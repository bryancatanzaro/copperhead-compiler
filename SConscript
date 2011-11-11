import os
import inspect
import fnmatch

# try to import an environment first
try:
    Import('env')
except:
#    exec open(os.path.join('build', "build-env.py"))
    env = Environment()

#XXX Don't hard code this    
if env['PLATFORM'] == 'darwin':
    env.Append(CPPPATH = "/Users/catanzar/boost_1_47_0")
    env.Append(LIBPATH="/Users/catanzar/boost_1_47_0/stage/lib")
    
env.Append(CPPPATH = "inc")
env.Append(CCFLAGS = "-std=c++0x -Wall -O3")
   

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
    env.SharedObject(target=target, source=x)
    #XXX Don't hardcode this!
    objects.append(target + '.os')

basename = 'build/lib/copperhead'

libcopperhead = env.SharedLibrary(target=basename, source=objects)
Return('libcopperhead')
