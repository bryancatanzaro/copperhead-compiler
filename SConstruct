from __future__ import print_function
import os
import re
import subprocess
import operator

#try to import an environment first
try:
    Import('env')
except:
    exec open(os.path.join('config', 'build-env.py'))
    env = Environment()

def CheckVersion(context, cmd, exp, required, extra_error=''):
    context.Message("Checking {cmd} version... ".format(cmd=cmd.split()[0]))

    try:
        log = context.sconf.logstream if context.sconf.logstream else file('/dev/null','w')
        vsout = subprocess.check_output([cmd], shell=True, stderr=log)
    except subprocess.CalledProcessError:
        context.Result('%s was not found %s' % (cmd.split()[0], extra_error) )
        return False
    match = exp.search(vsout)
    if not match:
        context.Result('%s returned unexpected output' % (cmd, extra_error) )
        return False
    version = match.group(1)
    exploded_version = version.split('.')
    if not all(map(operator.le, required, exploded_version)):
        context.Result("%s returned version %s, but we need version %s or better." % (cmd, version, '.'.join(required), extra_error) )
        return False
    context.Result(version)
    return True
thrust_version_check_file = """
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP
#include <thrust/version.h>
#include <iostream>
int main() {
  std::cout << THRUST_MAJOR_VERSION << std::endl;
  std::cout << THRUST_MINOR_VERSION << std::endl;
  std::cout << THRUST_SUBMINOR_VERSION << std::endl;
  return 0;
}
"""

def CheckThrustVersion(context, required_version):
    context.Message("Checking Thrust version...")
    int_required_version = [int(x) for x in required_version]
    result = context.TryRun(thrust_version_check_file, ".cpp")[1]
    returned_version = result.splitlines(False)
    version = '.'.join(returned_version)
    context.Result(version)

    int_returned_version = [int(x) for x in returned_version]
    return all(map(operator.le, int_required_version, int_returned_version))


# Check dependencies
conf=Configure(env, custom_tests = {'CheckVersion':CheckVersion,
                                    'CheckThrustVersion':CheckThrustVersion})
siteconf = {}

# Check to see if the user has written down siteconf stuff
if os.path.exists("siteconf.py"):
    glb = {}
    execfile("siteconf.py", glb, siteconf)
else:

    print("""
*************** siteconf.py not found ***************
We will try building anyway, but may not succeed.
Read the README for more details.
""")
        
    siteconf['THRUST_DIR'] = None
    siteconf['BOOST_INC_DIR'] = None
    siteconf['TBB_INC_DIR'] = None
    siteconf['TBB_LIB_DIR'] = None

    f = open("siteconf.py", 'w')
    for k, v in siteconf.items():
        if v:
            v = '"' + str(v) + '"'
        print('%s = %s' % (k, v), file=f)
        
    f.close()

if siteconf['THRUST_DIR']:
    #Must prepend because build-env might have found an old system Thrust
    env.Prepend(CPPPATH=siteconf['THRUST_DIR'])
if siteconf['BOOST_INC_DIR']:
    env.Append(CPPPATH=siteconf['BOOST_INC_DIR'])
#Ensure we have g++ >= 4.5
gpp_re = re.compile(r'g\+\+ \(.*\) ([\d\.]+)')
conf.CheckVersion('g++ --version', gpp_re, (4,5))

    
# Check dependencies
if not conf.CheckHeader('boost/variant.hpp', language='C++'):
	print("You need the boost::variant library to compile this program")
	Exit(1)
if not conf.CheckHeader('boost/mpl/logical.hpp', language='C++'):
	print("You need the boost::mpl library to compile this program")
	Exit(1)
        
#Check we have a Thrust installation
if not conf.CheckCXXHeader('thrust/host_vector.h'):
    print("You need Thrust version 1.6 to compile this program")
    print("Point us to your Thrust installation by changing THRUST_DIR in siteconf.py")
    Exit(1)

#Ensure Thrust Version > 1.6
if not conf.CheckThrustVersion((1,6)):
    print("You need Thrust version 1.6 or greater")
    print("Change THRUST_DIR in siteconf.py to point to your Thrust installation.")
    Exit(1)


    
try:
    Import('cuda_support')
except:
    # Check to see if we have nvcc >= 4.1
    nv_re = re.compile(r'release ([\d\.]+)')
    cuda_support=conf.CheckVersion('nvcc --version', nv_re, (4,1), extra_error="nvcc was not found. No CUDA support will be included.")
try:
    Import('omp_support')
except:
    omp_support = False
try:
    Import('tbb_support')
except:
    tbb_support = False
    
conf.Finish()

        
#Parallelize the build maximally
import multiprocessing
n_jobs = multiprocessing.cpu_count()
SetOption('num_jobs', n_jobs)

Export('env')
Export('cuda_support')

SConscript('src/SConscript', variant_dir='build', duplicate=0)
