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

# Check dependencies
conf=Configure(env, custom_tests = {'CheckVersion':CheckVersion})
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
        
    siteconf['THRUST_PATH'] = None
    siteconf['BOOST_INC_DIR'] = None
    f = open("siteconf.py", 'w')
    for k, v in siteconf.items():
        if v:
            v = '"' + str(v) + '"'
        print('%s = %s' % (k, v), file=f)
        
    f.close()

if siteconf['THRUST_PATH']:
    #Must prepend because build-env might have found an old system Thrust
    env.Prepend(CPPPATH=siteconf['THRUST_PATH'])
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
    print("Point us to your Thrust installation by changing THRUST_PATH in siteconf.py")
    Exit(1)

#XXX Insert Thrust version check
    
try:
    Import('cuda_support')
except:
    # Check to see if we have nvcc >= 4.1
    nv_re = re.compile(r'release ([\d\.]+)')
    cuda_support=conf.CheckVersion('nvcc --version', nv_re, (4,1), extra_error="nvcc was not found. No CUDA support will be included.")
conf.Finish()

        
#Parallelize the build maximally
import multiprocessing
n_jobs = multiprocessing.cpu_count()
SetOption('num_jobs', n_jobs)

Export('env')
Export('cuda_support')

SConscript('src/SConscript', variant_dir='build', duplicate=0)
