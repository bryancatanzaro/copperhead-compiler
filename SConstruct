import os

#try to import an environment first
try:
    Import('env')
except:
    exec open(os.path.join('config', 'build-env.py'))
    env = Environment()

try:
    Import('cuda_support')
except:
    cuda_support = False
    
#Parallelize the build maximally
import multiprocessing
n_jobs = multiprocessing.cpu_count()
SetOption('num_jobs', n_jobs)

Export('env')
Export('cuda_support')

SConscript('src/SConscript', variant_dir='build', duplicate=0)
