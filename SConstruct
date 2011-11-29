import os

#try to import an environment first
try:
    Import('env')
except:
    exec open(os.path.join('config', 'build-env.py'))
    env = Environment()

#Parallelize the build maximally
import multiprocessing
n_jobs = multiprocessing.cpu_count()
SetOption('num_jobs', n_jobs)

Export('env')


SConscript('src/SConscript', variant_dir='build', duplicate=0)
