import os 


__version__ = "0.3.0"


# allow multiple OpenMP instances
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# change workdir to package root
os.chdir(os.path.dirname(os.path.realpath(__file__)))
