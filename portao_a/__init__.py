"""Current owner-authorized Portao A driver, separate from historical campaigns."""
import os
import sys

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'JULIA_NUM_THREADS',
              'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('JAX_ENABLE_X64', 'true')
sys.dont_write_bytecode = True
