"""Deterministic positive-weight reductions with explicit missing support."""
from portao_a.common import *
import math
import numpy as np


def atomic_npz(path,**arrays):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.pending.'+str(os.getpid()))
    with temp.open('wb') as f:
        np.savez(f,**arrays)
        f.flush();os.fsync(f.fileno())
    temp.replace(path)
    fsync_directory(path.parent)


class PositiveAccumulator:
    """A known contribution and its unknown capacity, never a normalized subset."""
    def __init__(self,shape):
        self.known=np.zeros(shape,dtype=np.float64)
        self.compensation=np.zeros(shape,dtype=np.float64)
        self._represented=[]
        self._missing=[]
        self.incorporated=set()

    def add(self,key,weight,response):
        if key in self.incorporated:
            raise ValueError('DUPLICATE_CONTRIBUTION')
        if not np.isfinite(weight) or weight<0:
            raise ValueError('NONNEGATIVE_FINITE_WEIGHT_REQUIRED')
        if response is None:
            self.incorporated.add(key)
            self._missing.append(float(weight))
            return
        value=np.asarray(response,dtype=np.float64)
        if value.shape!=self.known.shape or not np.isfinite(value).all():
            raise ValueError('FINITE_COMPATIBLE_RESPONSE_REQUIRED')
        self.incorporated.add(key)
        self._add_vector(float(weight)*value)
        self._represented.append(float(weight))

    def _add_vector(self,value):
        term=value-self.compensation
        total=self.known+term
        self.compensation=(total-self.known)-term
        self.known=total

    def add_known(self,key,value,represented_capacity,missing_capacity):
        if key in self.incorporated:
            raise ValueError('DUPLICATE_CONTRIBUTION')
        value=np.asarray(value,dtype=np.float64)
        if value.shape!=self.known.shape or not np.isfinite(value).all():
            raise ValueError('FINITE_COMPATIBLE_RESPONSE_REQUIRED')
        if (not np.isfinite([represented_capacity,missing_capacity]).all() or
            min(represented_capacity,missing_capacity)<0):
            raise ValueError('NONNEGATIVE_FINITE_CAPACITY_REQUIRED')
        self.incorporated.add(key)
        self._add_vector(value)
        self._represented.append(float(represented_capacity))
        self._missing.append(float(missing_capacity))

    @property
    def represented_capacity(self):
        return math.fsum(self._represented)

    @property
    def missing_capacity(self):
        return math.fsum(self._missing)

    @property
    def complete(self):
        return self.missing_capacity==0.

    def total(self):
        return np.array(self.known,copy=True) if self.complete else np.full_like(self.known,np.nan)


def verify_raw(path,expected_shape=None):
    directory=Path(path)
    record=read(directory/'RESULT.json')
    if not record.get('complete'):
        return record,None
    data=directory/'Y.npy'
    if sha(data)!=record['Y_sha256']:
        raise ValueError('RAW_ARRAY_HASH_MISMATCH:'+str(data))
    array=np.load(data,allow_pickle=False,mmap_mode='r')
    shape=(len(record['d_nm']),len(record['ells']),2,20,3)
    if array.dtype!=np.float64 or array.shape!=shape or not np.isfinite(array).all():
        raise ValueError('RAW_ARRAY_CONTRACT_VIOLATION:'+str(data))
    if expected_shape is not None and shape!=expected_shape:
        raise ValueError('RAW_ARRAY_AXES_MISMATCH')
    if float(np.max(np.linalg.norm(array,axis=-1)))>1.+1e-10:
        raise ValueError('RAW_UNIT_VECTOR_BOUND_FAILED')
    return record,array


def reduce_rotations(rotation_paths,rotation_weights,output,identity):
    """Persist a kernel integrated in R and exact accounting of missing rotations."""
    if len(rotation_paths)!=len(rotation_weights) or abs(float(np.sum(rotation_weights))-1.)>1e-12:
        raise ValueError('COMPLETE_R_RULE_REQUIRED')
    shape=tuple(identity['shape'])
    accumulator=PositiveAccumulator(shape)
    incorporated=[]
    for index,(path,weight) in enumerate(zip(rotation_paths,rotation_weights)):
        record,array=(None,None) if path is None else verify_raw(path,shape)
        if record is not None:
            for name,expected in [('d_nm',identity['d_nm']),('ells',identity['ells'])]:
                if record[name]!=expected:
                    raise ValueError('RAW_PHYSICAL_GRID_MISMATCH:'+name)
            spec=record['spec']
            if (float(spec['node']['s'])!=float(identity['s']) or
                [float(spec['node'][k]) for k in ('c_x','c_y','c_z')]!=identity['c'] or
                float(spec['T_C'])!=float(identity['T_event_C']) or spec['protocol']!='PX12' or
                spec['rotation_index']!=index or spec.get('R_count',12)!=len(rotation_paths)):
                raise ValueError('RAW_PHYSICAL_IDENTITY_MISMATCH')
        accumulator.add(str(index),float(weight),array)
        incorporated.append(dict(R_index=index,weight=float(weight),complete=array is not None,
             result_path=str(Path(path)/'RESULT.json') if path else None,
             result_sha256=sha(Path(path)/'RESULT.json') if path else None,
             Y_sha256=record.get('Y_sha256') if record else None,
             numerical_signature=record.get('F2_signature') if record else None,
             numeric_extension=record.get('numeric_extension') if record else None))
    output=Path(output)
    atomic_npz(output/'KERNEL.npz',known=accumulator.known,complete_value=accumulator.total(),
               compensation=accumulator.compensation,d_nm=np.asarray(identity['d_nm']),
               ell=np.asarray(identity['ells']))
    metadata=dict(schema='PORTAO_A_R_INTEGRATED_KERNEL_V1',**identity,
        complete_R=accumulator.complete,represented_R_weight=accumulator.represented_capacity,
        missing_R_weight=accumulator.missing_capacity,incorporated_R=incorporated,
        dtype_calculated='float64',dtype_stored='float64',parental_population_weights_applied=False,
        model_order=['F','N1'],axis_order=['diameter','ell','model','readout','vector_component'],
        data_sha256=sha(output/'KERNEL.npz'),reduction_code_sha256=sha(__file__),
        missing_contribution_replaced_with_zero=False)
    save(output/'KERNEL.json',metadata)
    return metadata


def reduce_population(kernel_paths,shape,*,expected_SD_capacity):
    """weights correspond to complete (s,d,c) support, including unavailable nodes."""
    accumulator=PositiveAccumulator(shape)
    sources=[]
    declared_capacity=0.
    for key,path,d_weights,c_weight in kernel_paths:
        w=np.asarray(d_weights,dtype=np.float64)*float(c_weight)
        if np.any(w<0) or not np.isfinite(w).all():
            raise ValueError('INVALID_POPULATION_WEIGHTS')
        capacity=float(w.sum())
        declared_capacity+=capacity
        if path is None:
            accumulator.add(key,capacity,None)
            sources.append(dict(key=key,kernel=None,missing_capacity=capacity))
            continue
        directory=Path(path)
        meta=read(directory/'KERNEL.json')
        if sha(directory/'KERNEL.npz')!=meta['data_sha256']:
            raise ValueError('KERNEL_HASH_MISMATCH')
        with np.load(directory/'KERNEL.npz',allow_pickle=False) as data:
            known=data['known']
            if known.shape!=(len(w),*shape):
                raise ValueError('KERNEL_SHAPE_MISMATCH')
            value=np.tensordot(w,known,axes=(0,0))
        # known already contains R weights. Do not multiply those weights again.
        accumulator.add_known(key,value,capacity*meta['represented_R_weight'],
                              capacity*meta['missing_R_weight'])
        sources.append(dict(key=key,kernel=str(directory),metadata_sha256=sha(directory/'KERNEL.json'),
            data_sha256=meta['data_sha256'],capacity=capacity,complete_R=meta['complete_R']))
    if abs(declared_capacity-expected_SD_capacity)>1e-11:
        raise ValueError('INCOMPLETE_DECLARED_POPULATION_DENOMINATOR')
    return dict(mu_known=accumulator.known,mu_complete=accumulator.total(),complete=accumulator.complete,
        represented_SD_parental_capacity=accumulator.represented_capacity,
        missing_SD_parental_capacity=accumulator.missing_capacity,
        total_SD_parental_capacity=expected_SD_capacity,
        excluded_non_SD_parental_capacity=1.-expected_SD_capacity,
        subset_renormalized=False,sources=sources)
