"""Deterministic positive-weight reductions with explicit missing support."""
from portao_a.common import *
import math
import numpy as np


def atomic_npz(path,*,compressed=False,**arrays):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.pending.'+str(os.getpid()))
    with temp.open('wb') as f:
        (np.savez_compressed if compressed else np.savez)(f,**arrays)
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


def numerical_contract(record):
    required=('F2_signature','base_version','transport','protocol_sha256','resolution','dtype',
              'environment_sha256')
    if any(key not in record for key in required) or 'numeric_extensions' not in record['spec']:
        raise ValueError('RAW_NUMERICAL_IDENTITY_REQUIRED')
    return {**{key:record[key] for key in required},
            'numeric_extensions':record['spec']['numeric_extensions']}


def kernel_file(directory,metadata):
    name=metadata.get('data_file','KERNEL.npz')
    if Path(name).name!=name or not name.endswith('.npz'):
        raise ValueError('UNSAFE_KERNEL_DATA_PATH')
    return Path(directory)/name


def read_kernel(directory):
    directory=Path(directory)
    metadata=read(directory/'KERNEL.json')
    path=kernel_file(directory,metadata)
    if sha(path)!=metadata['data_sha256']:
        raise ValueError('KERNEL_HASH_MISMATCH')
    with np.load(path,allow_pickle=False) as data:
        arrays={key:np.array(data[key],copy=True) for key in data.files}
    return metadata,arrays


def reduce_rotations(rotation_paths,rotation_weights,output,identity):
    """Commit a canonical R prefix atomically; resume never adds an R twice.

    A missing R stops the prefix, so the final compensated sum has the same
    order after any crash. Later available rotations remain explicitly pending.
    New data generations precede the atomic metadata commit. Existing generations
    are removed only after the new committed pair has been read and verified.
    """
    weights=[float(w) for w in rotation_weights]
    if (len(rotation_paths)!=len(weights) or not weights or
        any(not math.isfinite(w) or w<=0 for w in weights) or abs(math.fsum(weights)-1.)>1e-12):
        raise ValueError('COMPLETE_POSITIVE_R_RULE_REQUIRED')
    if 'numerical_contract' not in identity:
        raise ValueError('EXPECTED_NUMERICAL_IDENTITY_REQUIRED')
    output=Path(output)
    shape=tuple(identity['shape'])
    identity_hash=digest(dict(identity=identity,R_weights=weights))
    accumulator=PositiveAccumulator(shape)
    incorporated=[]
    old_path=None
    prior=None
    if (output/'KERNEL.json').exists():
        prior,arrays=read_kernel(output)
        if prior.get('identity_sha256')!=identity_hash:
            raise ValueError('KERNEL_RESUME_IDENTITY_MISMATCH')
        incorporated=prior['incorporated_R']
        if [r['R_index'] for r in incorporated]!=list(range(len(incorporated))):
            raise ValueError('KERNEL_NONCANONICAL_R_PREFIX')
        accumulator.known=arrays['known']
        if (accumulator.known.shape!=shape or accumulator.known.dtype!=np.float64 or
            not np.isfinite(accumulator.known).all()):
            raise ValueError('KERNEL_KNOWN_SHAPE_OR_VALUES_INVALID')
        if prior['complete_R']:
            return prior
        accumulator.compensation=arrays['compensation']
        if (accumulator.compensation.shape!=shape or accumulator.compensation.dtype!=np.float64 or
            not np.isfinite(accumulator.compensation).all()):
            raise ValueError('KERNEL_COMPENSATION_INVALID')
        old_path=kernel_file(output,prior)
    previous_count=len(incorporated)
    pending_cause='NOT_YET_AVAILABLE'
    for index in range(previous_count,len(weights)):
        path=rotation_paths[index]
        if path is None:
            break
        directory=Path(path)
        if not (directory/'RESULT.json').exists():
            break
        if not (directory/'UNIT_RECEIPT.json').exists():
            pending_cause='ATOMIC_UNIT_RECEIPT_NOT_YET_AVAILABLE'
            break
        if read(directory/'UNIT_RECEIPT.json')['result_sha256']!=sha(directory/'RESULT.json'):
            raise ValueError('RAW_ATOMIC_RECEIPT_HASH_MISMATCH')
        record,array=verify_raw(directory,shape)
        if array is None:
            pending_cause=record.get('failure_kind','NUMERICAL_UNRESOLVED')+':'+record.get('reason','')
            break
        if any(record[name]!=identity[name] for name in ('d_nm','ells')):
            raise ValueError('RAW_PHYSICAL_GRID_MISMATCH')
        spec=record['spec']
        if (float(spec['node']['s'])!=float(identity['s']) or
            [float(spec['node'][k]) for k in ('c_x','c_y','c_z')]!=identity['c'] or
            float(spec['T_C'])!=float(identity['T_event_C']) or spec['protocol']!='PX12' or
            spec['rotation_index']!=index or spec.get('R_count',12)!=len(weights)):
            raise ValueError('RAW_PHYSICAL_IDENTITY_MISMATCH')
        if digest(numerical_contract(record))!=digest(identity['numerical_contract']):
            raise ValueError('RAW_NUMERICAL_IDENTITY_MISMATCH')
        if 'R_matrices' in identity and record.get('rotation_matrix')!=identity['R_matrices'][index]:
            raise ValueError('RAW_HAAR_ROTATION_MATRIX_MISMATCH')
        accumulator._add_vector(weights[index]*array)
        incorporated.append(dict(R_index=index,weight=weights[index],complete=True,
            result_path=str(directory/'RESULT.json'),result_sha256=sha(directory/'RESULT.json'),
            Y_sha256=record['Y_sha256']))
    if prior is not None and len(incorporated)==previous_count:
        return prior
    complete=len(incorporated)==len(weights)
    generation=digest(dict(identity=identity_hash,incorporated=incorporated))[:24]
    path=output/f'KERNEL.{generation}.npz'
    arrays=dict(known=accumulator.known,d_nm=np.asarray(identity['d_nm']),ell=np.asarray(identity['ells']))
    if not complete:
        arrays['compensation']=accumulator.compensation
    atomic_npz(path,compressed=True,**arrays)
    metadata=dict(schema='PORTAO_A_R_INTEGRATED_KERNEL_V2',**identity,identity_sha256=identity_hash,
        complete_R=complete,represented_R_weight=math.fsum(weights[:len(incorporated)]),
        missing_R_weight=math.fsum(weights[len(incorporated):]),incorporated_R=incorporated,
        pending_R=list(range(len(incorporated),len(weights))),first_pending_cause=None if complete else pending_cause,
        R_weights=weights,dtype_calculated='float64',dtype_stored='float64',
        parental_population_weights_applied=False,model_order=['F','N1'],
        axis_order=['diameter','ell','model','readout','vector_component'],
        data_file=path.name,data_sha256=sha(path),reduction_code_sha256=sha(__file__),
        missing_contribution_replaced_with_zero=False,canonical_summation_order='R_index increasing')
    save(output/'KERNEL.json',metadata)
    committed,_=read_kernel(output)
    assert committed['data_sha256']==metadata['data_sha256']
    if old_path is not None and old_path!=path:
        old_path.unlink();fsync_directory(output)
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
        meta,data=read_kernel(directory)
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
