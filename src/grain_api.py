"""Evaluate thermal histories for grains specified by size, shape and orientation."""
import math,time,json,hashlib,copy
import numpy as np
import grain_energy as core
import population_cdf as population
import thermal_executor as thermal

class GrainExecutor(thermal.Executor):
    def __init__(self,d_nm,s,c,R,resolution,protocols,checkpoint=None,R_metadata=None,TM=0):
        d_nm=float(d_nm);s=float(s);c=np.asarray(c,float);R=np.asarray(R,float)
        if c.shape!=(3,) or not np.all(np.isfinite(c)) or abs(np.linalg.norm(c)-1)>1e-12:raise ValueError('c_MUST_BE_UNIT_CRYSTAL_AXIS')
        if R.shape!=(3,3) or not np.all(np.isfinite(R)) or np.max(abs(R.T@R-np.eye(3)))>1e-12 or abs(np.linalg.det(R)-1)>1e-12:raise ValueError('R_MUST_MAP_CRYSTAL_TO_SAMPLE_WITH_DET_PLUS_ONE')
        if not math.isfinite(s) or abs(s)>population.inherited.SMAX+1e-14:raise ValueError('SHAPE_OUTSIDE_SUPPORTED_RANGE')
        if not math.isfinite(d_nm) or not 5<=d_nm<=population.inherited.sd_limit(s):raise ValueError('DIAMETER_OUTSIDE_SHAPE_DEPENDENT_SD_SUPPORT')
        if TM!=0:raise ValueError('MAGNETITE_REQUIRED')
        self.s=s;self.c=c.copy();self.R=R.copy();self.d=d_nm;self.Rspec=copy.deepcopy(R_metadata)
        if self.Rspec is not None:
            import mpmath as mp
            with mp.workdps(85):expected=np.array([float(x) for x in core.high_precision_row(self.R,self.Rspec)])
            if np.max(abs(expected-self.R[2]))>5e-15:raise ValueError('R_METADATA_DOES_NOT_DESCRIBE_RECEIVED_PREPARATION_ROW')
        self.case=dict(d_nm=d_nm,s=s,c=c.copy(),R=R.copy(),TM=TM,R_metadata=self.Rspec)
        self.level=copy.deepcopy(resolution);self.protocols=copy.deepcopy(protocols);self.checkpoint=checkpoint
        self.alignment_failures=[];self.alignment_calls=0;self.maximum_alignment_distance=0.;self.field_connection_checks=[];self.critical_catalog=[];self.event_solutions={};self.pending_critical_hint=None;self.cache={};self.operators={};self.events=[];self.trace=[];self.audit=[];self.cost={k:0. for k in ['landscapes','ramps','events','holds','readings','operators','propagation','serialization']};self.steps=0
        self.min_barrier_kBT=math.inf;self.max_mass_error=0.;self.min_probability=1.;self.start=time.monotonic()
        self.point_id=dict(s=s,c_crystal=self.c,R=self.R,d_nm=d_nm,TM=TM,R_metadata=self.Rspec,code_sha256=thermal.CODE_HASH,
            conventions='explicit_d_nm_s_c_unit_R_crystal_to_sample; number measure then volume exactly once')
        self.geometry_id={k:v for k,v in self.point_id.items() if k not in ['d_nm','R_metadata']}
        self.geometry_id.update(protocols=self.protocols,resolution=self.level)

    def key(self,T,B):
        # Geometry includes the exact crystal field; it never includes diameter.
        value=dict(s=self.s,c=self.c,T_C=float(T),B_crystal_T=self.R.T@np.asarray(B),TM=0,
            protocols=self.protocols,resolution=self.level,code_sha256=thermal.CODE_HASH)
        return hashlib.sha256(json.dumps(core.plain(value),sort_keys=True).encode()).hexdigest()

    def read(self,p,land,label):
        start=time.monotonic();K=land['identity']['Ms']*(self.R@(p@np.array([q['m'] for q in land['minima']])))
        M=1e-18*K;self.cost['readings']+=time.monotonic()-start
        return dict(label=label,K_Am=K,M_Am2=M,Y=M/thermal.S,p=p.copy(),temperature_C=20.,kernel_before_volume_and_S=True)

def sentinel_inputs(case):
    c=np.asarray(case['c_crystal_raw'],float);c/=np.linalg.norm(c)
    if case['R_crystal_to_sample']=='identity':R=np.eye(3);metadata=None
    else:
        record=core.haar_orientations(12)[0];R=np.asarray(record['R_fs']);metadata=record['euler_haar']
    return dict(d_nm=min(45.,.9*population.inherited.sd_limit(case['s'])),s=case['s'],c=c,R=R,R_metadata=metadata)
