"""Exact reusable spectral/operator factors; every magnetic state stays local."""
import collections
import copy
import functools
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import grain_energy as core
import general_engine as original
import thermal_executor as thermal
from relaxation_spectrum import spectral_F,observer

VERSION='SPECTRAL_RESPONSE_CACHE_V1'


@functools.lru_cache(maxsize=1)
def numerical_id():
    root=Path(__file__).resolve().parent
    files=['response_compiler.py','general_engine.py','relaxation_spectrum.py','thermal_executor.py','grain_energy.py']
    return tuple((n,hashlib.sha256((root/n).read_bytes()).hexdigest()) for n in files)


def response_graph_key(graph):
    # The stable-energy route also consumes stationary saddle seeds and its
    # existing energetic evidence, beyond the original local spectrum key.
    fields=['status','identity','points','minima','channels','canonical_geometry_and_energy','stable_energy_evidence']
    value={k:graph.get(k) for k in fields}
    return hashlib.sha256(json.dumps(core.plain(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class ResponseCache:
    def __init__(self,spectral_capacity=8192,operator_capacity=65536):
        self.spectra=collections.OrderedDict()
        self.operators=collections.OrderedDict()
        self.spectral_capacity=spectral_capacity
        self.operator_capacity=operator_capacity
        self.counts=collections.Counter()
        self.cost=collections.Counter()

    def acquire(self,bank,key,producer,capacity,name):
        if key in bank:
            bank.move_to_end(key);self.counts[name+'_hits']+=1
            return bank[key]
        start=time.monotonic();value=producer()
        self.cost[name+'_producer_exclusive_s']+=time.monotonic()-start
        self.counts[name+'_producers']+=1
        bank[key]=value
        while len(bank)>capacity:
            bank.popitem(last=False);self.counts[name+'_evictions']+=1
        return value

    def snapshot(self):
        return dict(version=VERSION,counts=dict(self.counts),exclusive_producer_s=dict(self.cost),
                    spectra=len(self.spectra),operators=len(self.operators),
                    magnetic_occupations_cached=False,all_F_modes_retained=True,
                    exact_vector_and_ordered_geometry_identity=True)


class ResponseCompiler:
    """Compile graph identities once per immutable tape, then prepare each d."""
    def __init__(self,tape,R,cache=None):
        # Isolate the graph data from caller mutation. Only the inherited
        # stable-energy evidence cache may be added inside this private copy.
        self.tape=copy.deepcopy(tape)
        self.R=np.array(R,copy=True)
        self.cache=cache if cache is not None else ResponseCache()
        self.actions=[]
        identities={}
        for action in self.tape['actions']:
            if action['kind']=='operator':
                graph=action['graph'];token=id(graph)
                if token not in identities:
                    identities[token]=(graph,response_graph_key(graph))
                else:
                    assert identities[token][0] is graph
                self.actions.append((action,(identities[token][1],float(action['T_C']))))
            else:self.actions.append((action,None))
        self.source_tape_actions_unchanged=True

    def prepare(self,d_nm,progress=None):
        started=time.monotonic()
        prefix={m:np.array(self.tape['initial_p'],float,copy=True) for m in ['F','N1']}
        readouts={'F':[],'N1':[]};current={};hold=None
        max_mass=0.;minimum=1.;n_modes=set()
        Rkey=self.R.tobytes()
        for i,(action,graph_key) in enumerate(self.actions):
            kind=action['kind']
            if kind=='operator':
                graph=action['graph']
                skey=(graph_key,float(d_nm),Rkey,VERSION,numerical_id())
                def make_spectrum():
                    sp=spectral_F(graph,d_nm,action['T_C'])
                    selected=sp.select(observer(graph,self.R))
                    return sp,selected
                sp,selected=self.cache.acquire(self.cache.spectra,skey,make_spectrum,self.cache.spectral_capacity,'spectral_selection')
                n_modes.add(sp.n)
                if action['label']=='event_hold':
                    if hold is not None:raise RuntimeError('MULTIPLE_EVENT_HOLDS')
                    hold=sp,selected
                    current={m:np.eye(sp.n) for m in prefix}
                    continue
                matrices={}
                for model in prefix:
                    key=(skey,float(action['seconds']),model)
                    def make_operator():
                        U,info=sp.propagator(action['seconds'],model,selected)
                        U.setflags(write=False)
                        return U,dict(info)
                    U,info=self.cache.acquire(self.cache.operators,key,make_operator,self.cache.operator_capacity,'operator_'+model)
                    matrices[model]=U
                    max_mass=max(max_mass,info['mass_residual'])
                    minimum=min(minimum,info['minimum'])
            elif kind=='map':
                matrices={m:np.asarray(action['matrix'],float) for m in prefix}
            elif kind=='read':
                if hold is None:raise RuntimeError('READING_BEFORE_EVENT_HOLD')
                W=observer(action['graph'],self.R)*(1e-18/thermal.S)
                for model in prefix:readouts[model].append(W@current[model])
                continue
            else:raise RuntimeError('UNKNOWN_TAPE_ACTION')
            for model,U in matrices.items():
                if hold is None:
                    prefix[model]=U@prefix[model]
                    thermal.checked(prefix[model],'E_prefix_'+model)
                else:
                    current[model]=U@current[model]
                    if not np.all(np.isfinite(current[model])):raise core.Unresolved('NONFINITE_COMPOSED_MAP')
                    error=float(np.max(abs(current[model].sum(axis=0)-1)))
                    if error>2e-11 or np.min(current[model]) < -2e-11:
                        raise core.Unresolved('COMPOSED_MAP_INVARIANTS_UNRESOLVED')
            if progress and i%128==0:
                progress(dict(phase='spectra_response',action=i,total_actions=len(self.actions),d_nm=d_nm))
        if hold is None:raise core.Unresolved('PARTIAL_TAPE_DID_NOT_REACH_EVENT_HOLD')
        return dict(prefix=prefix,readouts=readouts,hold=hold,
                    profile=dict(wall_s=time.monotonic()-started,modes=sorted(n_modes),
                                 max_MP_mass_residual=max_mass,minimum_MP_probability=minimum,
                                 exact_response_cache=self.cache.snapshot()))


evaluate_duration=original.evaluate_duration
