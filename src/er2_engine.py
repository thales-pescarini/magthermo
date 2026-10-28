"""E-R2 exact memoization of pure geometry, with independent reference route.

Energy landscapes and root continuation never contain a magnetic occupation.
Only successful complete producers are published. No physical key is rounded.
The original nineteen numerical source files remain available unchanged.
"""
import collections
import contextlib
import copy
import functools
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import e1_core as core
import general_engine as inherited
import r1_landscape as landscape
import g1_geometry as geometry
import thermal_executor as thermal

ROOT=Path(__file__).resolve().parents[1]
ALGORITHM_VERSION='ER2_EXACT_GEOMETRY_MEMO_V1'
TRANSPORT_R1='E1-TQ01-2026-10-25-UNKNOWN'
TRANSPORT_AMENDED='E1-TQ01-R1'

# Preserve the frozen component benchmark response API without editing it.
prepare_response=inherited.prepare_response
evaluate_duration=inherited.evaluate_duration


def digest(value):
    return hashlib.sha256(json.dumps(core.plain(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@functools.lru_cache(maxsize=1)
def source_signature():
    files=['src/er2_engine.py','src/r1_landscape.py','src/g1_geometry.py','src/e1_core.py','src/grain_api.py',
           'src/thermal_executor.py','src/general_engine.py','src/factorized_executor.py',
           'src/gate1_verified_methods.py','config/B2_CENSUS_POLICY.json','config/protocols.yaml']
    return digest({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files})


def geometry_fingerprint(graph):
    # Include all stationary states and channel order, not just the stable minima.
    return digest({k:graph.get(k) for k in ['identity','points','minima','channels','counts',
                   'connection_source','no_unobserved_heteroclinic_switch_assumption',
                   'canonical_geometry_and_energy','census_agreement','seed_levels']})


class SharedCaches(inherited.SharedCaches):
    def __init__(self,capacity=4096):
        super().__init__();self.capacity=capacity;self.banks=collections.defaultdict(collections.OrderedDict)
        self.er2_counts=collections.Counter();self.er2_cost=collections.Counter();self.current_phase='unspecified'
        self.unique_physical_landscapes=set();self.failures=collections.Counter();self._installed=False
        self.seen_keys=collections.defaultdict(set);self.seen_energies=collections.defaultdict(set)
        self.census_requests=[]

    def publish(self,bank,key,value):
        # Copy before publication, so an exception or caller mutation cannot expose a partial object.
        payload=copy.deepcopy(value);bucket=self.banks[bank]
        bucket[key]=payload;bucket.move_to_end(key)
        while len(bucket)>self.capacity:
            bucket.popitem(last=False);self.er2_counts[bank+':evictions']+=1

    def lookup(self,bank,key):
        bucket=self.banks[bank]
        if key not in bucket:return None
        bucket.move_to_end(key);return copy.deepcopy(bucket[key])

    def physical_key(self,pm,kind,args,kwargs):
        return digest(dict(kind=kind,algorithm=ALGORITHM_VERSION,source_signature=source_signature(),
                           units='SI energy density; B crystal vector tesla; degrees C',
                           energy=pm.identity(),energy_coefficients=dict(k1=pm.k1,k2=pm.k2,D=pm.D,h=pm.h),
                           arguments=args,keyword_arguments=kwargs))

    @contextlib.contextmanager
    def pure_geometry_context(self):
        if self._installed:raise RuntimeError('ER2_CACHE_CONTEXT_REENTRANCE')
        self._installed=True;originals={}
        def wrap(name,original,cacheable=False):
            @functools.wraps(original)
            def call(pm,*args,**kwargs):
                tag='zero' if np.array_equal(np.asarray(pm.B),np.zeros(3)) else 'nonzero'
                counter=name+':'+tag;self.er2_counts[counter+':requests']+=1
                begin=time.monotonic();key=None;use_cache=cacheable
                if cacheable:
                    try:key=self.physical_key(pm,name,args,kwargs)
                    except (TypeError,ValueError):
                        use_cache=False;self.failures[counter+':identity_not_serializable_no_reuse']+=1
                    cached=self.lookup(name,key) if use_cache else None
                    if cached is not None:
                        self.er2_counts[counter+':hits']+=1
                        self.er2_counts['phase:'+str(self.current_phase)+':'+counter+':hits']+=1
                        self.er2_cost[counter+':lookup_inclusive_s']+=time.monotonic()-begin
                        self.er2_cost[counter+':query_inclusive_s']+=time.monotonic()-begin
                        if name=='enumerate_land':self.census_requests.append(dict(T_C=pm.T,B_crystal_T=pm.B.tolist(),phase=self.current_phase,key=key,hit=True,seconds=time.monotonic()-begin))
                        return cached
                    self.er2_counts[counter+':misses']+=1
                    if use_cache:
                        energy=digest(pm.identity())
                        reason=('previous_key_evicted' if key in self.seen_keys[name]
                                else 'new_exact_physical_landscape' if energy not in self.seen_energies[name]
                                else 'same_energy_new_old_roots_or_verification_arguments')
                        self.er2_counts[name+':miss_reason:'+reason]+=1
                        self.seen_keys[name].add(key);self.seen_energies[name].add(energy)
                self.er2_counts[counter+':producer_calls']+=1
                self.er2_counts['phase:'+str(self.current_phase)+':'+counter+':producer_calls']+=1
                try:
                    result=original(pm,*args,**kwargs)
                except Exception:
                    self.failures[counter+':producer_exception']+=1
                    self.er2_cost[counter+':query_inclusive_s']+=time.monotonic()-begin
                    raise
                finally:
                    self.er2_cost[counter+':producer_inclusive_s']+=time.monotonic()-begin
                okay=True
                if name=='enumerate_land':
                    censuses=result.get('censuses',[])
                    okay=(result.get('status')=='COMPUTED_VALID' and result.get('census_agreement') is True
                          and len(censuses)>=2 and all(c.get('euler')==2 and c.get('all_morse') for c in censuses)
                          and result.get('actual_state_count')==len(result.get('minima',[]))
                          and bool(result.get('connections')) and bool(result.get('channels'))
                          and all(c.get('valid') for c in result.get('connections',[])))
                    if okay and key is not None:self.unique_physical_landscapes.add(key)
                elif name=='continuous_points':okay=bool(result[1])
                elif name=='flow':okay=(result.get('destination') is not None and result.get('energy_monotone_decreasing') is True)
                if use_cache and okay:self.publish(name,key,result)
                elif use_cache:self.failures[counter+':incomplete_not_published']+=1
                if name=='enumerate_land':self.census_requests.append(dict(T_C=pm.T,B_crystal_T=pm.B.tolist(),phase=self.current_phase,key=key,hit=False,seconds=time.monotonic()-begin,complete=okay))
                self.er2_cost[counter+':query_inclusive_s']+=time.monotonic()-begin
                return result
            return call
        try:
            # Profiling parents overlap; report these inclusive views without summing them.
            for name,cacheable in [('enumerate_land',True),('continuous_points',True),('flow',True),('connect',False),('census',False),('batch_newton',False)]:
                original=getattr(landscape,name);originals[(landscape,name)]=original
                setattr(landscape,name,wrap(name,original,cacheable))
            original=geometry.refine_kkt;originals[(geometry,'refine_kkt')]=original
            geometry.refine_kkt=wrap('refine_kkt',original,False)
            yield
        finally:
            for (module,name),original in originals.items():setattr(module,name,original)
            self._installed=False

    def er2_snapshot(self):
        return dict(algorithm=ALGORITHM_VERSION,source_signature=source_signature(),counts=dict(self.er2_counts),
                    inclusive_profile_seconds=dict(self.er2_cost),profile_parents_and_children_not_additive=True,
                    objects={k:len(v) for k,v in self.banks.items()},unique_complete_landscapes=len(self.unique_physical_landscapes),
                    failures_not_reused_as_complete=dict(self.failures),capacity_per_kind=self.capacity,
                    rounded_physical_keys=False,propagated_magnetic_states_or_responses_cached=False)


class GeneralTape(inherited.GeneralTape):
    def __init__(self,*args,transport_version=TRANSPORT_R1,**kwargs):
        if transport_version not in [TRANSPORT_R1,TRANSPORT_AMENDED]:raise ValueError('ER2_UNKNOWN_TRANSPORT_VERSION')
        super().__init__(*args,**kwargs);self._transport_version=transport_version
        if not isinstance(self.shared,SharedCaches):raise TypeError('ER2_EXPLICIT_SHARED_CACHE_REQUIRED')
        self.er2_amendment_events=[]

    @property
    def er2_transport_version(self):return self._transport_version

    def _er2_c110_column(self,event,parent_index):
        from er2_transport import eligible_column
        self.shared.er2_counts['amendment_guard_requests']+=1
        result=eligible_column(event,parent_index,self.c,self.er2_transport_version)
        if result is not None:
            self.er2_amendment_events.append(core.plain(result));self.shared.er2_counts['amendment_guard_accepted_columns']+=1
        else:self.shared.er2_counts['amendment_guard_not_proven_or_outside_scope']+=1
        return result

    def _event_reference(self,*args,**kwargs):
        if self.er2_transport_version==TRANSPORT_R1:return super()._event_reference(*args,**kwargs)
        from er2_event_reference import event_reference_er2
        return event_reference_er2(self,*args,**kwargs)

    def prefix_geometry_key(self,preparation_graph):
        return digest(dict(inherited_key=super().prefix_geometry_key(preparation_graph),
                           transport_version=self.er2_transport_version,algorithm=ALGORITHM_VERSION,signature=source_signature()))

    def laboratory_geometry_key(self,land,p,protocol):
        zero_catalog=[h for h in self.critical_catalog if np.array_equal(np.asarray(h['B_sample_T']),np.zeros(3))]
        zero_solutions=[(k,v) for k,v in sorted(self.event_solutions.items()) if self.zero_field_segment(k)]
        return digest(dict(kind='ER2_LABORATORY_GEOMETRY_NO_PLACEHOLDER_V1',s=self.s,c=self.c,R=self.R,R_metadata=self.Rspec,
                           TM=0,protocol=protocol,protocols=self.protocols,resolution=self.level,reference_only=self.reference_only,
                           entry_geometry=geometry_fingerprint(land),zero_critical_catalog=zero_catalog,zero_event_solutions=zero_solutions,
                           partition_policy='reference_first_minimum_recovery_after_failure',transport_version=self.er2_transport_version,
                           source_signature=source_signature()))

    def restore_laboratory_geometry(self,cached,key,p):
        # All map columns were compiled and checked independently of the placeholder.
        # Reapply those maps to THIS placeholder; no prior magnetic state is imported.
        if len(p)!=len(cached['entry_placeholder']):raise core.Unresolved('ER2_LAB_ENTRY_DIMENSION_DRIFT')
        compatible=dict(cached,entry_placeholder=np.array(p,copy=True))
        result=super().restore_laboratory_geometry(compatible,key,p)
        self.laboratory_geometry_reuse.update(placeholder_identity_removed=True,
             reason='Inherited geometry compiler never propagates rates; maps and all columns independent of occupation.',
             current_placeholder_replayed=True,transport_version=self.er2_transport_version)
        self.shared.er2_counts['laboratory_geometry:exact_path_hits']+=1
        return result

    def ramp(self,*args,**kwargs):
        label=kwargs.get('label',args[7] if len(args)>7 else 'unspecified')
        old=self.shared.current_phase;self.shared.current_phase=label
        self.shared.er2_counts['ramps:'+label]+=1
        try:return super().ramp(*args,**kwargs)
        finally:self.shared.current_phase=old

    def event(self,*args,**kwargs):
        self.shared.er2_counts['event_method_calls']+=1;t=time.monotonic()
        try:return super().event(*args,**kwargs)
        finally:self.shared.er2_cost['event_method_inclusive_s']+=time.monotonic()-t

    def align(self,*args,**kwargs):
        self.shared.er2_counts['alignment_method_calls']+=1;t=time.monotonic()
        try:return super().align(*args,**kwargs)
        finally:self.shared.er2_cost['alignment_method_inclusive_s']+=time.monotonic()-t

    def compile(self,protocol='P12'):
        # The independent reference route explicitly bypasses the new shared provider.
        if self.reference_only:
            if self.shared._installed:raise RuntimeError('ER2_INDEPENDENT_REFERENCE_CANNOT_SHARE_ACTIVE_CACHE_CONTEXT')
            return super().compile(protocol)
        with self.shared.pure_geometry_context():result=super().compile(protocol)
        result.update(er2_geometry_profile=self.shared.er2_snapshot(),physical_transport_version=self.er2_transport_version)
        return result
