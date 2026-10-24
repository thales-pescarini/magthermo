"""Geometry tape compiled once per (s,c,R); independent all-mode advance per d.

The compiler never evaluates a rate or a magnetic response. Its normalized
placeholder only checks the dimensions/stochasticity of linear state maps.
Scalar references do not import or consume a tape, operator or occupation.
"""
import copy,json,time,hashlib,collections,math
import numpy as np
import e1_core as core
import thermal_executor as thermal
from grain_api import GrainExecutor

def digest(v):return hashlib.sha256(json.dumps(core.plain(v),sort_keys=True,separators=(',',':')).encode()).hexdigest()

class SharedCaches:
    def __init__(self):self.zero_graphs={};self.zero_origin={};self.operators={};self.operator_origin={};self.counts=collections.Counter();self.cost=collections.Counter()
    def zero_key(self,e,T):return digest(dict(s=e.s,c=e.c,T_C=float(T),TM=0,protocols=e.protocols,resolution=e.level,code_sha256=thermal.CODE_HASH))
    def operator(self,e,land,T,d,seconds):
        # Ordered physical roots/channels are part of the key. No occupations.
        identity=dict(identity=land['identity'],minima=[dict(m=p['m'],E=p.get('E_mp',p['E']),point_id=p['point_id']) for p in land['minima']],
            channels=[dict(saddle_id=c['saddle_id'],endpoints=c['endpoints'],E=c.get('saddle_E_mp',c['saddle_E'])) for c in land['channels']],
            d_nm=float(d),T_C=float(T),seconds=float(seconds),protocols=e.protocols,resolution=e.level,code_sha256=thermal.CODE_HASH)
        key=digest(identity)
        if key in self.operators:
            self.counts['operator_hits']+=1
            if self.operator_origin[key]!=digest(e.R):self.counts['operator_cross_R_hits']+=1
            return self.operators[key]
        self.counts['operator_misses']+=1;t=time.monotonic();value=thermal.operator(land,T,d,seconds);self.cost['operator_assembly_s']+=time.monotonic()-t
        self.operators[key]=value;self.operator_origin[key]=digest(e.R);return value

class GeometryTape(GrainExecutor):
    def __init__(self,*args,shared=None,**kwargs):
        super().__init__(*args,**kwargs);self.shared=shared or SharedCaches();self.actions=[];self.readings=[];self.initial_p=None
    def landscape(self,T,B,previous=None,force=False):
        if np.array_equal(np.asarray(B),np.zeros(3)):
            key=self.shared.zero_key(self,T)
            if key in self.shared.zero_graphs:
                self.shared.counts['zero_geometry_hits']+=1
                if self.shared.zero_origin[key]!=digest(self.R):self.shared.counts['zero_geometry_cross_R_hits']+=1
                land=self.shared.zero_graphs[key];self.cache[self.key(T,B)]=land;return land
            self.shared.counts['zero_geometry_misses']+=1;land=super().landscape(T,B,previous,force);self.shared.zero_graphs[key]=land;self.shared.zero_origin[key]=digest(self.R);return land
        return super().landscape(T,B,previous,force)
    def apply(self,p,land,T,seconds,label):
        if self.initial_p is None:self.initial_p=p.copy()
        self.actions.append(dict(kind='operator',graph=land,T_C=float(T),seconds=float(seconds),label=label));self.steps+=1
        return p.copy()
    def transport(self,P,p,label,event_record=None):
        if self.initial_p is None:self.initial_p=p.copy()
        self.actions.append(dict(kind='map',matrix=P.copy(),label=label,event=event_record))
        return P@p
    def transaction_token(self):return len(self.actions)
    def rollback_transaction(self,token):
        self.shared.counts['uncommitted_tape_actions_discarded']+=len(self.actions)-token
        del self.actions[token:]
    def read(self,p,land,label):
        self.actions.append(dict(kind='read',graph=land,label=label))
        return dict(label=label,geometry_only_placeholder=True)
    def compile(self,protocol='P12'):
        t=time.monotonic();v=self.run(protocol)
        for event in self.events:
            for k in ['occupation_before_map','occupation_after_map','probability_check']:event.pop(k,None)
            event['geometry_only_occupations_not_propagated']=True
        self.trace=[{k:v for k,v in row.items() if k not in ['state','forked_probability']} for row in self.trace]
        return dict(status='COMPUTED_VALID',kind='geometry_sequence_not_magnetic_trajectory',actions=self.actions,initial_p=self.initial_p,
            events=self.events,trace=self.trace,point_geometry={k:v for k,v in self.point_id.items() if k!='d_nm'},wall_s=time.monotonic()-t,
            geometry_cost_s=self.cost,alignment_calls=self.alignment_calls,maximum_alignment_distance=self.maximum_alignment_distance,
            field_connection_checks=self.field_connection_checks,new_diameter_trajectories=0)

def advance_size(e,tape,d,shared,on_checkpoint=None):
    """All probabilities and observables are fresh for this diameter and R."""
    start=time.monotonic();cpu=time.process_time();p=np.array(tape['initial_p'],copy=True);readings=[];transports=[];mass=0.;minimum=1.;barrier=math.inf;steps=0;prop_s=0.;read_s=0.;operator_request_s=0.
    for j,action in enumerate(tape['actions']):
        kind=action['kind']
        if kind=='operator':
            t=time.monotonic();U,info=shared.operator(e,action['graph'],action['T_C'],d,action['seconds']);operator_request_s+=time.monotonic()-t
            t=time.monotonic();p=U@p;prop_s+=time.monotonic()-t;barrier=min(barrier,info['min_barrier_over_kBT']);steps+=1
        elif kind=='map':
            before=p.copy();p=action['matrix']@p
            if action['event'] is not None:transports.append(dict(event=action['event'],occupation_before_map=before,occupation_after_map=p.copy(),probability_check=thermal.checked(p,action['label'])))
        else:
            t=time.monotonic();g=action['graph'];K=g['identity']['Ms']*(e.R@(p@np.array([x['m'] for x in g['minima']])));Y=1e-18*K/thermal.S
            readings.append(dict(label=action['label'],K_Am=K,Y=Y,p=p.copy(),temperature_C=20.));read_s+=time.monotonic()-t
        check=thermal.checked(p,action.get('label',kind));mass=max(mass,abs(check['mass_residual_before_correction']));minimum=min(minimum,check['minimum_probability'])
        if on_checkpoint and j%128==0:on_checkpoint(dict(d_nm=d,action=j,total_actions=len(tape['actions']),readings=readings,probability=p,operators=steps))
    return dict(status='COMPUTED_VALID',d_nm=float(d),point_identity=dict(e.point_id,d_nm=float(d)),readings=readings,transports=transports,
        trace=tape['trace'],events=tape['events'],wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,
        maximum_mass_error=mass,minimum_probability=minimum,min_barrier_kBT=barrier if math.isfinite(barrier) else None,
        low_barrier_alert=barrier<3,operators=steps,propagation_s=prop_s,readings_s=read_s,operator_requests_inclusive_s=operator_request_s,
        physical_population_curve=False,geometry_rebuilt_for_this_size=False)
