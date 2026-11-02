"""Explicit chronological tasks in one original prescribed step, A3 family.

Only control flow and guard bookkeeping live here. Event geometry, rates,
maps and magnetic propagation continue to use the inherited producers.
"""
import hashlib
import json
import math
import numpy as np
from e1_core import Unresolved

MAX_EVENTS = 32
SIGNATURE_VERSION = 'A3_EXACT_HEX_PHYSICAL_FRONTIER_V1'


def exact(value):
    if isinstance(value, np.ndarray):return exact(value.tolist())
    if isinstance(value, np.generic):return exact(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):raise Unresolved('UNKNOWN_EVENT_PROGRESS:NONFINITE_IDENTITY')
        return {'float64_hex': value.hex()}
    if isinstance(value, dict):return {str(k):exact(v) for k,v in sorted(value.items(),key=lambda x:str(x[0]))}
    if isinstance(value, (tuple,list)):return [exact(v) for v in value]
    if isinstance(value, (str,int,bool)) or value is None:return value
    raise TypeError('A3_UNSUPPORTED_SIGNATURE_VALUE:'+type(value).__name__)


def signature(value):
    return hashlib.sha256(json.dumps(exact(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def physical_graph(land):
    return {k:land.get(k) for k in ['identity','points','minima','channels','canonical_geometry_and_energy']}


class Step:
    def __init__(self,e,args):
        _,old,T0,B0,T1,B1,seconds,label=args
        self.identity=dict(point=e.point_id,protocols=e.protocols,resolution=e.level,
            T0=float(T0),B0=np.asarray(B0),T1=float(T1),B1=np.asarray(B1),
            seconds=float(seconds),label=label,initial_landscape=physical_graph(old))
        self.identity=exact(self.identity);self.original_signature=signature(self.identity)
        self.duration=float(seconds);self.discovered=set();self.rollbacks=0
        self.reset_committed_trial()

    def reset_committed_trial(self):
        # Called at construction or the existing proven whole-trial rollback.
        # The distinct-event budget belongs to the original step and survives.
        self.frontier=0.;self.last_event_alpha=-math.inf
        self.applied=set();self.frontiers=set();self.intervals=[];self.times=[]

    def enter(self,old,T0,B0,T1,B1,a,b):
        if not (0<=a<b<=1):raise Unresolved('UNKNOWN_EVENT_PROGRESS:PENDING_INTERVAL')
        key=signature(dict(step_sha256=self.original_signature,landscape=physical_graph(old),
            pending=dict(T0=float(T0),B0=B0,T1=float(T1),B1=B1,alpha=[a,b])))
        token=(key,self.frontier)
        if token in self.frontiers:raise Unresolved('UNKNOWN_EVENT_CYCLE:REPEATED_PENDING_FRONTIER')
        self.frontiers.add(token)

    def discover(self,record,left,right,T0,B0,T1,B1,a,b):
        critical=record['critical_root'];f=float(critical['fraction'])
        critical_without_local_fraction={k:v for k,v in critical.items() if k!='fraction'}
        alpha=a+f*(b-a)
        key=signature(dict(step_sha256=self.original_signature,critical=critical_without_local_fraction,
            physical_location=dict(T_C=T0+f*(T1-T0),B_sample_T=B0+f*(B1-B0)),
            left=physical_graph(left),right=physical_graph(right),map=record['map']))
        if key not in self.discovered:
            if len(self.discovered)>=MAX_EVENTS:raise Unresolved('UNKNOWN_EVENT_CAP')
            self.discovered.add(key)
        return key,alpha

    def consume(self,a,b,durations,kind):
        if a!=self.frontier or not (a<b<=1):raise Unresolved('UNKNOWN_EVENT_PROGRESS:NONADJACENT_OR_EMPTY_CONSUMPTION')
        if not all(math.isfinite(x) and x>=0 for x in durations):raise Unresolved('UNKNOWN_EVENT_PROGRESS:INVALID_DURATION')
        self.intervals.append(dict(alpha=[a,b],seconds=list(durations),kind=kind))
        self.times.extend(durations);self.frontier=b

    def apply_event(self,key,alpha):
        if key in self.applied:raise Unresolved('UNKNOWN_EVENT_CYCLE:REPEATED_APPLIED_EVENT')
        if not self.last_event_alpha<alpha:raise Unresolved('UNKNOWN_EVENT_PROGRESS:EVENT_ORDER')
        self.applied.add(key);self.last_event_alpha=alpha

    def finish(self,checked):
        if self.frontier!=1.:raise Unresolved('UNKNOWN_EVENT_PROGRESS:UNCONSUMED_TAIL')
        # Reuse the inherited no-correction conservation guard. No new time
        # tolerance, rescaling of physical durations or normalization of Y.
        if self.duration>0:checked(np.asarray(self.times)/self.duration,'A3_original_step_time_partition')
        elif self.duration<0 or any(self.times):raise Unresolved('UNKNOWN_EVENT_PROGRESS:ZERO_OR_NEGATIVE_TIME')

    def evidence(self,status):
        return dict(signature_version=SIGNATURE_VERSION,status=status,
            original_step_sha256=self.original_signature,distinct_events=len(self.discovered),
            applied_events=len(self.applied),maximum_events=MAX_EVENTS,
            transaction_rollbacks=self.rollbacks,frontier_alpha=self.frontier,
            intervals=self.intervals,prescribed_seconds=self.duration,
            consumed_seconds=math.fsum(self.times),
            time_residual_s=math.fsum(self.times)-self.duration)


def prescribed(e,method,*args):
    if getattr(e,'_a3_original_step',None) is not None:
        raise Unresolved('UNKNOWN_EVENT_CYCLE:REENTRANT_ORIGINAL_STEP')
    ctx=Step(e,args);e._a3_original_step=ctx
    outcome='UNKNOWN'
    try:
        result=method(*args)
        from thermal_executor import checked
        ctx.finish(checked);outcome='CONSUMED_COMPLETE'
        return result
    finally:
        if ctx.discovered or outcome!='CONSUMED_COMPLETE':
            if not hasattr(e,'a3_sequence_audit'):e.a3_sequence_audit=[]
            e.a3_sequence_audit.append(ctx.evidence(outcome))
        e._a3_original_step=None


def rollback_original_trial(e):
    ctx=getattr(e,'_a3_original_step',None)
    if ctx is not None:
        ctx.rollbacks+=1;ctx.reset_committed_trial()


def _frame(e,ctx,p,old,T0,B0,T1,B1,seconds,label,depth,a,b):
    from thermal_executor import core,checked
    ctx.enter(old,T0,B0,T1,B1,a,b)
    B0=np.asarray(B0);B1=np.asarray(B1);Tm=(T0+T1)/2;Bm=(B0+B1)/2
    input_p=np.array(p,copy=True)
    transaction=e.transaction_token()
    hint=None
    if T0!=T1 and np.linalg.norm(B1-B0)==0:
        candidates=[x for x in e.critical_catalog if min(T0,T1)<x['T_C']<max(T0,T1) and np.linalg.norm(np.array(x['B_sample_T'])-B0)<1e-15]
        if candidates:hint=min(candidates,key=lambda x:abs(x['T_C']-T0))
    e.pending_critical_hint=hint
    try:
        if hint is not None:raise core.Unresolved('STATIONARY_EVENT_CANDIDATE')
        mid=e.landscape(Tm,Bm,old);end=e.landscape(T1,B1,mid)
        p=e.transport(e.align(old,mid,p),p,label+':align_mid');p=e.apply(p,mid,Tm,seconds,label);p=e.transport(e.align(mid,end,p),p,label+':align_end');checked(p,label+':identity')
        ctx.consume(a,b,[seconds],'ordinary')
        return p,end
    except core.Unresolved as exc:
        if str(exc) not in ['STATIONARY_EVENT_CANDIDATE','IDENTITY_CONTINUATION_REQUIRES_EVENT','CHANNEL_TOPOLOGY_CHANGE','NONPOSITIVE_BARRIER_NO_CLIPPING']:raise
    e.rollback_transaction(transaction)
    p=input_p.copy()
    lo,hi,left,right,P=e.event(T0,B0,T1,B1,old)
    own_record=e.events[-1]
    if not (0<=lo<hi<=1):raise Unresolved('UNKNOWN_EVENT_PROGRESS:EVENT_BRACKET')
    Tl=T0+lo*(T1-T0);Bl=B0+lo*(B1-B0);Tr=T0+hi*(T1-T0);Br=B0+hi*(B1-B0)
    al=a+lo*(b-a);ah=a+hi*(b-a)
    if not (a<=al<ah<=b):raise Unresolved('UNKNOWN_EVENT_PROGRESS:UNREPRESENTABLE_EVENT_INTERVAL')
    key,alpha=ctx.discover(own_record,left,right,T0,B0,T1,B1,a,b)
    event_left=left
    if lo>0:
        if Tl==T0 and np.array_equal(Bl,B0):raise Unresolved('UNKNOWN_EVENT_PROGRESS:UNREPRESENTABLE_PRE_FRONTIER')
        p,left=yield (p,old,T0,B0,Tl,Bl,seconds*lo,label+':pre',depth+1,a,al)
        p=e.transport(e.align(left,event_left,p),p,label+':align_event_left');left=event_left
    ctx.apply_event(key,alpha)
    p=e.apply(p,left,Tl,seconds*(hi-lo)/2,label+':event_left')
    before_map=p.copy();p=e.transport(P,p,label+':event_map',own_record);map_check=checked(p,label+':event_map')
    own_record.update(occupation_before_map=before_map,occupation_after_map=p.copy(),probability_check=map_check,physical_step_seconds=seconds,time_partition_seconds=[seconds*lo,seconds*(hi-lo)/2,seconds*(hi-lo)/2,seconds*(1-hi)],event_segment_label=label)
    p=e.apply(p,right,Tr,seconds*(hi-lo)/2,label+':event_right')
    ctx.consume(al,ah,[seconds*(hi-lo)/2,seconds*(hi-lo)/2],'event')
    if hi<1:
        if Tr==T1 and np.array_equal(Br,B1):raise Unresolved('UNKNOWN_EVENT_PROGRESS:UNREPRESENTABLE_POST_FRONTIER')
        p,end=yield (p,right,Tr,Br,T1,B1,seconds*(1-hi),label+':post',depth+1,ah,b)
    else:end=right
    return p,end


def advance(e,p,old,T0,B0,T1,B1,seconds,label,depth=0):
    ctx=getattr(e,'_a3_original_step',None)
    if ctx is None:
        return prescribed(e,lambda *xs:advance(e,*xs),p,old,T0,B0,T1,B1,seconds,label)
    stack=[_frame(e,ctx,p,old,T0,B0,T1,B1,seconds,label,depth,0.,1.)]
    value=None
    while stack:
        try:
            request=stack[-1].send(value);value=None
        except StopIteration as done:
            stack.pop();value=done.value;continue
        stack.append(_frame(e,ctx,*request))
    return value
