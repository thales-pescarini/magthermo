"""Track verified critical points on reverse field ramps, retaining side tests.

The inherited catalog and hint selector handled temperature ramps only. This
extends the same geometric hint mechanism to a fixed-temperature straight
field segment. No event map, rate, occupation, split or physical time changes.
"""
from portao_a.common import *
import contextlib
import copy
import inspect
import textwrap
import numpy as np
import a3_event_sequence as sequence
import er2_event_reference as event_module

VERSION='PORTAO_A_REVERSE_FIELD_CRITICAL_CATALOG_V1'
ORIGINAL_FRAME=sequence._frame
ORIGINAL_EVENT=event_module.event_reference_er2
SOURCE_MAP=[]


def select_hint(catalog,T0,B0,T1,B1):
    B0,B1=np.asarray(B0),np.asarray(B1)
    if T0!=T1 and np.linalg.norm(B1-B0)==0:
        candidates=[x for x in catalog if min(T0,T1)<x['T_C']<max(T0,T1)
                    and np.linalg.norm(np.array(x['B_sample_T'])-B0)<1e-15]
        return min(candidates,key=lambda x:abs(x['T_C']-T0)) if candidates else None
    delta=B1-B0
    denominator=float(delta@delta)
    if T0!=T1 or denominator==0:
        return None
    candidates=[]
    for hint in catalog:
        if hint['T_C']!=T0 or not hint.get('field_critical_verified',False):
            continue
        position=np.asarray(hint['B_sample_T'])
        fraction=float((position-B0)@delta/denominator)
        residual=float(np.linalg.norm(position-(B0+fraction*delta)))
        tolerance=64*np.finfo(float).eps*max(5e-5,float(np.linalg.norm(position)))
        if 0.<fraction<1. and residual<=tolerance:
            selected=copy.deepcopy(hint)
            selected['_field_segment_fraction']=fraction
            selected['field_segment_projection_residual_T']=residual
            selected['field_segment_projection_tolerance_T']=tolerance
            candidates.append(selected)
    return min(candidates,key=lambda x:x['_field_segment_fraction']) if candidates else None


def clone(function,replacements,extra):
    source=textwrap.dedent(inspect.getsource(function))
    changed=source
    for old,new in replacements:
        assert changed.count(old)==1,'FROZEN_EVENT_PATTERN_CHANGED:'+old[:60]
        changed=changed.replace(old,new)
    namespace=dict(function.__globals__,**extra)
    exec(compile(changed,'<'+VERSION+':'+function.__name__+'>','exec'),namespace)
    SOURCE_MAP.append(dict(function=function.__module__+'.'+function.__name__,
        original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        adapted_source_sha256=hashlib.sha256(changed.encode()).hexdigest()))
    return namespace[function.__name__]


def register_field_critical(executor,T0,B0,T1,B1,record):
    B0,B1=np.asarray(B0),np.asarray(B1)
    if T0!=T1 or np.array_equal(B0,B1):return
    critical=record.get('critical_root')
    if record.get('status')!='COMPUTED_VALID' or not critical:return
    f=float(critical['fraction'])
    position=B0+f*(B1-B0)
    for prior in executor.critical_catalog:
        if prior['T_C']==T0 and np.linalg.norm(np.asarray(prior['B_sample_T'])-position)<1e-15:
            return
    executor.critical_catalog.append(dict(T_C=float(T0),B_sample_T=position.copy(),
        critical=copy.deepcopy(critical),field_critical_verified=True,
        producer=VERSION,source_segment=dict(T_C=[float(T0),float(T1)],B_sample_T=[B0.copy(),B1.copy()]),
        source_bracket=copy.deepcopy(record['fraction_bracket']),
        verification='existing full left/right censuses, critical KKT and signed curvature tests passed; no magnetic state cached'))


def build():
    SOURCE_MAP.clear()
    old_hint="""    hint=None
    if T0!=T1 and np.linalg.norm(B1-B0)==0:
        candidates=[x for x in e.critical_catalog if min(T0,T1)<x['T_C']<max(T0,T1) and np.linalg.norm(np.array(x['B_sample_T'])-B0)<1e-15]
        if candidates:hint=min(candidates,key=lambda x:abs(x['T_C']-T0))
"""
    frame=clone(ORIGINAL_FRAME,[(old_hint,'    hint=pa_select_hint(e.critical_catalog,T0,B0,T1,B1)\n'),
        ("'CHANNEL_TOPOLOGY_CHANGE','NONPOSITIVE_BARRIER_NO_CLIPPING'",
         "'CHANNEL_TOPOLOGY_CHANGE','NONPOSITIVE_BARRIER_NO_CLIPPING','STATE_COUNT_CHANGED_WITHOUT_EVENT','CONTINUED_STATE_NOT_IN_TARGET_CENSUS'")],
        dict(pa_select_hint=select_hint))
    producer=clone(ORIGINAL_EVENT,[
        ("fc=(hint['T_C']-T0)/(T1-T0);lo=", 
         "fc=(hint['_field_segment_fraction'] if '_field_segment_fraction' in hint else (hint['T_C']-T0)/(T1-T0));lo=")],{})
    def event(executor,T0,B0,T1,B1,old):
        result=producer(executor,T0,B0,T1,B1,old)
        register_field_critical(executor,T0,B0,T1,B1,executor.events[-1])
        return result
    return frame,event


@contextlib.contextmanager
def installed():
    assert sequence._frame is ORIGINAL_FRAME,'EVENT_SEQUENCE_PATCH_CONTEXT_CONFLICT'
    assert event_module.event_reference_er2 is ORIGINAL_EVENT,'EVENT_PRODUCER_PATCH_CONTEXT_CONFLICT'
    frame,event=build()
    try:
        sequence._frame=frame
        event_module.event_reference_er2=event
        yield
    finally:
        sequence._frame=ORIGINAL_FRAME
        event_module.event_reference_er2=ORIGINAL_EVENT
