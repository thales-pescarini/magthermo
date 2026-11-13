"""Append interior lateral anchors when a verified fold lies near a step edge."""
from portao_a.common import *
import contextlib
import numpy as np
from portao_a import field_events

VERSION='PORTAO_A_INTERIOR_LATERAL_ANCHORS_V1'
SOURCE_MAP=[]


def extend_offsets(offsets,T0,B0,T1,B1,fraction,tolerance):
    result=list(offsets)
    span=abs(T1-T0) if T1!=T0 else float(np.linalg.norm((np.asarray(B1)-np.asarray(B0))/5e-5))
    margin=min(fraction,1.-fraction)*span
    first=min(min(result),margin/4.)
    # These are finite geometric anchors outside the critical enclosure. If
    # no such room remains, keep the unresolved result instead of enlarging a
    # physical segment or accepting an anchor inside its enclosure.
    for value in (first,first/5.,first/25.):
        if value>4*tolerance*span and value not in result:
            result.append(float(value))
    return result


def transforms():
    old='    if retry_offsets is not None:offsets=retry_offsets\n'
    new=old+'    offsets=pa_extend_offsets(offsets,T0,B0,T1,B1,fc,tol)\n'
    return [(old,new)],dict(pa_extend_offsets=extend_offsets)


@contextlib.contextmanager
def installed():
    original=field_events.event_module.event_reference_er2
    assert original is field_events.ORIGINAL_EVENT,'ANCHOR_PATCH_CONTEXT_CONFLICT'
    replacements,extra=transforms()
    producer=field_events.clone(original,replacements,extra)
    SOURCE_MAP[:]=field_events.SOURCE_MAP[-1:]
    try:
        field_events.event_module.event_reference_er2=producer
        yield
    finally:
        field_events.event_module.event_reference_er2=original
