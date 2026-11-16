"""Opt-in R2 event guard for the already authorized F1 zero-field lab550.

The original R2 event producer retained its 350 C guard after the general
material/thermal/A2 adapters were widened. Only this guard is adapted here.
The original-segment, critical-root, census and transport checks stay intact.
"""
import numpy as np
from portao_a import lateral_anchors

VERSION='PORTAO_A_ZERO_FIELD_LAB_EVENT_DOMAIN_V1'


def anchor_domain(Tl,Tr,Bl,Br):
    if not (20.<=Tl<=550. and 20.<=Tr<=550.):return False
    Bl,Br=np.asarray(Bl),np.asarray(Br)
    if not np.isfinite(Bl).all() or not np.isfinite(Br).all():return False
    if Tl>350. and not np.array_equal(Bl,np.zeros(3)):return False
    if Tr>350. and not np.array_equal(Br,np.zeros(3)):return False
    return max(np.linalg.norm(Bl),np.linalg.norm(Br))<=5e-5*(1+1e-12)


def transforms():
    replacements,extra=lateral_anchors.transforms()
    old='        if not (20<=Tl0<=350 and 20<=Tr0<=350) or max(np.linalg.norm(Bl0),np.linalg.norm(Br0))>5e-5*(1+1e-12):\n'
    new='        if not pa_laboratory_anchor_domain(Tl0,Tr0,Bl0,Br0):\n'
    return replacements+[(old,new)],dict(extra,pa_laboratory_anchor_domain=anchor_domain)
