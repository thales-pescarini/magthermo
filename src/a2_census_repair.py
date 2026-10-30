"""A2 bounded seed refinement after the exact inherited exhausted-census class.

Only side censuses are retried. The caller keeps its original critical root,
offsets, continuation, channels and physical transport. Normal successful
events do not call this module. Root lists never cross candidate/reference.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
VERSION='ER3_A2_FINITE_LATERAL_CENSUS_REPAIR_V1'


def eligible(record):
    if record.get('reason')!='FINITE_LATERAL_ANCHORS_EXHAUSTED':return False
    anchors=record.get('anchors',[])
    if not anchors or any(a.get('status')=='COMPUTED_VALID' for a in anchors):return False
    return any(g.get('reason')=='CENSUS_DISAGREEMENT_OR_NON_MORSE' and g.get('census_agreement') is False
               for a in anchors for g in [a.get('left_graph',{}),a.get('right_graph',{})])


def event_key(executor,T0,B0,T1,B1):
    data=dict(s=float(executor.s),c=np.asarray(executor.c).tolist(),R=np.asarray(executor.R).tolist(),
              event_T_C=float(executor.event_T_C),T_segment=[float(T0),float(T1)],
              B_segment_T=[np.asarray(B0).tolist(),np.asarray(B1).tolist()])
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def policy():
    data=json.loads((ROOT/'config/CENSUS_REPAIR_A2.json').read_text())
    if data['candidate_pairs']!=[[12,24],[24,48]] or data['control_pairs']!=[[24,48],[48,96]] or data['maximum_seed_level']!=96:
        raise ValueError('A2_CENSUS_SEQUENCE_CHANGED')
    return data


def recover_anchor_censuses(executor,T0,B0,T1,B1,old,offsets,record,persist):
    if not eligible(record):return None
    role=getattr(executor,'a2_census_role',None)
    # Old callers without an explicit authorized A2 context keep old behavior.
    if role not in ['candidate','reference']:return None
    import e1_core as core
    import r1_landscape as ls
    p=policy();key=event_key(executor,T0,B0,T1,B1)
    if role=='candidate':pairs=p['candidate_pairs']
    else:
        mapping=getattr(executor,'a2_reference_resolution_targets',{})
        if key not in mapping:raise core.Unresolved('A2_REFERENCE_RESOLUTION_TARGET_NOT_BOUND')
        chosen=list(mapping[key])
        if chosen not in p['candidate_pairs']:raise core.Unresolved('A2_REFERENCE_INVALID_CANDIDATE_LEVEL_METADATA')
        pairs=[p['control_pairs'][p['candidate_pairs'].index(chosen)]]
    temperature=abs(T1-T0)>0
    original_anchors=list(record['anchors'])
    started=time.monotonic()
    activation=dict(policy=VERSION,event_key=key,role=role,
        event_T_C=float(executor.event_T_C),T_segment=[float(T0),float(T1)],
        B_segment_T=[np.asarray(B0).tolist(),np.asarray(B1).tolist()],
        candidate_sequence=p['candidate_pairs'],executed_role_pairs=pairs,
        offsets=list(offsets),attempts=[],status='RUNNING',
        root_lists_or_caches_shared_between_routes=False,
        reference_cross_route_information='Only candidate resolution choice by exact event-key; no roots, critical fraction, Y, tapes or caches.' if role=='reference' else None)
    if not hasattr(executor,'a2_fallback_activations'):executor.a2_fallback_activations=[]
    executor.a2_fallback_activations.append(activation)
    record['A2_census_repair']=activation;persist()
    for pair in pairs:
        # An existing B2 identity may already have exhausted12/24. Its exact
        # failed attempt is not repeated; the next fixed pair is considered.
        already=all(any(a.get('offset')==offset and
                        all(a.get(side+'_graph',{}).get('seed_levels')==pair for side in ['left','right'])
                        for a in original_anchors) for offset in offsets)
        if already:
            activation['attempts'].append(dict(pair=pair,status='ALREADY_EXHAUSTED_IN_THIS_ROUTE_NOT_REPLAYED'));continue
        for offset in offsets:
            delta=offset/abs(T1-T0) if temperature else offset/max(np.linalg.norm((B1-B0)/5e-5),1e-300)
            fc=record['critical_root']['fraction'];fl,fr=fc-delta,fc+delta
            al=dict(offset=offset,fractions=[fl,fr],seed_pair=pair,role=role,A2_repair=True,status='RUNNING')
            record['anchors'].append(al);activation['attempts'].append(al);persist()
            if not 0<fl<fr<1:
                al.update(status='NUMERICAL_UNRESOLVED',reason='ANCHOR_OUTSIDE_ORIGINAL_SEGMENT');persist();continue
            Tl=T0+fl*(T1-T0);Tr=T0+fr*(T1-T0);Bl=B0+fl*(B1-B0);Br=B0+fr*(B1-B0)
            if not (20<=Tl<=350 and 20<=Tr<=350) or max(np.linalg.norm(Bl),np.linalg.norm(Br))>5e-5*(1+1e-12):
                al.update(status='NUMERICAL_UNRESOLVED',reason='ANCHOR_OUTSIDE_PHYSICAL_RAMP_DOMAIN');persist();continue
            left=ls.enumerate_land(executor.pm(Tl,Bl),tuple(pair));al['left_graph']=left;persist()
            right=ls.enumerate_land(executor.pm(Tr,Br),tuple(pair));al['right_graph']=right;persist()
            continued,consistent=executor.stationary_points(Tl,Bl,old)
            same=consistent and ls.gate.same_census(
                dict(points=continued,counts={k:sum(q['index']==k for q in continued) for k in range(3)}),
                dict(points=left.get('points',[]),counts={k:sum(q['index']==k for q in left.get('points',[])) for k in range(3)}))
            al['continued_old_roots_agree_with_full_left_census']=bool(same)
            al['raw_seed_Newton_failures']={s:[c['newton_failures'] for c in g['censuses']] for s,g in [('left',left),('right',right)]}
            # Exactly the inherited acceptance, as explicitly clarified by owner.
            if left['status']=='COMPUTED_VALID' and right['status']=='COMPUTED_VALID' and same and left['counts']!=right['counts']:
                al['status']='COMPUTED_VALID'
                activation.update(status='COMPUTED_VALID_LATERAL_CENSUS_REPAIR',selected_pair=pair,selected_offset=offset,wall_s=time.monotonic()-started)
                persist();return left,right,fl,fr,offset
            al.update(status='NUMERICAL_UNRESOLVED',reason='LATERAL_CENSUS_CONNECTION_OR_TOPOLOGY_NOT_RESOLVED');persist()
    activation.update(status='FIXED_SEQUENCE_EXHAUSTED',wall_s=time.monotonic()-started)
    record['reason']='FINITE_LATERAL_ANCHORS_EXHAUSTED';persist()
    return None
