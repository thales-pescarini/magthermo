"""Refine stationary-point seed grids on both sides of a critical event.

When the lateral censuses disagree or a stationary point is degenerate,
retry the census at the configured seed resolutions. The critical point,
offsets, continuation and transport law are fixed during this calculation."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
VERSION='FINITE_LATERAL_CENSUS_REFINEMENT_V1'


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
    data=json.loads((ROOT/'config/stationary_point_refinement.json').read_text())
    if data['candidate_pairs']!=[[12,24],[24,48]] or data['control_pairs']!=[[24,48],[48,96]] or data['maximum_seed_level']!=96:
        raise ValueError('UNSUPPORTED_SEED_REFINEMENT_SEQUENCE')
    return data


def recover_anchor_censuses(executor,T0,B0,T1,B1,old,offsets,record,persist):
    if not eligible(record):return None
    role=getattr(executor,'census_refinement_role',None)
    # Refinement requires an explicit candidate or independent reference resolution.
    if role not in ['candidate','reference']:return None
    import grain_energy as core
    import energy_landscape as ls
    p=policy();key=event_key(executor,T0,B0,T1,B1)
    if role=='candidate':pairs=p['candidate_pairs']
    else:
        mapping=getattr(executor,'reference_resolution_targets',{})
        uniform=getattr(executor,'reference_uniform_candidate_pair',None)
        if key in mapping:
            chosen=list(mapping[key]);binding_scope='Exact event-key resolution metadata'
        elif uniform is not None:
            # The independent locator must not inherit candidate critical
            # fractions. When every candidate repair in this exact case used
            # the same pair, that pair defines its uniform finer control.
            chosen=list(uniform);binding_scope='Uniform resolution choice in the exact bound physical case; independently located event'
        else:raise core.Unresolved('REFERENCE_RESOLUTION_TARGET_MISSING')
        if chosen not in p['candidate_pairs']:raise core.Unresolved('REFERENCE_INVALID_CANDIDATE_SEED_LEVEL')
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
        reference_cross_route_information='Only resolution choices in the exact physical case, by exact event-key or its uniform pair; no roots, critical fraction, Y, tapes or caches.' if role=='reference' else None)
    if role=='reference':activation['resolution_metadata_binding_scope']=binding_scope
    if not hasattr(executor,'census_refinements'):executor.census_refinements=[]
    executor.census_refinements.append(activation)
    record['lateral_census_refinement']=activation;persist()
    for pair in pairs:
        # Skip seed pairs already attempted at every lateral offset.
        already=all(any(a.get('offset')==offset and
                        all(a.get(side+'_graph',{}).get('seed_levels')==pair for side in ['left','right'])
                        for a in original_anchors) for offset in offsets)
        if already:
            activation['attempts'].append(dict(pair=pair,status='ALREADY_EXHAUSTED_IN_THIS_ROUTE_NOT_REPLAYED'));continue
        for offset in offsets:
            delta=offset/abs(T1-T0) if temperature else offset/max(np.linalg.norm((B1-B0)/5e-5),1e-300)
            fc=record['critical_root']['fraction'];fl,fr=fc-delta,fc+delta
            al=dict(offset=offset,fractions=[fl,fr],seed_pair=pair,role=role,refined_census=True,status='RUNNING')
            record['anchors'].append(al);activation['attempts'].append(al);persist()
            if not 0<fl<fr<1:
                al.update(status='NUMERICAL_UNRESOLVED',reason='ANCHOR_OUTSIDE_ORIGINAL_SEGMENT');persist();continue
            Tl=T0+fl*(T1-T0);Tr=T0+fr*(T1-T0);Bl=B0+fl*(B1-B0);Br=B0+fr*(B1-B0)
            if not (20<=Tl<=350 and 20<=Tr<=350) or max(np.linalg.norm(Bl),np.linalg.norm(Br))>5e-5*(1+1e-12):
                al.update(status='NUMERICAL_UNRESOLVED',reason='ANCHOR_OUTSIDE_PHYSICAL_RAMP_DOMAIN');persist();continue
            left=ls.enumerate_land(executor.pm(Tl,Bl),tuple(pair));al['left_graph']=left;persist()
            right=ls.enumerate_land(executor.pm(Tr,Br),tuple(pair));al['right_graph']=right;persist()
            continued,consistent=executor.stationary_points(Tl,Bl,old)
            same=consistent and ls.stationary.same_census(
                dict(points=continued,counts={k:sum(q['index']==k for q in continued) for k in range(3)}),
                dict(points=left.get('points',[]),counts={k:sum(q['index']==k for q in left.get('points',[])) for k in range(3)}))
            al['continued_old_roots_agree_with_full_left_census']=bool(same)
            al['raw_seed_Newton_failures']={s:[c['newton_failures'] for c in g['censuses']] for s,g in [('left',left),('right',right)]}
            # Accept only valid censuses on both sides consistent with the continued roots.
            if left['status']=='COMPUTED_VALID' and right['status']=='COMPUTED_VALID' and same and left['counts']!=right['counts']:
                al['status']='COMPUTED_VALID'
                activation.update(status='COMPUTED_VALID_LATERAL_CENSUS_REPAIR',selected_pair=pair,selected_offset=offset,wall_s=time.monotonic()-started)
                # All original selector locals must travel with the selected
                # graphs; the caller's last exhausted offset is a different PM.
                persist();return left,right,fl,fr,offset,Tl,Tr,Bl,Br
            al.update(status='NUMERICAL_UNRESOLVED',reason='LATERAL_CENSUS_CONNECTION_OR_TOPOLOGY_NOT_RESOLVED');persist()
    activation.update(status='FIXED_SEQUENCE_EXHAUSTED',wall_s=time.monotonic()-started)
    record['reason']='FINITE_LATERAL_ANCHORS_EXHAUSTED';persist()
    return None
