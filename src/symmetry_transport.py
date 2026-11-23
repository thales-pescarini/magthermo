"""Split occupation between symmetry-related exits of a zero-field [110] grain.

The local transport rule requires a verified stationary parent, two
symmetry-related receivers and agreement of the one-sided descent paths."""
from fractions import Fraction
import copy
import hashlib
import json
import math
import numpy as np

BASELINE = 'verified_continuation_v1'
VERSION = 'zero_field_c110_symmetric_exit_v1'
G=np.array([[0.,-1.,0.],[-1.,0.,0.],[0.,0.,1.]])


# Accept the transport identifiers stored in earlier numerical records.
def transport_mode(version):
    return {'E1-TQ01-2026-10-25-UNKNOWN': BASELINE, 'E1-TQ01-R1': VERSION}.get(version, version)


def counts(graph):
    value=graph.get('counts')
    return [int(value.get(str(k),value.get(k,-1))) for k in range(3)] if isinstance(value,dict) else list(value or [])


def complete_graph(graph,anchor=False):
    c=counts(graph);points=graph.get('points',[])
    if (graph.get('status')!='COMPUTED_VALID' or len(c)!=3 or c[0]-c[1]+c[2]!=2
        or len(points)!=sum(c) or len(graph.get('minima',[]))!=c[0]
        or not all(p.get('morse') and p.get('index') in [0,1,2] for p in points)
        or not graph.get('connections') or not all(x.get('valid') for x in graph['connections'])):return False
    if anchor and (graph.get('census_agreement') is not True or len(graph.get('censuses',[]))<2):return False
    return True


def exact_energy_symmetry(identity,input_axis):
    c=np.asarray(input_axis,float)
    if (c.shape!=(3,) or not np.isfinite(c).all() or c[0]!=c[1] or c[2]!=0 or c[0]==0
        or abs(np.linalg.norm(c)-1)>1e-12):return False
    if identity.get('TM')!=0 or type(identity.get('TM')) is not int:return False
    if not np.array_equal(np.asarray(identity.get('B_crystal_T')),np.zeros(3)):return False
    effective=np.asarray(identity.get('c'))
    if effective.shape!=(3,) or effective[0]!=effective[1] or effective[2]!=0:return False
    N=np.asarray(identity.get('N'))
    if N.shape!=(3,3) or not np.array_equal(G.T@N@G,N):return False
    # Both cubic monomials are invariant under this signed permutation exactly;
    # the demagnetizing quadratic form is invariant by exact coefficient equality.
    return all(math.isfinite(float(identity[k])) for k in ['K1','K2','Ms'])


def fixed_root_certificate(point):
    direction=point.get('direction_mp')
    if not direction or len(direction)!=3 or point.get('geometry_precision',0)<85:return False
    if Fraction(direction[0])!=-Fraction(direction[1]):return False
    u=np.asarray(point['m']);mp_float=np.array([float(x) for x in direction])
    if np.linalg.norm(u-mp_float)>5e-15:return False
    return (point.get('morse') is True and point.get('gradient_residual',math.inf)<1e-40
            and point.get('normal_residual',math.inf)<1e-40
            and point.get('angular_correction_eta',math.inf)<=1e-9)


def eligible_column(event,parent_index,input_axis,version=VERSION):
    """Return one verified new column, or None without changing old treatment."""
    if transport_mode(version)!=VERSION:return None
    fields=event.get('B_segment_T',[])
    if len(fields)!=2 or not all(np.array_equal(np.asarray(b),np.zeros(3)) for b in fields):return None
    left=event.get('pre_graph',{});right=event.get('post_graph',{})
    if not (exact_energy_symmetry(left.get('identity',{}),input_axis)
            and exact_energy_symmetry(right.get('identity',{}),input_axis)):return None
    if not complete_graph(left) or not complete_graph(right):return None
    anchors=[a for a in event.get('anchors',[]) if a.get('status')=='COMPUTED_VALID']
    if not anchors:return None
    anchor=anchors[-1]
    if (anchor.get('continued_old_roots_agree_with_full_left_census') is not True
        or not complete_graph(anchor.get('left_graph',{}),True)
        or not complete_graph(anchor.get('right_graph',{}),True)):return None
    lateral=event.get('lateral_continuation',{})
    for side,graph in [('left',left),('right',right)]:
        steps=lateral.get(side,[])
        if not steps or not all(s.get('valid') is True for s in steps) or steps[-1].get('counts')!=counts(graph):return None
    if not 0<=parent_index<len(left['minima']):return None
    parent=left['minima'][parent_index]
    if parent.get('index')!=0 or min(parent.get('eig',[-1]))<=0 or not fixed_root_certificate(parent):return None
    targets=right['minima']
    if len(targets)!=2 or counts(right)[0]!=2:return None
    U=np.array([t['m'] for t in targets]);parent_u=np.asarray(parent['m'])
    if np.linalg.norm(U[0]-U[1])<=1e-6:return None
    if not all(t.get('index')==0 and t.get('morse') and min(t['eig'])>0 for t in targets):return None
    permutation=[]
    for u in U:
        distances=np.linalg.norm(U-G@u,axis=1);j=int(np.argmin(distances))
        if distances[j]>1e-9:return None
        permutation.append(j)
    if permutation!=[1,0]:return None
    post=[p for p in right['points'] if p.get('index')!=0 and p.get('morse') and fixed_root_certificate(p)
          and np.linalg.norm(np.asarray(p['m'])-parent_u)<1e-6]
    if len(post)!=1:return None
    endpoint=post[0]
    if not (min(parent['eig'])>0 and min(endpoint['eig'])<0 and max(endpoint['eig'])>0):return None
    column=np.array([.5,.5])
    return dict(version=VERSION,parent_index=parent_index,parent_point_id=parent.get('point_id'),
                receivers=[0,1],receiver_point_ids=[p.get('point_id') for p in targets],column=column,
                exact_class='c110 with G(x,y,z)=(-y,-x,z)',exact_coefficient_symmetry=True,
                exact_parent_fixed_by_G_in_MP_representation=True,target_permutation=permutation,
                signed_parent_curvatures=[parent['eig'],endpoint['eig']],
                complete_geometry_from_two_complete_anchors_and_verified_lateral_continuation=True,
                physical_change=True,global_equilibrium_imposed=False,barrier_zero_not_claimed=True,
                occupancy_inferred_from_geometry=False,scope='local symmetry-related exit rule for zero field')


def complete_fixture_map(event,input_axis,version=VERSION):
    """Preserve verified survival columns and calculate eligible symmetry splits."""
    left=event['pre_graph'];right=event['post_graph'];U=np.array([p['m'] for p in left['minima']]);V=np.array([p['m'] for p in right['minima']])
    P=np.zeros((len(V),len(U)));known=set();records=[]
    # Use verified lateral continuation and flow tests to identify surviving states.
    checks={c['old']:c for c in event.get('minimum_continuation_checks',[])}
    for i,u in enumerate(U):
        check=checks.get(i,{})
        if check.get('Newton_converged') and check.get('index')==0 and check.get('nearest_distance',math.inf)<1e-6:
            v=np.asarray(check.get('corrected_direction',check.get('candidate_direction')))
            distances=np.linalg.norm(V-v,axis=1);j=int(np.argmin(distances))
            if distances[j]<1e-6 and np.linalg.norm(v-u)<.15:P[j,i]=1.;known.add(i)
    inherited=P.copy()
    for i in range(len(U)):
        if i in known:continue
        record=eligible_column(event,i,input_axis,version)
        if record is not None:P[:,i]=record['column'];known.add(i);records.append(record)
    unknown=sorted(set(range(len(U)))-known)
    if not unknown and (np.min(P)<0 or not np.array_equal(P.sum(axis=0),np.ones(len(U)))):raise ValueError('SYMMETRIC_EXIT_MAP_INVARIANTS')
    return dict(matrix=P,known_columns=sorted(known),unknown_columns=unknown,complete=not unknown,
                new_columns=records,inherited_valid_columns=inherited,version=version,
                old_unknown_not_reclassified=True)


def transferred_moment_record(grain_id,native_weight,before,map_record,model,history,event_id,state_proof=None):
    if not map_record['complete']:raise ValueError('GEOMETRY_CACHE_PARTIAL_MAP_CANNOT_PROPAGATE_STATE')
    if not state_proof or state_proof.get('kind') not in ['fresh_F_N1_sequential','manufactured_unit_only']:
        raise ValueError('GEOMETRY_CACHE_PHYSICAL_STATE_PROVENANCE_REQUIRED_NOT_PLACEHOLDER')
    before=np.asarray(before,float);P=map_record['matrix']
    if before.shape!=(P.shape[1],) or not np.isfinite(before).all() or np.min(before)<-2e-12 or abs(before.sum()-1)>2e-11:
        raise ValueError('GEOMETRY_CACHE_PHYSICAL_STATE_REQUIRED')
    parents=[r['parent_index'] for r in map_record['new_columns']]
    after=P@before
    return dict(grain_id=grain_id,history=history,event_id=event_id,model=model,transport_version=map_record['version'],
                support_capacity=native_weight if parents else 0.,
                transferred_moment_fraction=float(native_weight*before[parents].sum()),
                occupation_actually_transferred=float(before[parents].sum()),
                physical_probability_before=before.tolist(),physical_probability_after=after.tolist(),
                geometry_placeholder_used=False,eligible_parent_indices=parents,
                state_evidence=state_proof,scientific_measurement=state_proof['kind']=='fresh_F_N1_sequential',
                support_grain_union_key=grain_id,global_population_effect=None,
                full_history_Y_delta=None,old_full_history_reference='UNKNOWN; no numerical zero delta invented')
