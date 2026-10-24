"""R1 E development engine using the unchanged all-state physical reference.

History anchors are explicit. F/N1 share geometry and spectra but never states.
Geometrical tapes contain no magnetic propagation; duration changes only the
event-hold propagator. Failed tapes are retained and cannot count as complete.
"""
import copy
import hashlib
import json
import math
import time
import numpy as np
import e1_core as core
import thermal_executor as thermal
import r1_landscape as landscape
import g1_geometry as geometry
from scipy.optimize import linear_sum_assignment
from factorized_executor import GeometryTape, SharedCaches
from wp1c_spectral import spectral_F, observer


def spectral_graph_key(graph):
    """Content identity survives serialization; field and ordered roots matter."""
    identity = dict(physical=graph['identity'],
                    minima=[dict(id=p.get('point_id'), m=p['m'], E=p.get('E_mp', p['E'])) for p in graph['minima']],
                    channels=[dict(id=c.get('saddle_id'), endpoints=c['endpoints'], E=c.get('saddle_E_mp', c['saddle_E'])) for c in graph['channels']],
                    canonical=graph.get('canonical_geometry_and_energy', False))
    return hashlib.sha256(json.dumps(core.plain(identity), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class GeneralTape(GeometryTape):
    def __init__(self, *args, event_T_C, reference_only=False, use_prefix_cache=True,
                 use_laboratory_cache=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_T_C = float(event_T_C)
        self.reference_only=reference_only
        self.use_prefix_cache=bool(use_prefix_cache)
        self.use_laboratory_cache=bool(use_laboratory_cache)
        self.prefix_geometry_reuse=dict(reused=False,occupations_reused=False)
        self.laboratory_geometry_reuse=dict(reused=False,occupations_reused=False)
        self.minimum_branch_proofs=[]
        self.reference_partition_failures=[]
        self._reference_attempt_active=False
        self._minimum_retry_active=False

    def prefix_geometry_key(self, preparation_graph):
        # A complete event geometry chronology is common to P12 and P24.
        # R and its exact preparation metadata remain in this key even at B=0.
        value=dict(kind='E_GEOMETRY_PREFIX_V1',s=self.s,c=self.c,R=self.R,R_metadata=self.Rspec,
                   TM=0,event_T_C=self.event_T_C,protocols=self.protocols,resolution=self.level,
                   reference_only=self.reference_only,
                   partition_policy='reference_first_minimum_recovery_after_failure',
                   preparation_graph=spectral_graph_key(preparation_graph))
        return hashlib.sha256(json.dumps(core.plain(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def restore_geometry_prefix(self, cached, key):
        restored=copy.deepcopy(cached)
        if not np.array_equal(self.initial_p,restored['initial_p']):
            raise core.Unresolved('PREFIX_PREPARATION_DIFFERENCE')
        self.actions.extend(restored['actions'])
        self.events.extend(restored['events'])
        self.trace.extend(restored['trace'])
        self.critical_catalog.extend(restored['critical_catalog'])
        self.event_solutions.update(restored['event_solutions'])
        self.minimum_branch_proofs.extend(restored['minimum_branch_proofs'])
        self.reference_partition_failures.extend(restored.get('reference_partition_failures',[]))
        for check in restored['field_connection_checks']:
            check['verification_reused_for_identical_geometry_prefix']=True
            self.field_connection_checks.append(check)
        self.maximum_alignment_distance=max(self.maximum_alignment_distance,restored['maximum_alignment_distance'])
        # These probabilities only check the dimensions of geometry maps. No
        # magnetic occupation propagated for a previous d/history is loaded.
        p=self.initial_p.copy()
        for action in restored['actions']:
            if action['kind']=='map':
                p=action['matrix']@p
                thermal.checked(p,'replayed_geometry_map_placeholder')
        self.steps+=sum(a['kind']=='operator' for a in restored['actions'])
        self.readings=[dict(label='NRM',geometry_only_placeholder=True)]
        land=restored['final_land']
        self.cache[self.key(self.protocols['preparation']['temperature_C'],np.zeros(3))]=land
        self.prefix_geometry_reuse=dict(reused=True,key=key,source_protocol=restored['source_protocol'],
             physical_actions=len(restored['actions']),source_prefix_wall_s=restored['prefix_wall_s'],
             avoided_alignment_calls=restored['alignment_calls'],occupations_reused=False,
             laboratory_sequence_reused=False,all_F_modes_retained=True)
        return p,land

    def store_geometry_prefix(self, key, land, started, protocol):
        bank=getattr(self.shared,'E_geometry_prefixes',None)
        if bank is None:
            bank={};self.shared.E_geometry_prefixes=bank
        bank[key]=copy.deepcopy(dict(actions=self.actions,events=self.events,trace=self.trace,
            initial_p=self.initial_p,final_land=land,critical_catalog=self.critical_catalog,
            event_solutions=self.event_solutions,minimum_branch_proofs=self.minimum_branch_proofs,
            reference_partition_failures=self.reference_partition_failures,
            field_connection_checks=self.field_connection_checks,maximum_alignment_distance=self.maximum_alignment_distance,
            alignment_calls=self.alignment_calls,prefix_wall_s=time.monotonic()-started,source_protocol=protocol))
        while len(bank)>8:bank.pop(next(iter(bank)))
        self.prefix_geometry_reuse=dict(reused=False,key=key,occupations_reused=False,
                                       laboratory_sequence_reused=False,all_F_modes_retained=True)

    @staticmethod
    def zero_field_segment(key):
        return np.array_equal(np.asarray(key[1]),np.zeros(3)) and np.array_equal(np.asarray(key[3]),np.zeros(3))

    def laboratory_geometry_key(self, land, p, protocol):
        # Event temperature is deliberately absent: laboratory geometry begins
        # at this exact ordered graph, with its own prescribed zero-field path.
        # Keep the geometric placeholder and all relevant critical context too.
        # It is never a propagated magnetic occupation or an input response.
        zero_catalog=[h for h in self.critical_catalog
                      if np.array_equal(np.asarray(h['B_sample_T']),np.zeros(3))]
        zero_solutions=[(key,value) for key,value in sorted(self.event_solutions.items())
                        if self.zero_field_segment(key)]
        value=dict(kind='E_LABORATORY_GEOMETRY_V1',s=self.s,c=self.c,R=self.R,R_metadata=self.Rspec,
                   TM=0,protocol=protocol,protocols=self.protocols,resolution=self.level,
                   reference_only=self.reference_only,
                   partition_policy='reference_first_minimum_recovery_after_failure',
                   entry_graph=land,geometric_placeholder=p,
                   zero_critical_catalog=zero_catalog,zero_event_solutions=zero_solutions)
        return hashlib.sha256(json.dumps(core.plain(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def laboratory_token(self):
        fields=['actions','events','trace','readings','minimum_branch_proofs',
                'reference_partition_failures','field_connection_checks']
        return dict(lengths={name:len(getattr(self,name)) for name in fields},
                    alignment_calls=self.alignment_calls,started=time.monotonic())

    def store_laboratory_geometry(self, key, token, entry_p, land):
        payload={name:getattr(self,name)[offset:] for name,offset in token['lengths'].items()}
        payload.update(entry_placeholder=entry_p,final_land=land,
            zero_critical_catalog=[h for h in self.critical_catalog
                                   if np.array_equal(np.asarray(h['B_sample_T']),np.zeros(3))],
            zero_event_solutions={k:v for k,v in self.event_solutions.items() if self.zero_field_segment(k)},
            maximum_alignment_distance=self.maximum_alignment_distance,
            alignment_calls=self.alignment_calls-token['alignment_calls'],
            source_event_T_C=self.event_T_C,laboratory_wall_s=time.monotonic()-token['started'])
        payload=copy.deepcopy(payload)  # Preserve graph/event aliases within the slice.
        for event in payload['events']:
            for name in ['occupation_before_map','occupation_after_map','probability_check']:event.pop(name,None)
        payload['trace']=[{k:v for k,v in row.items() if k not in ['state','forked_probability']}
                          for row in payload['trace']]
        bank=getattr(self.shared,'E_laboratory_geometries',None)
        if bank is None:
            bank={};self.shared.E_laboratory_geometries=bank
        bank[key]=payload
        while len(bank)>8:bank.pop(next(iter(bank)))
        self.laboratory_geometry_reuse=dict(reused=False,key=key,occupations_reused=False,
                                            all_F_modes_retained=True)

    def restore_laboratory_geometry(self, cached, key, p):
        restored=copy.deepcopy(cached)
        if not np.array_equal(p,restored['entry_placeholder']):
            raise core.Unresolved('LABORATORY_GEOMETRIC_PLACEHOLDER_DIFFERENCE')
        fresh_p=np.array(p,copy=True)
        for action in restored['actions']:
            if action['kind']=='map':
                fresh_p=action['matrix']@fresh_p
                thermal.checked(fresh_p,'replayed_laboratory_geometry_map_placeholder')
        for name in ['actions','events','trace','readings','minimum_branch_proofs',
                     'reference_partition_failures','field_connection_checks']:
            getattr(self,name).extend(restored[name])
        self.steps+=sum(a['kind']=='operator' for a in restored['actions'])
        self.critical_catalog=[h for h in self.critical_catalog
                               if not np.array_equal(np.asarray(h['B_sample_T']),np.zeros(3))]
        self.critical_catalog.extend(restored['zero_critical_catalog'])
        self.event_solutions.update(restored['zero_event_solutions'])
        self.maximum_alignment_distance=max(self.maximum_alignment_distance,restored['maximum_alignment_distance'])
        self.laboratory_geometry_reuse=dict(reused=True,key=key,
            source_event_T_C=restored['source_event_T_C'],physical_actions=len(restored['actions']),
            source_laboratory_wall_s=restored['laboratory_wall_s'],
            avoided_alignment_calls=restored['alignment_calls'],occupations_reused=False,
            entry_placeholder_scope='dimension and stochastic-map checks only; no magnetic state reused',
            all_F_modes_retained=True)
        return fresh_p,restored['final_land']

    def track_minima(self,T,B,old):
        """Continue occupational states, independently of unrelated maxima.

        This is stationary-equation continuation with positive tangent
        curvature at prescribed parameter subdivisions. It is not matching
        by energy or nearest directions alone. Any min loss, duplication,
        index change or unresolved solve returns to the full event handler.
        """
        T0=old['identity']['T_C']
        B0=self.R@np.asarray(old['identity']['B_crystal_T'])
        B=np.asarray(B,float)
        steps=max(1,math.ceil(abs(T-T0)/.25),math.ceil(np.linalg.norm(B-B0)/(5e-5/32)))
        if steps>128:raise core.Unresolved('E_MINIMUM_TRACKING_SUBDIVISION_LIMIT')
        points=[dict(p) for p in old['minima']]
        initial=np.array([p['m'] for p in points]);minimum_curvature=math.inf;max_move=0.
        for step in range(1,steps+1):
            f=step/steps;pm=self.pm(T0+f*(T-T0),B0+f*(B-B0))
            U=np.array([p['m'] for p in points]);u,ok=landscape.batch_newton(pm,U)
            changed=[]
            for i,p in enumerate(points):
                q=None
                if ok[i]:
                    try:q=landscape.point(pm,u[i])
                    except core.Unresolved:pass
                if q is None:
                    sep=min((np.linalg.norm(U[i]-v) for j,v in enumerate(U) if j!=i),default=2.)
                    for dps in [85,120]:
                        try:q=geometry.refine_kkt(pm,p['m'],dps,maximum_move=min(.1,.25*sep));break
                        except (ValueError,ZeroDivisionError,core.Unresolved):pass
                if q is None or q['index']!=0 or not q['morse']:
                    raise core.Unresolved('E_MINIMUM_BRANCH_NOT_PRESERVED_REQUIRES_EVENT')
                move=float(np.linalg.norm(q['m']-U[i]));max_move=max(max_move,move)
                if move>=.1:raise core.Unresolved('E_MINIMUM_BRANCH_TRAVEL_REQUIRES_EVENT')
                minimum_curvature=min(minimum_curvature,min(q['eig']))
                changed.append(dict(q,point_id=p['point_id']))
            final=np.array([q['m'] for q in changed])
            if any(np.linalg.norm(final[i]-final[j])<=1e-6 for i in range(len(final)) for j in range(i)):
                raise core.Unresolved('E_MINIMUM_BRANCH_MERGE_REQUIRES_EVENT')
            points=changed
        proof=dict(method='stationary-equation continuation of every minimum, positive tangent Hessian and distinct branches at parameter subdivisions',
                   T_start_C=T0,T_end_C=float(T),B_start_sample_T=B0,B_end_sample_T=B,
                   subdivisions=steps,minimum_positive_curvature=float(minimum_curvature),
                   maximum_single_step_direction_change=max_move,initial_minima=initial,
                   final_minima=np.array([p['m'] for p in points]),
                   occupational_states_removed=0,physical_transport_law_changed=False,
                   qualification='finite branch verification; not uniform bifurcation exclusion')
        return points,proof

    def landscape(self,T,B,previous=None,force=False):
        try:
            return super().landscape(T,B,previous,force)
        except core.Unresolved as original:
            if self.reference_only or self._reference_attempt_active or previous is None or force or str(original)!='STATIONARY_EVENT_CANDIDATE':
                raise
            start=time.monotonic()
            try:
                minima,proof=self.track_minima(T,B,previous)
                pm=self.pm(T,B)
                fresh=landscape.enumerate_land(pm,self.census_levels(T,B))
                if fresh['status']!='COMPUTED_VALID' or len(fresh['minima'])!=len(minima):
                    raise core.Unresolved('E_INDEPENDENT_CENSUS_NOT_SAME_OCCUPATIONAL_STATES')
                U=np.array([p['m'] for p in minima]);V=np.array([p['m'] for p in fresh['minima']])
                a,b=linear_sum_assignment(np.linalg.norm(U[:,None]-V[None,:],axis=2))
                if max(np.linalg.norm(U[a]-V[b],axis=1))>1e-6:
                    raise core.Unresolved('E_TRACKED_MINIMUM_NOT_IN_INDEPENDENT_CENSUS')
                proof['new_census_counts']=fresh['counts'];proof['previous_census_counts']=previous.get('counts')
                fresh['E_minimum_branch_proof']=proof
                self.minimum_branch_proofs.append(proof)
                self.cache[self.key(T,B)]=fresh
                self.cost['landscapes']+=time.monotonic()-start
                return fresh
            except (core.Unresolved,ValueError,ZeroDivisionError):
                raise original

    def align(self,source,target,occupation=None):
        try:
            return super().align(source,target,occupation)
        except core.Unresolved as original:
            if self.reference_only or self._reference_attempt_active or str(original)!='IDENTITY_CONTINUATION_REQUIRES_EVENT':raise
            T=target['identity']['T_C'];B=self.R@np.asarray(target['identity']['B_crystal_T'])
            try:
                points,proof=self.track_minima(T,B,source)
                U=np.array([p['m'] for p in points]);V=np.array([p['m'] for p in target['minima']])
                if len(U)!=len(V):raise core.Unresolved('E_STATE_COUNT_CHANGED_REQUIRES_EVENT')
                D=np.linalg.norm(U[:,None]-V[None,:],axis=2);a,b=linear_sum_assignment(D)
                if max(D[a,b])>1e-6:raise core.Unresolved('E_BRANCH_ASSOCIATION_UNRESOLVED')
                P=np.zeros((len(V),len(U)));P[b,a]=1.
                self.minimum_branch_proofs.append(proof)
                return P
            except (core.Unresolved,ValueError,ZeroDivisionError):raise original

    def verify_minimum_recovery(self,old,T,B):
        """Require continued positive minima and an independent complete end census."""
        points,proof=self.track_minima(T,B,old)
        fresh=landscape.enumerate_land(self.pm(T,B),self.census_levels(T,B))
        if fresh['status']!='COMPUTED_VALID' or len(fresh['minima'])!=len(points):
            raise core.Unresolved('E_RECOVERY_CHANGES_OCCUPATIONAL_STATE_COUNT')
        U=np.array([p['m'] for p in points]);V=np.array([p['m'] for p in fresh['minima']])
        D=np.linalg.norm(U[:,None]-V[None,:],axis=2);a,b=linear_sum_assignment(D)
        if len(a)!=len(points) or max(D[a,b],default=0.)>1e-6:
            raise core.Unresolved('E_RECOVERY_MINIMUM_NOT_IN_INDEPENDENT_CENSUS')
        proof.update(independent_end_census_counts=fresh['counts'],
                     scope='finite verification of each original minimum and complete end census; not uniform exclusion of intermediate bifurcations')
        return proof

    @staticmethod
    def nonminimum_fold_hint(hint):
        critical=hint.get('critical',{})
        if critical.get('kind')!='fold_stationary_KKT':return False
        values=np.asarray(critical.get('eigenvalues',[]),float)
        if values.shape!=(2,) or not np.all(np.isfinite(values)):return False
        other=values[np.argmax(abs(values))]
        return other<0 and min(abs(values))<=max(1e-4,1e-6*abs(other))

    def advance(self,p,old,T0,B0,T1,B1,seconds,label,depth=0):
        """Use complete reference partitions before attempting minimum recovery.

        A failed full event producer is kept as evidence, then its uncommitted
        actions are rolled back. Recovery solves the existing equations and
        never relaxes the minimum/state-map guards or changes a probability.
        """
        if self.reference_only or self._reference_attempt_active or self._minimum_retry_active:
            return super().advance(p,old,T0,B0,T1,B1,seconds,label,depth)
        token=self.transaction_token();events_at=len(self.events);trace_at=len(self.trace)
        steps_at=self.steps;catalog_at=list(self.critical_catalog);input_p=np.array(p,copy=True)
        original_error=None
        self._reference_attempt_active=True
        try:
            return super().advance(p,old,T0,B0,T1,B1,seconds,label,depth)
        except core.Unresolved as exc:
            if str(exc) not in {
                'LATERAL_ROOT_CONTINUATION_UNRESOLVED','FINITE_LATERAL_ANCHORS_EXHAUSTED',
                'B2_LATERAL_24_STEP_CAP','CRITICAL_ROOT_NOT_RESOLVED',
                'EVENT_HAS_NO_DEMONSTRATED_SIGNED_CURVATURE_BRACKET',
            }:raise
            original_error=exc
        finally:
            self._reference_attempt_active=False
        failure=dict(reason=str(original_error),T_segment=[float(T0),float(T1)],
                     B_segment_T=[np.asarray(B0),np.asarray(B1)],label=label,
                     failed_events=copy.deepcopy(self.events[events_at:]),
                     diagnostic=copy.deepcopy(getattr(self,'diagnostic',None)),
                     recovery_accepted=False)
        self.reference_partition_failures.append(failure)
        try:
            proof=self.verify_minimum_recovery(old,T1,B1)
        except (core.Unresolved,ValueError,ZeroDivisionError) as guard:
            failure['recovery_guard_failure']=str(guard)
            raise original_error
        self.rollback_transaction(token)
        del self.events[events_at:];del self.trace[trace_at:];self.steps=steps_at
        # Only an already identified maximum/saddle fold may be bypassed in
        # this bounded retry. A minimum/saddle hint remains compulsory.
        B0=np.asarray(B0);B1=np.asarray(B1)
        removed=[];retained=[]
        for hint in catalog_at:
            inside=(T0!=T1 and min(T0,T1)<hint['T_C']<max(T0,T1)
                    and np.array_equal(B0,B1)
                    and np.linalg.norm(np.asarray(hint['B_sample_T'])-B0)<1e-15)
            (removed if inside and self.nonminimum_fold_hint(hint) else retained).append(hint)
        self.critical_catalog=retained;self.pending_critical_hint=None
        self.minimum_branch_proofs.append(proof)
        self._minimum_retry_active=True
        try:
            result=super().advance(input_p,old,T0,B0,T1,B1,seconds,label,depth)
            failure.update(recovery_accepted=True,minimum_branch_proof=proof,
                           omitted_nonminimum_hint_count=len(removed),
                           occupations_reused=False,physical_transport_law_changed=False)
            return result
        except Exception as exc:
            failure['recovery_retry_failure']=str(exc)
            raise
        finally:
            self._minimum_retry_active=False
            for hint in removed:
                if not any(hint is present for present in self.critical_catalog):self.critical_catalog.append(hint)

    def failure_evidence(self):
        return dict(actions=self.actions,events=self.events,initial_p=self.initial_p,
                    geometry_cost_s=self.cost,diagnostic=getattr(self,'diagnostic',None),
                    minimum_branch_proofs=self.minimum_branch_proofs,
                    reference_partition_failures=self.reference_partition_failures,
                    prefix_geometry_reuse=self.prefix_geometry_reuse,
                    laboratory_geometry_reuse=self.laboratory_geometry_reuse,
                    scope='geometry producer failure; any recorded probabilities are placeholders, never reused magnetic occupations')

    def compile(self,protocol='P12'):
        result=super().compile(protocol)
        result['minimum_branch_proofs']=self.minimum_branch_proofs
        result['reference_partition_failures']=self.reference_partition_failures
        result['partition_policy']='reference_first_minimum_recovery_after_failure'
        result['reference_only']=self.reference_only
        result['prefix_geometry_reuse']=self.prefix_geometry_reuse
        result['laboratory_geometry_reuse']=self.laboratory_geometry_reuse
        return result

    def run(self, protocol, prefix=None, on_prefix=None):
        if prefix is not None:
            raise ValueError('EXTERNAL_PREFIX_REQUIRES_HISTORY_IDENTITY')
        T0 = float(self.protocols['preparation']['temperature_C'])
        zero = np.zeros(3)
        field = np.asarray(self.protocols['event']['field_T'])
        event = self.protocols['event']
        lab = self.protocols['laboratory']
        T = self.event_T_C
        land = self.landscape(T0, zero)
        p = core.prepare(self.pm(T0, zero), land, self.R, self.Rspec)
        thermal.checked(p, 'E_preparation')
        self.initial_p = p.copy()
        self.readings = []
        started=time.monotonic();prefix_key=self.prefix_geometry_key(land)
        cached=getattr(self.shared,'E_geometry_prefixes',{}).get(prefix_key) if self.use_prefix_cache else None
        if cached is not None:
            p,land=self.restore_geometry_prefix(cached,prefix_key)
        else:
            p, land = self.ramp(p, land, T0, zero, T0, field, event['field_on_ramp_seconds'], 'field_on')
            p, land = self.ramp(p, land, T0, field, T, field, (T-T0) / event['heating_rate_C_per_second'], 'event_heating')
            # Placeholder duration is replaced only at the explicit hold action.
            p = self.hold(p, land, T, 1., 'event_hold')
            p, land = self.ramp(p, land, T, field, T0, field, (T-T0) / event['cooling_rate_C_per_second'], 'event_cooling')
            p, land = self.ramp(p, land, T0, field, T0, zero, event['field_off_ramp_seconds'], 'field_off')
            p = self.hold(p, land, T0, event['final_zero_field_wait_seconds'], 'event_wait')
            self.readings.append(self.read(p, land, 'NRM'))
            if self.use_prefix_cache:self.store_geometry_prefix(prefix_key,land,started,protocol)
        lab_key=self.laboratory_geometry_key(land,p,protocol)
        use_lab=self.use_laboratory_cache and not self.reference_only
        cached=getattr(self.shared,'E_laboratory_geometries',{}).get(lab_key) if use_lab else None
        if cached is not None:
            p,land=self.restore_laboratory_geometry(cached,lab_key,p)
            return self.result('COMPUTED_VALID')
        token=self.laboratory_token();entry_p=p.copy()
        for plateau in lab[protocol]['plateaus_C']:
            temperature = float(plateau)
            p, land = self.ramp(p, land, T0, zero, temperature, zero, (temperature-T0) / lab['heating_rate_C_per_second'], 'lab_heat_'+str(plateau))
            p = self.hold(p, land, temperature, lab['hold_seconds'], 'lab_hold_'+str(plateau))
            p, land = self.ramp(p, land, temperature, zero, T0, zero, (temperature-T0) / lab['cooling_rate_C_per_second'], 'lab_cool_'+str(plateau))
            p = self.hold(p, land, T0, lab['wait_before_measurement_seconds'], 'lab_wait_'+str(plateau))
            self.readings.append(self.read(p, land, str(plateau)))
        if use_lab:self.store_laboratory_geometry(lab_key,token,entry_p,land)
        return self.result('COMPUTED_VALID')


def prepare_response(tape, d_nm, R, progress=None):
    """One pass for H(d) and all sequential A_j(d); no inverse state map.

    A_j includes laboratory dependence on the state after the event, even
    through rectangular topology maps. All F modes are retained. This is an
    exact algebraic factorization of the specified discrete chronology.
    """
    started = time.monotonic()
    prefix = {m: np.array(tape['initial_p'], float, copy=True) for m in ['F', 'N1']}
    readouts = {'F': [], 'N1': []}
    current = {}
    hold = None
    spectral_seconds = 0.
    selection_seconds = 0.
    prop_seconds = 0.
    algebra_seconds = 0.
    cache = {}
    n_modes = set()
    max_mass = 0.
    min_probability = 1.
    for i, action in enumerate(tape['actions']):
        kind = action['kind']
        if kind == 'operator':
            graph = action['graph']
            key = (spectral_graph_key(graph), action['T_C'])
            if key not in cache:
                t = time.monotonic()
                sp = spectral_F(graph, d_nm, action['T_C'])
                spectral_seconds += time.monotonic() - t
                t = time.monotonic()
                selected = sp.select(observer(graph, R))
                selection_seconds += time.monotonic() - t
                cache[key] = sp, selected
                # A bounded cache avoids an MP eigensystem per repeated action.
                if len(cache) > 256:
                    cache.pop(next(iter(cache)))
            sp, selected = cache[key]
            n_modes.add(sp.n)
            if action['label'] == 'event_hold':
                if hold is not None:
                    raise RuntimeError('MULTIPLE_EVENT_HOLDS')
                hold = sp, selected
                current = {m: np.eye(sp.n) for m in prefix}
                continue
            matrices = {}
            for model in prefix:
                t = time.monotonic()
                matrices[model], info = sp.propagator(action['seconds'], model, selected)
                prop_seconds += time.monotonic() - t
                max_mass = max(max_mass, info['mass_residual'])
                min_probability = min(min_probability, info['minimum'])
        elif kind == 'map':
            matrices = {m: np.asarray(action['matrix'], float) for m in prefix}
        elif kind == 'read':
            if hold is None:
                raise RuntimeError('READING_BEFORE_EVENT_HOLD')
            W = observer(action['graph'], R) * (1e-18 / thermal.S)
            for model in prefix:
                readouts[model].append(W @ current[model])
            continue
        else:
            raise RuntimeError('UNKNOWN_TAPE_ACTION')
        t = time.monotonic()
        for model, U in matrices.items():
            if hold is None:
                prefix[model] = U @ prefix[model]
                thermal.checked(prefix[model], 'E_prefix_'+model)
            else:
                current[model] = U @ current[model]
                if not np.all(np.isfinite(current[model])):
                    raise core.Unresolved('NONFINITE_COMPOSED_MAP')
                error = float(np.max(abs(current[model].sum(axis=0)-1)))
                if error > 2e-11 or np.min(current[model]) < -2e-11:
                    raise core.Unresolved('COMPOSED_MAP_INVARIANTS_UNRESOLVED')
        algebra_seconds += time.monotonic() - t
        if progress and i % 128 == 0:
            progress(dict(phase='spectra_response', action=i, total_actions=len(tape['actions']), d_nm=d_nm))
    if hold is None:
        raise core.Unresolved('PARTIAL_TAPE_DID_NOT_REACH_EVENT_HOLD')
    return dict(prefix=prefix, readouts=readouts, hold=hold,
                profile=dict(wall_s=time.monotonic()-started, spectral_exclusive_s=spectral_seconds,
                             selection_exclusive_s=selection_seconds, propagator_exclusive_s=prop_seconds,
                             algebra_exclusive_s=algebra_seconds, modes=sorted(n_modes),
                             max_MP_mass_residual=max_mass, minimum_MP_probability=min_probability))


def evaluate_duration(prepared, ell):
    start = time.monotonic()
    sp, selected = prepared['hold']
    results = {}
    for model in ['F', 'N1']:
        U, info = sp.propagator(10.**ell, model, selected)
        p = U @ prepared['prefix'][model]
        thermal.checked(p, 'E_duration_'+model)
        results[model] = np.array([A @ p for A in prepared['readouts'][model]])
        if not np.all(np.isfinite(results[model])):
            raise core.Unresolved('NONFINITE_READOUT')
    return results, dict(marginal_duration_s=time.monotonic()-start, ell=ell,
                         F_all_modes=True, separate_occupations=True)
