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
    def __init__(self, *args, event_T_C, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_T_C = float(event_T_C)

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
        p, land = self.ramp(p, land, T0, zero, T0, field, event['field_on_ramp_seconds'], 'field_on')
        p, land = self.ramp(p, land, T0, field, T, field, (T-T0) / event['heating_rate_C_per_second'], 'event_heating')
        # Placeholder duration is replaced only at the explicit hold action.
        p = self.hold(p, land, T, 1., 'event_hold')
        p, land = self.ramp(p, land, T, field, T0, field, (T-T0) / event['cooling_rate_C_per_second'], 'event_cooling')
        p, land = self.ramp(p, land, T0, field, T0, zero, event['field_off_ramp_seconds'], 'field_off')
        p = self.hold(p, land, T0, event['final_zero_field_wait_seconds'], 'event_wait')
        self.readings.append(self.read(p, land, 'NRM'))
        for plateau in lab[protocol]['plateaus_C']:
            temperature = float(plateau)
            p, land = self.ramp(p, land, T0, zero, temperature, zero, (temperature-T0) / lab['heating_rate_C_per_second'], 'lab_heat_'+str(plateau))
            p = self.hold(p, land, temperature, lab['hold_seconds'], 'lab_hold_'+str(plateau))
            p, land = self.ramp(p, land, temperature, zero, T0, zero, (temperature-T0) / lab['cooling_rate_C_per_second'], 'lab_cool_'+str(plateau))
            p = self.hold(p, land, T0, lab['wait_before_measurement_seconds'], 'lab_wait_'+str(plateau))
            self.readings.append(self.read(p, land, str(plateau)))
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
    return results, dict(marginal_duration_s=time.monotonic()-start, ell=ell,
                         F_all_modes=True, separate_occupations=True)
