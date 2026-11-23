"""Partial population observables and masks from the same immutable evidence.

Grain Y already contains V_ref/S and Ms at the reading temperature. The
capacity multiplies it by w_number * Haar * V / D_full exactly once.
"""
import gzip
import hashlib
import json
import math
import re


def evidence_contributions(panel, connection, root):
    populations = {p['id']: p for p in panel['populations']}
    blocks = {b['id']: b for p in panel['populations'] for b in p['blocks']}
    histories = {h['id']: h for h in panel['histories']}
    result = {}
    selected = connection.execute(
        'SELECT id,task,commit_hash,path FROM attempts WHERE id IN '
        '(SELECT MAX(id) FROM attempts WHERE invalidated=0 GROUP BY task) '
        "AND task NOT LIKE 'G%' ORDER BY id").fetchall()
    for attempt, task, commit, path in selected:
        raw = (root / path).read_bytes()
        evidence = json.loads(gzip.decompress(raw))
        match = re.fullmatch(r'(A[1-4]_S\d+_C\d+)_R(\d+)_T(\d+)_(P12|P24)', task)
        if not match or evidence['task'] != task or evidence['commit'] != commit:
            raise ValueError('ATTEMPT_EVIDENCE_IDENTITY_MISMATCH')
        block_id, rotation_index, temperature, protocol = match.groups()
        population = block_id.split('_')[0]
        block = blocks[block_id]
        rotation = panel['rotations'][int(rotation_index)]
        if evidence['rule_id'] != panel['rule_id']:
            raise ValueError('ATTEMPT_QUADRATURE_IDENTITY_MISMATCH')
        indexed = connection.execute(
            'SELECT history,di,capacity,complete,qualified,readings,cause FROM nodes WHERE attempt=?',
            (attempt,)).fetchall()
        sql_rows = {(h, di): (cap, bool(comp), bool(qual), readings, cause)
                    for h, di, cap, comp, qual, readings, cause in indexed}
        if len(sql_rows) != len(indexed) or len(indexed) != len(evidence['nodes']):
            raise ValueError('DUPLICATE_OR_MISSING_INDEXED_NODE')
        source = dict(attempt=attempt, task=task, commit=commit, path=path,
                      compressed_sha256=hashlib.sha256(raw).hexdigest(),
                      tape_path=evidence['tape_path'],
                      tape_compressed_sha256=evidence['tape_compressed_sha256'])
        seen = set()
        for node in evidence['nodes']:
            h, di = node['history'], node['di']
            key = h, di
            if key in seen or key not in sql_rows:
                raise ValueError('DUPLICATE_OR_UNINDEXED_EVIDENCE_NODE')
            seen.add(key)
            expected_history = f'T{int(temperature):03d}_L{node["ell"]:02d}_{protocol}'
            if h != expected_history or h not in histories:
                raise ValueError('NODE_HISTORY_IDENTITY_MISMATCH')
            expected_capacity = (block['raw_number_weights'][di] * block['volumes_m3'][di]
                                 / populations[population]['denominator_raw_m3']
                                 * rotation['number_fraction'])
            if (not math.isfinite(node['capacity']) or node['capacity'] < 0
                    or not math.isclose(node['capacity'], expected_capacity, rel_tol=2e-13, abs_tol=1e-19)
                    or node['d_nm'] != block['d_nm'][di]):
                raise ValueError('NODE_FULL_MEASURE_MISMATCH')
            indexed_value = (node['capacity'], node['complete'], node['qualified'],
                             node['included_readings'], node['cause'])
            if indexed_value != sql_rows[key]:
                raise ValueError('SQL_AND_IMMUTABLE_NODE_MISMATCH')
            result.setdefault((population, h), []).append(dict(
                capacity=node['capacity'], complete=node['complete'], qualified=node['qualified'],
                included_readings=list(range(node['included_readings'])), cause=node['cause'],
                Y_F=node.get('Y_F', []), Y_N1=node.get('Y_N1', []),
                physical_node=f'{task}/D{di:03d}', source=source))
    return result


def partial_observables(panel, contributions):
    result = []
    sources = {}
    for population in panel['populations']:
        for history in panel['histories']:
            rows = contributions.get((population['id'], history['id']), [])
            count = history['observations']
            seen = set()
            terms = {m: [[[] for _ in range(3)] for _ in range(count)] for m in ['F', 'N1']}
            mask = []
            for row in rows:
                identity = row['physical_node']
                if identity in seen:
                    raise ValueError('OVERLAPPING_PHYSICAL_OBSERVABLE_NODES')
                seen.add(identity)
                readings = row['included_readings']
                if (len(set(readings)) != len(readings) or
                        any(j not in range(count) for j in readings)):
                    raise ValueError('INVALID_OBSERVABLE_READING_MASK')
                mask.append([identity, row['capacity'], readings])
                source = row.get('source')
                if source:
                    sources[source['attempt']] = source
                for model in terms:
                    values = row['Y_' + model]
                    if len(values) != len(readings):
                        raise ValueError('AVAILABLE_READING_HAS_NO_OBSERVABLE')
                    for j, vector in zip(readings, values):
                        if (len(vector) != 3 or not all(math.isfinite(x) for x in vector)
                                or math.sqrt(math.fsum(x*x for x in vector)) > 1 + 2e-10):
                            raise ValueError('INVALID_NORMALIZED_GRAIN_OBSERVABLE')
                        for c, value in enumerate(vector):
                            terms[model][j][c].append(row['capacity'] * value)
            included = [math.fsum(r['capacity'] for r in rows if j in r['included_readings'])
                        for j in range(count)]
            result.append(dict(population=population['id'], history=history['id'], protocol=history['protocol'],
                               rule_id=panel['rule_id'], denominator_raw_m3=population['denominator_raw_m3'],
                               physical_nodes=len(seen), f_included=included,
                               beta_U=[max(0., 1-f) for f in included],
                               mask_sha256=hashlib.sha256(json.dumps(sorted(mask), sort_keys=True,
                                                                    separators=(',', ':')).encode()).hexdigest(),
                               Y_partial={m: [[math.fsum(t) for t in reading] for reading in values]
                                          for m, values in terms.items()},
                               N0=None, N0_status='NOT_EVALUATED_IN_THIS_REDUCTION',
                               status='DESCRIPTIVE_PARTIAL_SUM' if seen else 'EMPTY_PARTIAL_SUM',
                               error_total=None, quadrature_error=None,
                               grain_Y_convention='V_ref Ms(20 C) R<u>/S; capacity supplies number weight, Haar and volume/full denominator once',
                               qualification='no inference to full continuous population; numerical terms remain open'))
    return dict(histories=result, sources=[sources[k] for k in sorted(sources)],
                summation='math.fsum on the same available reading mask as the coverage reduction',
                normalization='full parent denominator',
                empty_sum_convention='zero denotes sum of no included kernels, never a predicted full-population response')
