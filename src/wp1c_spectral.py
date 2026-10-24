"""E2.6 reversible F (every mode) and preparation-independent N1.

The eight frozen B2 numerical files are not modified. All MP values remain
MP through spectral selection and exponentiation. Only evaluated operators
are converted to float64 for the chronological linear-algebra consumer.
"""
import hashlib
import json
import math
import time
import numpy as np
import mpmath as mp
import e1_core as core
import thermal_executor as old

VERSION = 'WP1C_V0'


def maxabs(a):
    return max((abs(x) for x in a), default=mp.mpf(0))


def strings(a, digits=30):
    if isinstance(a, mp.matrix):
        return [[mp.nstr(a[i, j], digits) for j in range(a.cols)] for i in range(a.rows)]
    return [mp.nstr(v, digits) for v in a]


class Spectral:
    def __init__(self, Q, pi, dps=80, symmetric=None, identity=None):
        self.dps = int(dps)
        self.identity = identity
        self.start = time.monotonic()
        with mp.workdps(self.dps):
            self.Q = mp.matrix(Q)
            n = self.Q.rows
            self.n = n
            if n != self.Q.cols:
                raise core.Unresolved('NON_SQUARE_GENERATOR')
            if n > 1:
                reachable = {0}
                for _ in range(n):
                    reachable |= {i for j in list(reachable) for i in range(n) if i != j and self.Q[i, j] > 0}
                if len(reachable) != n:
                    raise core.Unresolved('N1_UNDEFINED_NONUNIQUE_EQUILIBRIUM')
            self.pi = mp.matrix(pi)
            if len(self.pi) != n or min(self.pi) <= 0:
                raise core.Unresolved('POSITIVE_EQUILIBRIUM_UNRESOLVED')
            self.pi /= mp.fsum(self.pi)
            sq = [mp.sqrt(x) for x in self.pi]
            self.sq = sq
            A = mp.matrix(symmetric) if symmetric is not None else mp.matrix(n, n)
            if symmetric is None:
                for i in range(n):
                    for j in range(n):
                        A[i, j] = self.Q[i, j] * sq[j] / sq[i]
            scale = max(mp.mpf(1), maxabs(self.Q))
            mass = max(abs(mp.fsum(self.Q[i, j] for i in range(n))) for j in range(n))
            stat = maxabs(self.Q * self.pi)
            reversible = maxabs(A - A.T)
            tol = mp.mpf('1e-20')
            if mass / scale > tol or stat / scale > tol or reversible / scale > tol:
                raise core.Unresolved('GENERATOR_OR_REVERSIBILITY_RESIDUAL')
            if any(self.Q[i, j] < 0 for i in range(n) for j in range(n) if i != j):
                raise core.Unresolved('NEGATIVE_OFF_DIAGONAL_RATE')
            self.projectors = [self.pi * mp.ones(1, n)]
            self.rates = [mp.mpf(0)]
            eigen_residuals = []
            if n == 2:
                # Exact binary reduction retains the entire one-dimensional
                # nonstationary subspace, including asymmetric/parallel rates.
                rate = self.Q[0, 1] + self.Q[1, 0]
                if rate <= 0:
                    raise core.Unresolved('POSITIVE_BINARY_RATE_UNRESOLVED')
                self.rates.append(rate)
                self.projectors.append(mp.eye(n) - self.projectors[0])
                eigen_residuals.append(maxabs(self.Q * self.projectors[1] + rate * self.projectors[1]) / rate)
                overlap = mp.mpf(1)
            elif n > 2:
                eig, Z = mp.eigsy(A)
                overlaps = [abs(mp.fsum(sq[i] * Z[i, j] for i in range(n))) for j in range(n)]
                zero = max(range(n), key=lambda j: overlaps[j])
                overlap = overlaps[zero]
                if abs(overlap - 1) > tol:
                    raise core.Unresolved('STATIONARY_VECTOR_NOT_IDENTIFIED')
                nonzero = sorted((j for j in range(n) if j != zero), key=lambda j: -eig[j])
                for j in nonzero:
                    rate = -eig[j]
                    if rate <= 0:
                        raise core.Unresolved('NONZERO_MODE_NOT_SEPARATED_FROM_ZERO')
                    r = mp.matrix([sq[i] * Z[i, j] for i in range(n)])
                    l = mp.matrix([[Z[i, j] / sq[i] for i in range(n)]])
                    P = r * l
                    residual = maxabs(A * Z[:, j] - eig[j] * Z[:, j]) / rate
                    if residual > mp.mpf('1e-15'):
                        raise core.Unresolved('SLOW_MODE_RELATIVE_RESIDUAL_UNRESOLVED')
                    self.rates.append(rate)
                    self.projectors.append(P)
                    eigen_residuals.append(residual)
            else:
                overlap = mp.mpf(1)
            total = mp.zeros(n)
            reconstruction = mp.zeros(n)
            for rate, P in zip(self.rates, self.projectors):
                total += P
                reconstruction -= rate * P
            self.diagnostics = dict(
                dps=self.dps, n_modes=n, retained_modes=len(self.rates),
                stationary_identification='unique positive equilibrium and sqrt(pi) overlap; no largest-rate cutoff',
                stationary_overlap=str(overlap), generator_mass_relative=str(mass / scale),
                stationary_residual_relative=str(stat / scale), reversibility_relative=str(reversible / scale),
                individual_nonzero_relative_residuals=strings(eigen_residuals),
                projector_completeness=str(maxabs(total - mp.eye(n))),
                generator_reconstruction_relative=str(maxabs(self.Q - reconstruction) / scale),
                all_individual_rates_preserved=True)
            self.wall_s = time.monotonic() - self.start

    def select(self, W):
        with mp.workdps(self.dps):
            if self.n == 1:
                return dict(status='TRIVIAL_IF_F_SUPPORTED', rate=mp.mpf(0), groups=[], W_convention='Ms R[u_i], A/m; no volume')
            w = mp.matrix(np.asarray(W, float).tolist())
            C = mp.eye(self.n) - mp.ones(self.n) / self.n
            groups = []
            for k in range(1, self.n):
                rate = self.rates[k]
                if groups and abs(rate - groups[-1]['min_rate']) <= mp.mpf('1e-8') * max(rate, groups[-1]['min_rate']):
                    g = groups[-1]
                    g['indices'].append(k)
                    g['projector'] += self.projectors[k]
                    g['max_rate'] = rate
                else:
                    groups.append(dict(indices=[k], min_rate=rate, max_rate=rate, projector=self.projectors[k].copy()))
            for g in groups:
                v = w * g['projector'] * C
                g['b'] = mp.sqrt(mp.fsum(x*x for x in v))
            denominator = mp.fsum(g['b'] for g in groups)
            if denominator <= 0:
                raise core.Unresolved('N1_UNDEFINED_VISIBILITY_DENOMINATOR')
            selected = None
            rows = []
            for g in groups:
                visibility = g['b'] / denominator
                rows.append(dict(indices=g['indices'], rate=mp.nstr(g['min_rate'], self.dps),
                                 max_rate=mp.nstr(g['max_rate'], self.dps),
                                 relative_spread=mp.nstr((g['max_rate']-g['min_rate'])/g['min_rate'], self.dps),
                                 b=mp.nstr(g['b'], self.dps), visibility=mp.nstr(visibility, self.dps)))
                if visibility > mp.mpf('1e-6') and selected is None:
                    selected = g['min_rate']
            if selected is None:
                raise core.Unresolved('N1_UNDEFINED_NO_VISIBLE_GROUP')
            return dict(status='DEFINED', rate=selected, groups=rows,
                        threshold='strict v > 1e-6', grouping_relative_tolerance='1e-8',
                        W_convention='Ms R[u_i], A/m; no volume', preparation_used=False)

    def propagator(self, seconds, model='F', selected=None, as_mp=False):
        if seconds < 0:
            raise ValueError('NEGATIVE_TIME')
        with mp.workdps(self.dps):
            t = mp.mpf(seconds)
            if model == 'N1':
                if selected is None:
                    raise ValueError('N1_REQUIRES_OPERATOR_SELECTION')
                rate = selected['rate']
                f = mp.exp(-rate*t)
                U = f*mp.eye(self.n) - mp.expm1(-rate*t)*self.projectors[0]
            else:
                U = mp.zeros(self.n)
                for rate, P in zip(self.rates, self.projectors):
                    U += mp.exp(-rate*t)*P
            mass = max(abs(mp.fsum(U[i, j] for i in range(self.n))-1) for j in range(self.n))
            minimum = min(U)
            if mass > mp.mpf('1e-12') or minimum < -mp.mpf('1e-12'):
                raise core.Unresolved('WP1C_OPERATOR_PROBABILITY_FAILURE')
            info = dict(mass_residual=float(mass), minimum=float(minimum), dps=self.dps, all_modes=(model=='F'))
            return (U if as_mp else np.array(U.tolist(),float)), info

    def summary(self, full=False):
        with mp.workdps(self.dps):
            out = dict(identity=self.identity, diagnostics=self.diagnostics,
                       rates=strings(self.rates,self.dps), pi=strings(self.pi,self.dps),
                       assembly_s=self.wall_s)
            if full:
                out['Q_columns'] = strings(self.Q,self.dps)
                out['projectors'] = [strings(P,self.dps) for P in self.projectors]
            return out


def physical_inputs(land, T, d):
    if land['status'] != 'COMPUTED_VALID':
        raise core.Unresolved('INVALID_GRAPH_NO_SPECTRUM')
    E = [p['E'] for p in land['minima']]
    n = len(E)
    V = math.pi/6*(d*1e-9)**3
    if n == 1 and not land['channels']:
        raise core.Unresolved('F_SINGLE_STATE_NOT_SUPPORTED_BY_FROZEN_OPERATOR')
    use_mp = land.get('canonical_geometry_and_energy',False) or min(min(c['barrier_density']) for c in land['channels']) < 1e-7*max(1.,max(abs(v) for v in E))
    energies, saddles = old.stable_energies(land) if use_mp else (E,[c['saddle_E'] for c in land['channels']])
    log_rates=[]
    for c,es in zip(land['channels'],saddles):
        for j in c['endpoints']:
            barrier=es-energies[j]
            if barrier <= 0:
                raise core.Unresolved('NONPOSITIVE_BARRIER_NO_CLIPPING')
            log_rates.append(math.log(1e9)-V*float(barrier)/(core.KB*(T+273.15)))
    span=(max(log_rates)-min(log_rates))/math.log(10)
    dps=int(65+1.3*span)
    if dps > 2048:
        raise core.Unresolved('FINITE_MP_PRECISION_CAP')
    return energies,saddles,V,dps,use_mp,span


def spectral_F(land, d, T, dps=None):
    energies,saddles,V,base_dps,use_mp,span=physical_inputs(land,T,d)
    precision=base_dps if dps is None else int(dps)
    if precision > 2048:
        raise core.Unresolved('FINITE_MP_PRECISION_CAP')
    with mp.workdps(precision):
        n=len(energies);e=list(map(mp.mpf,energies));Vm=mp.mpf(V)
        kT=mp.mpf(core.KB)*mp.mpf(T+273.15)
        Q=mp.zeros(n);A=mp.zeros(n)
        for c,es0 in zip(land['channels'],saddles):
            i,j=c['endpoints'];es=mp.mpf(es0)
            Q[j,i]+=mp.mpf(1e9)*mp.exp(-Vm*(es-e[i])/kT)
            Q[i,j]+=mp.mpf(1e9)*mp.exp(-Vm*(es-e[j])/kT)
            v=mp.mpf(1e9)*mp.exp(-Vm*(es-(e[i]+e[j])/2)/kT)
            A[i,j]+=v;A[j,i]+=v
        for j in range(n):
            Q[j,j]=-mp.fsum(Q[i,j] for i in range(n) if i!=j);A[j,j]=Q[j,j]
        ex=[-Vm*v/kT for v in e];top=max(ex);weights=[mp.exp(v-top) for v in ex]
        pi=mp.matrix(weights)/mp.fsum(weights)
        ident=dict(graph_identity=core.plain(land['identity']),d_nm=d,T_C=T,TM=0,
                   base_dps=base_dps,precision=precision,version=VERSION,
                   ordered_minima=[dict(id=p['point_id'],m=core.plain(p['m']),E=p.get('E_mp',p['E'])) for p in land['minima']],
                   ordered_channels=[dict(id=c['saddle_id'],endpoints=c['endpoints'],E=c.get('saddle_E_mp',c['saddle_E'])) for c in land['channels']])
        ident['spectral_key']=hashlib.sha256(json.dumps(ident,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        obj=Spectral(Q,pi,precision,A,ident)
        obj.diagnostics.update(log10_rate_span=span,stable_energy_refinement=use_mp,
                               channel_count=len(land['channels']),parallel_channels_summed=True)
        return obj


def observer(graph,R):
    return graph['identity']['Ms'] * (np.asarray(R,float) @ np.array([p['m'] for p in graph['minima']],float).T)
