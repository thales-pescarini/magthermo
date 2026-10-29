"""Same MP polynomial with scalar contractions, scoped to the exact candidate."""
import contextlib
import functools
import hashlib
from pathlib import Path
import mpmath as mp
import numpy as np
import er2_engine as base
import g1_geometry as geometry
from er3_rfree_timing import RfreeTiming

VERSION='ER3_EXACT_MP_SCALAR_V1'
ORIGINAL_POLYNOMIAL=geometry.polynomial
ORIGINAL_SOURCE_SHA=geometry.SOURCE_SHA


@functools.lru_cache(maxsize=1)
def signature():
    return base.digest(dict(version=VERSION,original_geometry_sha256=ORIGINAL_SOURCE_SHA,
                            scalar_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            timing_source_sha256=hashlib.sha256(Path(__file__).with_name('er3_rfree_timing.py').read_bytes()).hexdigest()))


def polynomial(k1,k2,D,h,u):
    if D.rows!=3 or D.cols!=3 or u.rows!=3 or u.cols!=1 or any(D[a,b]!=D[b,a] for a,b in [(0,1),(0,2),(1,2)]):
        return ORIGINAL_POLYNOMIAL(k1,k2,D,h,u)
    x,y,z=u[0],u[1],u[2];q=[x*x,y*y,z*z];v=[x,y,z]
    Du=[mp.fdot([D[a,0],D[a,1],D[a,2]],v) for a in range(3)]
    E=k1*(q[0]*q[1]+q[0]*q[2]+q[1]*q[2])+k2*q[0]*q[1]*q[2]+mp.fdot(v,Du)-mp.fdot([h[0],h[1],h[2]],v)
    g=mp.matrix([2*v[a]*(k1*(q[(a+1)%3]+q[(a+2)%3])+k2*q[(a+1)%3]*q[(a+2)%3])+2*Du[a]-h[a] for a in range(3)])
    H=mp.matrix(3)
    for a in range(3):
        b,c=[j for j in range(3) if j!=a]
        H[a,a]=2*(k1*(q[b]+q[c])+k2*q[b]*q[c])+2*D[a,a]
        for b in range(3):
            if a!=b:H[a,b]=4*v[a]*v[b]*(k1+k2*q[3-a-b])+2*D[a,b]
    return E,g,H


class SharedCaches(RfreeTiming,base.SharedCaches):
    def physical_key(self,pm,kind,args,kwargs):
        return base.digest(dict(base=super().physical_key(pm,kind,args,kwargs),scalar_signature=signature()))

    @contextlib.contextmanager
    def pure_geometry_context(self):
        if geometry.polynomial is not ORIGINAL_POLYNOMIAL or geometry.SOURCE_SHA!=ORIGINAL_SOURCE_SHA:
            raise RuntimeError('ER3_MP_PROVIDER_CONTEXT_CONFLICT')
        try:
            geometry.polynomial=polynomial
            geometry.SOURCE_SHA=signature()
            with self.instrument_producers():
                with super().pure_geometry_context():yield
        finally:
            geometry.polynomial=ORIGINAL_POLYNOMIAL
            geometry.SOURCE_SHA=ORIGINAL_SOURCE_SHA

    def er2_snapshot(self):
        return dict(super().er2_snapshot(),exact_scalar_polynomial_version=VERSION,
                    exact_scalar_source_signature=signature(),precision_reduced=False)


class GeneralTape(base.GeneralTape):
    def prefix_geometry_key(self,graph):
        return base.digest(dict(base=super().prefix_geometry_key(graph),scalar_signature=signature()))
    def laboratory_geometry_key(self,land,p,protocol):
        zero=bool(np.array_equal(np.asarray(land['identity']['B_crystal_T']),np.zeros(3)))
        canonical=self.shared.zero_graphs.get(self.shared.zero_key(self,land['identity']['T_C']))
        same=canonical is not None and base.geometry_fingerprint(canonical)==base.geometry_fingerprint(land)
        prior_zero_events=[x for x in self.critical_catalog if np.array_equal(np.asarray(x['B_sample_T']),np.zeros(3))]
        prior_zero_solutions=[k for k in self.event_solutions if self.zero_field_segment(k)]
        verified=zero and land['identity']['T_C']==20. and same and not prior_zero_events and not prior_zero_solutions
        previous=self.shared.rfree_lab_verified_this_case
        self.shared.rfree_lab_verified_this_case=bool(verified and previous is not False)
        self.er3_Rfree_lab_certificate=dict(entry_field_exactly_zero=zero,entry_T_C=land['identity']['T_C'],
            entry_is_received_R_independent_zero_bank_geometry=same,
            prior_zero_critical_count=len(prior_zero_events),prior_zero_solution_count=len(prior_zero_solutions),
            laboratory_fields='exactly zero by unchanged inherited P12 chronology',
            placeholder='all map columns independent; only dimensions/mass checked; no magnetic states reused',
            source_scope='unchanged general_engine, factorized_executor and er2_engine; finite software dependency proof',
            verified=bool(verified))
        return base.digest(dict(base=super().laboratory_geometry_key(land,p,protocol),scalar_signature=signature()))


prepare_response=base.prepare_response
evaluate_duration=base.evaluate_duration
