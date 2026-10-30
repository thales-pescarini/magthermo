"""E1-R1 sequential midpoint thermal executor for finite point sentinels.

Exact field; crystal c and sample R kept separate. Caches contain physical
parameters and code identities. No population atlas or inverse solver here.
"""
from pathlib import Path
import math,time,json,hashlib,copy
import numpy as np
import mpmath as mp
from scipy.optimize import linear_sum_assignment,root
from scipy.optimize import brentq
import e1_core as core
import r1_landscape as ls
import g1_geometry as geom
import population_cdf as population

ROOT=Path(__file__).resolve().parents[1]
CODE_HASH=hashlib.sha256(b''.join((ROOT/'src'/p).read_bytes() for p in ['e1_core.py','r1_landscape.py','thermal_executor.py','population_cdf.py','g1_geometry.py','grain_api.py','factorized_executor.py','assembly.py'])).hexdigest()
S=4.884610123479026e-13

def checked(p,context):
    if not np.all(np.isfinite(p)):raise core.Unresolved('NONFINITE_PROBABILITY:'+context)
    p=np.asarray(p,float);res=float(p.sum()-1);minimum=float(np.min(p))
    if abs(res)>1e-12 or minimum< -1e-12:raise core.Unresolved('PROBABILITY_CONSERVATION_FAILURE:'+context+':'+str((res,minimum)))
    return dict(context=context,mass_residual_before_correction=res,minimum_probability=minimum,correction_applied=False)

def stable_energies(land):
    """Refine stationary roots for energy cancellation; coefficients unchanged."""
    if land.get('stable_energy_evidence'):
        cache=land['stable_energy_evidence']
        with mp.workdps(85):
            core.COUNTS['stable_energy_geometry_cache_hits']+=1
            return list(map(mp.mpf,cache['minimum_energies'])),list(map(mp.mpf,cache['saddle_energies']))
    core.COUNTS['stable_energy_geometry_cache_misses']+=1
    if land.get('canonical_geometry_and_energy'):
        with mp.workdps(85):
            byid={p['point_id']:p for p in land['points']}
            e=[mp.mpf(p['E_mp']) for p in land['minima']];es=[mp.mpf(byid[c['saddle_id']]['E_mp']) for c in land['channels']]
            land['stable_energy_evidence']=dict(dps=85,canonical_geometry_shared=True,minimum_energies=[str(v) for v in e],saddle_energies=[str(v) for v in es])
            return e,es
    i=land['identity'];D=np.asarray(i['N'])*(.5*core.legacy.MU0*i['Ms']**2)
    # Remove the same isotropic constant on the unit sphere before subtraction.
    D=D-np.trace(D)/3*np.eye(3)
    with mp.workdps(85):
        k1=mp.mpf(i['K1']);k2=mp.mpf(i['K2']);dm=mp.matrix(D.tolist());h=mp.matrix((i['Ms']*np.array(i['B_crystal_T'])).tolist())
        def energy(u):
            q=[x*x for x in u]
            return k1*(q[0]*q[1]+q[0]*q[2]+q[1]*q[2])+k2*q[0]*q[1]*q[2]+(u.T*dm*u)[0]-(h.T*u)[0]
        def equations(x,y,z,lag):
            u=mp.matrix([x,y,z]);q=[a*a for a in u]
            g=mp.matrix([2*u[a]*(k1*(q[(a+1)%3]+q[(a+2)%3])+k2*q[(a+1)%3]*q[(a+2)%3]) for a in range(3)])+2*dm*u-h
            return tuple(g[a]-lag*u[a] for a in range(3))+(x*x+y*y+z*z-1,)
        values={};roots=[]
        needed={p['point_id'] for p in land['minima']}|{ch['saddle_id'] for ch in land['channels']}
        for p in land['points']:
            if p['point_id'] not in needed:continue
            u=mp.matrix(p['m'].tolist());q=[x*x for x in u]
            g=mp.matrix([2*u[a]*(k1*(q[(a+1)%3]+q[(a+2)%3])+k2*q[(a+1)%3]*q[(a+2)%3]) for a in range(3)])+2*dm*u-h
            sol=mp.findroot(equations,tuple(u)+((u.T*g)[0],),tol=mp.mpf('1e-65'),maxsteps=50)
            v=mp.matrix(list(sol[:3]));move=float(mp.sqrt(mp.fsum((v[a]-u[a])**2 for a in range(3))))
            if move>1e-5:raise core.Unresolved('MP_ROOT_REFINEMENT_IDENTITY_CHANGE')
            values[p['point_id']]=energy(v)
            roots.append(dict(point_id=p['point_id'],direction_change=move,residual=str(max(abs(v) for v in equations(*sol)))))
        e=[values[p['point_id']] for p in land['minima']];es=[values[ch['saddle_id']] for ch in land['channels']]
        for ch,v in zip(land['channels'],es):
            if min(v-e[a] for a in ch['endpoints'])<=0:raise core.Unresolved('MP_CONFIRMED_NONPOSITIVE_BARRIER_NO_CLIPPING')
        land['stable_energy_evidence']=dict(dps=85,roots=roots,minimum_energies=[str(v) for v in e],saddle_energies=[str(v) for v in es],isotropic_constant_removed=True)
        return e,es

def operator(land,T,d,seconds):
    """Reversible spectral factorization in MP, exponentiating ALL modes."""
    if land['status']!='COMPUTED_VALID':raise core.Unresolved('INVALID_GRAPH_NO_OPERATOR')
    if seconds<0:raise ValueError('NEGATIVE_TIME')
    E=[p['E'] for p in land['minima']];N=len(E);V=math.pi/6*(d*1e-9)**3
    use_mp=land.get('canonical_geometry_and_energy',False) or min(min(ch['barrier_density']) for ch in land['channels'])<1e-7*max(1.,max(abs(v) for v in E))
    energies,saddles=stable_energies(land) if use_mp else (E,[ch['saddle_E'] for ch in land['channels']])
    rates=[]
    barriers=[]
    for ch,es in zip(land['channels'],saddles):
        for i in ch['endpoints']:
            barrier=es-energies[i]
            if barrier<=0:raise core.Unresolved('NONPOSITIVE_BARRIER_NO_CLIPPING')
            barriers.append(float(barrier))
            rates.append(math.log(1e9)-V*float(barrier)/(core.KB*(T+273.15)))
    span=(max(rates)-min(rates))/math.log(10);dps=int(65+1.3*span)
    if dps>2048:raise core.Unresolved('FINITE_MP_PRECISION_CAP')
    with mp.workdps(dps):
        e=list(map(mp.mpf,energies));Vm=mp.mpf(V);kT=mp.mpf(core.KB)*mp.mpf(T+273.15)
        Q=mp.zeros(N);A=mp.zeros(N)
        for ch,es0 in zip(land['channels'],saddles):
            a,b=ch['endpoints'];es=mp.mpf(es0)
            Q[b,a]+=mp.mpf(1e9)*mp.exp(-Vm*(es-e[a])/kT)
            Q[a,b]+=mp.mpf(1e9)*mp.exp(-Vm*(es-e[b])/kT)
            symmetric=mp.mpf(1e9)*mp.exp(-Vm*(es-(e[a]+e[b])/2)/kT)
            A[a,b]+=symmetric;A[b,a]+=symmetric
        for j in range(N):Q[j,j]=-mp.fsum(Q[i,j] for i in range(N) if i!=j);A[j,j]=Q[j,j]
        ex=[-Vm*v/kT for v in e];top=max(ex);z=mp.fsum(mp.exp(v-top) for v in ex)
        sq=[mp.sqrt(mp.exp(v-top)/z) for v in ex]
        eigen,vectors=mp.eigsy(A);factor=[mp.exp(v*mp.mpf(seconds)) for v in eigen]
        U=mp.matrix(N)
        for i in range(N):
            for j in range(N):U[i,j]=sq[i]/sq[j]*mp.fsum(vectors[i,k]*factor[k]*vectors[j,k] for k in range(N))
        mass=max(abs(mp.fsum(U[i,j] for i in range(N))-1) for j in range(N));minimum=min(U)
        if mass>mp.mpf('1e-12') or minimum<mp.mpf('-1e-12'):raise core.Unresolved('MP_OPERATOR_PROBABILITY_FAILURE')
        out=np.array(U.tolist(),float)
        qres=float(max(abs(mp.fsum(Q[i,j] for i in range(N))) for j in range(N)))
        dbres=float(max(abs(Q[i,j]*sq[j]**2-Q[j,i]*sq[i]**2) for i in range(N) for j in range(N)))
    core.COUNTS['all_mode_operators']+=1;core.COUNTS['MP_modes_retained']+=N
    return out,dict(dps=dps,n_modes=N,log10_rate_span=span,operator_mass_residual=float(mass),
        operator_minimum=float(minimum),min_barrier_over_kBT=V*min(barriers)/(core.KB*(T+273.15)),stable_energy_refinement=use_mp,
        generator_column_residual=qres,generator_detailed_balance_residual=dbres,
        channel_count=len(land['channels']),parallel_channels_summed=True)

class Executor:
    def __init__(self,case,resolution,protocols,checkpoint=None):
        self.case=case;self.level=resolution;self.protocols=protocols;self.s=case['s'];self.c=np.array(case['c_crystal_raw'],float);self.c/=np.linalg.norm(self.c)
        if case['R_crystal_to_sample']=='identity':self.R=np.eye(3);self.Rspec=None
        else:
            orientation=core.haar_orientations(12)[0];self.Rspec=orientation['euler_haar'];self.R=np.asarray(orientation['R_fs'])
        self.d=min(45.,.9*population.inherited.sd_limit(self.s));self.checkpoint=checkpoint
        self.alignment_failures=[];self.alignment_calls=0;self.maximum_alignment_distance=0.;self.field_connection_checks=[];self.critical_catalog=[];self.event_solutions={};self.pending_critical_hint=None;self.cache={};self.operators={};self.events=[];self.trace=[];self.audit=[];self.cost={k:0. for k in ['landscapes','ramps','events','holds','readings']};self.steps=0
        self.min_barrier_kBT=math.inf;self.max_mass_error=0.;self.min_probability=1.;self.start=time.monotonic()
        self.point_id=dict(s=self.s,c_crystal=self.c,R=self.R,d_nm=self.d,TM=0,code_sha256=CODE_HASH,conventions='axial_c_crystal_R_crystal_to_sample_number_then_volume_once')

    def pm(self,T,B):return core.PointModel(self.s,self.c,float(T),self.R.T@np.asarray(B),0)

    def census_levels(self,T,B):
        policy=json.loads((ROOT/'config/B2_CENSUS_POLICY.json').read_text())
        for target in policy['targets']:
            if self.s==target['s'] and np.array_equal(self.c,np.asarray(target['c'])) and np.array_equal(self.R,np.asarray(target['R'])) and np.array_equal(np.asarray(B),np.asarray(target['B_sample_T'])):
                return tuple(target['seed_pair'])
        return tuple(policy['otherwise'])

    def transport(self,P,p,label,event_record=None):
        """Linear state map hook; occupation always belongs to this trajectory."""
        return P@p

    def transaction_token(self):return None
    def rollback_transaction(self,token):pass

    def key(self,T,B):
        return hashlib.sha256(json.dumps(core.plain(dict(**self.point_id,T_C=float(T),B_sample_T=np.asarray(B))),sort_keys=True).encode()).hexdigest()

    def stationary_points(self,T,B,old):
        """Continue away from a known soft root with finite geometric steps.

        This corrects branch loss after birth. Each correction retains indices,
        uniqueness, residual and direction tests; failure remains unresolved.
        No saddle flow, time evolution or occupation occurs in this routine.
        """
        Tstart=old['identity']['T_C'];Bstart=self.R@np.array(old['identity']['B_crystal_T'])
        delta=abs(float(T)-Tstart);fractions=[1.]
        if delta>0 and np.linalg.norm(np.asarray(B)-Bstart)<1e-15:
            nearby=[abs(Tstart-x['T_C']) for x in self.critical_catalog
                if np.linalg.norm(np.array(x['B_sample_T'])-Bstart)<1e-15
                and (Tstart-x['T_C'])*(float(T)-x['T_C'])>0]
            distance=min(nearby,default=math.inf)
            if 0<distance<.02 and delta>distance:
                first=distance/delta;fractions=[]
                for j in range(48):
                    f=first*(2.**j-1.)
                    if f>=1:break
                    if f>0:fractions.append(f)
                fractions.append(1.)
        points=old['points'];okay=True
        for f in fractions:
            pm=self.pm(Tstart+f*(float(T)-Tstart),Bstart+f*(np.asarray(B)-Bstart))
            points,okay=ls.continuous_points(pm,points)
            if not okay:break
        self.geometric_continuation_steps=getattr(self,'geometric_continuation_steps',0)+len(fractions)
        return points,okay

    def landscape(self,T,B,previous=None,force=False):
        key=self.key(T,B)
        if key in self.cache:return self.cache[key]
        t=time.monotonic();pm=self.pm(T,B)
        if previous is not None and not force:
            points,valid=self.stationary_points(T,B,previous)
            if valid:
                try:
                    refresh=(abs(self.c[0]-self.c[1])<1e-14 and abs(self.c[2])<1e-14 and np.linalg.norm(np.array(previous['identity']['B_crystal_T'])-pm.B)>0)
                    if refresh:
                        self.diagnostic=dict(kind='FIELD_CONNECTION_VERIFICATION',identity=pm.identity(),points=points,status='RUNNING')
                        if self.checkpoint:self.checkpoint(dict(diagnostic=self.diagnostic,events=self.events))
                    land=ls.from_points(pm,points,None if refresh else previous)
                    if refresh:
                        land['field_connections_freshly_verified']=True
                        self.field_connection_checks.append(dict(identity=pm.identity(),connections=land['connections'],channels=land['channels'],status='COMPUTED_VALID'))
                except core.Unresolved as exc:
                    if refresh:
                        self.diagnostic=dict(kind='FIELD_CONNECTION_FAILURE',identity=pm.identity(),evidence=getattr(exc,'evidence',{}),reason=str(exc))
                        if self.checkpoint:self.checkpoint(dict(diagnostic=self.diagnostic,events=self.events))
                        raise core.Unresolved('FIELD_CONNECTION_CLASSIFICATION_UNRESOLVED')
                    valid=False
            if not valid:raise core.Unresolved('STATIONARY_EVENT_CANDIDATE')
        else:
            land=ls.enumerate_land(pm,self.census_levels(T,B))
            if land['status']!='COMPUTED_VALID':
                self.diagnostic=dict(kind='FULL_GRAPH_FAILURE',graph=land)
                if self.checkpoint:self.checkpoint(dict(diagnostic=self.diagnostic,events=self.events))
                raise core.Unresolved('FULL_CENSUS_OR_CONNECTION_UNRESOLVED:'+str(land.get('reason')))
        self.cache[key]=land;self.cost['landscapes']+=time.monotonic()-t
        return land

    def align(self,source,target,occupation=None):
        """Bijection on the same side; never a merge, reset or relaxed match."""
        T=target['identity']['T_C'];B=self.R@np.array(target['identity']['B_crystal_T'])
        points,ok=self.stationary_points(T,B,source)
        U=np.array([p['m'] for p in points if p['index']==0]);V=np.array([p['m'] for p in target['minima']])
        D=np.linalg.norm(U[:,None]-V[None,:],axis=2) if len(U) and len(V) else np.empty((len(U),len(V)))
        a,b=linear_sum_assignment(D) if len(U) and len(V) else (np.array([],int),np.array([],int))
        error=float(max(D[a,b],default=0.))
        reason=None
        if not ok:reason='IDENTITY_CONTINUATION_REQUIRES_EVENT'
        elif len(U)!=len(V):reason='STATE_COUNT_CHANGED_WITHOUT_EVENT'
        elif len(a)!=len(U) or error>1e-6:reason='CONTINUED_STATE_NOT_IN_TARGET_CENSUS'
        if reason:
            def sep(X):return min((np.linalg.norm(v-w) for j,v in enumerate(X) for w in X[:j]),default=None)
            record=dict(kind='ALIGN_FAILURE',reason=reason,T_C=T,B_sample_T=B,
                source_graph=source,target_graph=target,continued_points=points,
                source_ids=[p['point_id'] for p in source['minima']],target_ids=[p['point_id'] for p in target['minima']],
                directions_before=np.array([p['m'] for p in source['minima']]),directions_after=U,
                distance_matrix=D,assignment_rows=a,assignment_columns=b,maximum_effective_distance=error,
                matching_tolerance=1e-6,internal_angular_target=1e-9,separations=[sep(U),sep(V)],occupation_input=occupation)
            self.alignment_failures.append(record);self.diagnostic=record
            if self.checkpoint:self.checkpoint(dict(diagnostic=record,events=self.events,checkpoint=getattr(self,'current',None)))
            raise core.Unresolved(reason)
        self.maximum_alignment_distance=max(self.maximum_alignment_distance,error);self.alignment_calls+=1
        P=np.zeros((len(V),len(U)));P[b,a]=1.;return P

    def apply(self,p,land,T,seconds,label):
        key=(self.key(T,self.R@np.array(land['identity']['B_crystal_T'])),float(seconds),self.d)
        # Operator ordering is tied to the exact cached minima order.
        order=tuple(tuple(p['m']) for p in land['minima']);key=key+(order,)
        if key not in self.operators:self.operators[key]=operator(land,T,self.d,seconds)
        U,info=self.operators[key];new=U@p;record=checked(new,label)
        self.max_mass_error=max(self.max_mass_error,abs(record['mass_residual_before_correction']));self.min_probability=min(self.min_probability,record['minimum_probability']);self.min_barrier_kBT=min(self.min_barrier_kBT,info['min_barrier_over_kBT']);self.steps+=1
        return new

    def event(self,T0,B0,T1,B1,old):
        """Keep the reference path first; narrow failed reverse-event anchors.

        A wide anchor can straddle a second, already encountered fold.  Root
        continuation must then reject that anchor rather than associate roots
        across the extra event.  Retry only the existing smaller temperature
        anchors, before any map/cache commit, with the exact same known critical
        point.  No occupation or transport rule is replaced.
        """
        hint=copy.deepcopy(self.pending_critical_hint)
        start=len(self.events);attempts=[];family=[.1,.02,.004]
        if getattr(self,'_retry_lateral_anchor_offsets',None) is not None:
            raise core.Unresolved('LATERAL_ANCHOR_RETRY_CONTEXT_ALREADY_ACTIVE')
        try:
            for _ in family:
                try:
                    result=self._event_reference(T0,B0,T1,B1,old)
                except core.Unresolved as exc:
                    record=self.events[-1] if len(self.events)>start else {}
                    if attempts:
                        record['lateral_anchor_retry_history']=copy.deepcopy(attempts)
                    eligible=(str(exc)=='LATERAL_ROOT_CONTINUATION_UNRESOLVED'
                        and hint is not None and float(T0)!=float(T1)
                        and np.array_equal(np.asarray(B0),np.zeros(3))
                        and np.array_equal(np.asarray(B1),np.zeros(3))
                        and record.get('critical_root',{}).get('reversed_known_critical_location') is True
                        and record.get('status')=='NUMERICAL_UNRESOLVED'
                        and 'pre_graph' not in record and 'post_graph' not in record
                        and record.get('selected_anchor_offset') in family
                        and record.get('lateral_continuation_in_progress',[])
                        and record['lateral_continuation_in_progress'][-1]['valid'] is False)
                    if not eligible:raise
                    selected=record['selected_anchor_offset']
                    remaining=family[family.index(selected)+1:]
                    if not remaining:raise
                    attempts.append(copy.deepcopy(record))
                    del self.events[start:]
                    self.pending_critical_hint=copy.deepcopy(hint)
                    self._retry_lateral_anchor_offsets=remaining
                    continue
                if attempts:
                    record=self.events[-1]
                    record['lateral_anchor_retry_history']=copy.deepcopy(attempts)
                    record['lateral_retry_scope']='same known reverse critical point; smaller declared anchors; unchanged transport'
                    key=(float(T0),tuple(np.asarray(B0)),float(T1),tuple(np.asarray(B1)))
                    if key in self.event_solutions:
                        entry=self.event_solutions[key]
                        self.event_solutions[key]=(*entry[:-1],copy.deepcopy(record))
                return result
            raise core.Unresolved('LATERAL_ANCHOR_RETRY_FAMILY_EXHAUSTED')
        finally:
            # Retain every rejected anchor even if a later, unrelated error
            # occurs after the two verified sides and before transport commit.
            if attempts and len(self.events)>start:
                self.events[-1]['lateral_anchor_retry_history']=copy.deepcopy(attempts)
            if hasattr(self,'_retry_lateral_anchor_offsets'):
                del self._retry_lateral_anchor_offsets

    def _event_reference(self,T0,B0,T1,B1,old):
        t=time.monotonic();T0=float(T0);T1=float(T1);B0=np.asarray(B0);B1=np.asarray(B1)
        temperature=abs(T1-T0)>0;tol=1e-6/abs(T1-T0) if temperature else 1e-8/max(float(np.linalg.norm((B1-B0)/5e-5)),1e-300)
        event_key=(float(T0),tuple(B0),float(T1),tuple(B1))
        if event_key in self.event_solutions:
            lo,hi,left,right,P,template=self.event_solutions[event_key]
            rec=copy.deepcopy(template);rec['graph_verification_reused_at_identical_segment']=True
            rec.pop('occupation_before_map',None);rec.pop('occupation_after_map',None);self.events.append(rec)
            return lo,hi,left,right,P.copy()
        hint=self.pending_critical_hint;self.pending_critical_hint=None
        early=dict(T_segment=[T0,T1],B_segment_T=[B0,B1],status='NUMERICAL_UNRESOLVED',fraction_bracket=[0.,1.],reason='CRITICAL_LOCATION_IN_PROGRESS')
        self.events.append(early)
        if self.checkpoint:self.checkpoint(dict(events=self.events,checkpoint=getattr(self,'current',None)))
        lo=0.;hi=1.;left=old;right=None;history=[]
        mirror_critical=None
        if hint is None and np.array_equal(B0,np.zeros(3)) and np.array_equal(B1,np.zeros(3)):
            mirror_critical=geom.locate_mirror_critical(lambda f:self.pm(T0+f*(T1-T0),B0),T0,T1)
        for iteration in range(64):
            if hint is not None or mirror_critical is not None:break
            if hi-lo<=tol:break
            mid=(lo+hi)/2;T=T0+mid*(T1-T0);B=B0+mid*(B1-B0);pm=self.pm(T,B)
            pts,valid=ls.continuous_points(pm,left['points'])
            eig=min((abs(e) for p in pts for e in p['eig']),default=math.inf)
            history.append(dict(fraction=mid,same_stationary_branches=valid,minimum_absolute_curvature=eig,
                geometry_is_current_verified_roots=valid))
            if valid:
                try:left=ls.from_points(pm,pts,left);lo=mid
                except core.Unresolved:hi=mid
            else:hi=mid
        if hint is not None:
            fc=(hint['T_C']-T0)/(T1-T0);lo=max(0.,fc-tol*.49);hi=min(1.,fc+tol*.49)
        elif mirror_critical is not None:
            fc=mirror_critical['fraction'];lo=max(0.,fc-tol*.49);hi=min(1.,fc+tol*.49)
        if hi-lo>tol:raise core.Unresolved('EVENT_BISECTION_LIMIT')
        # Resolve the critical root itself, separately from the Morse sides.
        # The diagonal zero-field axial branch exists analytically on both
        # sides; other folds use the stationary KKT plus det(H_tangent)=0.
        critical=dict(hint['critical'],fraction=fc,reversed_known_critical_location=True) if hint is not None else mirror_critical
        if critical is None and np.linalg.norm(B0)==0 and np.linalg.norm(B1)==0 and np.count_nonzero(abs(self.c)>1e-14)==1:
            axis=int(np.argmax(abs(self.c)));u=np.eye(3)[axis]
            def curvature(f):
                pm=self.pm(T0+f*(T1-T0),B0)
                return 2*geom.axial_A(pm) if geom.applicable(pm) else float(np.linalg.eigvalsh(ls.gate.riem(pm,u)[1])[0])
            ca,cb=curvature(0.),curvature(1.)
            if ca*cb<0:
                fc,info=brentq(curvature,0.,1.,xtol=max(1e-14,tol*.01),maxiter=64,full_output=True)
                critical=dict(kind='analytic_axial_branch_eigenvalue_root',fraction=fc,root_direction=u,
                    eigenvalue_at_root=curvature(fc),initial_signed_eigenvalues=[ca,cb],iterations=info.iterations)
        if critical is None:
            candidate=min(left['points'],key=lambda p:min(abs(np.array(p['eig']))));u=np.array(candidate['m']);sc=ls.scale(self.pm(T0,B0))
            def F(x):
                pm=self.pm(T0+x[4]*(T1-T0),B0+x[4]*(B1-B0));v=x[:3];vn=v/np.linalg.norm(v)
                g=pm.grad_np(v)/sc-x[3]*v;H=ls.gate.riem(pm,vn)[1]/sc
                return np.r_[g,.5*(v@v-1),np.linalg.det(H)]
            pmc=self.pm(T0+lo*(T1-T0),B0+lo*(B1-B0));initial=np.r_[u,u@pmc.grad_np(u)/sc,(lo+hi)/2]
            sol=root(F,initial,tol=1e-11,options={'maxfev':160});res=float(np.linalg.norm(F(sol.x)))
            fc=float(sol.x[4]);pmc=self.pm(T0+fc*(T1-T0),B0+fc*(B1-B0));eig=np.linalg.eigvalsh(ls.gate.riem(pmc,sol.x[:3]/np.linalg.norm(sol.x[:3]))[1])
            if res<1e-10 and 0<fc<1 and lo-4*tol<fc<hi+4*tol and min(abs(eig))<1e-5:
                critical=dict(kind='fold_stationary_KKT',fraction=fc,root_direction=sol.x[:3]/np.linalg.norm(sol.x[:3]),KKT_residual=res,eigenvalues=eig,scipy_success=bool(sol.success))
        record=dict(T_segment=[T0,T1],B_segment_T=[B0,B1],fraction_bracket=[lo,hi],
            history=history,critical_root=critical,status='NUMERICAL_UNRESOLVED',anchors=[],
            transactional_input_restored=True)
        early.clear();early.update(record);record=early
        def persist():
            self.diagnostic=dict(kind='EVENT_IN_PROGRESS',event=record)
            if self.checkpoint:self.checkpoint(dict(events=self.events,diagnostic=self.diagnostic,checkpoint=getattr(self,'current',None)))
        persist()
        if critical is None:
            record['reason']='CRITICAL_ROOT_NOT_RESOLVED';persist();raise core.Unresolved(record['reason'])
        fc=critical['fraction'];lo=max(0.,fc-tol*.49);hi=min(1.,fc+tol*.49)
        record.update(fraction_bracket=[lo,hi],temperature_bracket_C=[T0+lo*(T1-T0),T0+hi*(T1-T0)],field_fraction_width=hi-lo)
        offsets=[.1,.02,.004] if temperature else [.01,.002,.0004]
        retry_offsets=getattr(self,'_retry_lateral_anchor_offsets',None)
        if retry_offsets is not None:offsets=retry_offsets
        anchor_left=None;anchor_right=None
        for offset in offsets:
            delta=offset/abs(T1-T0) if temperature else offset/max(np.linalg.norm((B1-B0)/5e-5),1e-300)
            fl,fr=fc-delta,fc+delta
            al=dict(offset=offset,fractions=[fl,fr],status='RUNNING');record['anchors'].append(al);persist()
            if not 0<fl<fr<1:
                al.update(status='NUMERICAL_UNRESOLVED',reason='ANCHOR_OUTSIDE_ORIGINAL_SEGMENT');persist();continue
            Tl0=T0+fl*(T1-T0);Tr0=T0+fr*(T1-T0);Bl0=B0+fl*(B1-B0);Br0=B0+fr*(B1-B0)
            if not (20<=Tl0<=350 and 20<=Tr0<=350) or max(np.linalg.norm(Bl0),np.linalg.norm(Br0))>5e-5*(1+1e-12):
                al.update(status='NUMERICAL_UNRESOLVED',reason='ANCHOR_OUTSIDE_PHYSICAL_RAMP_DOMAIN');persist();continue
            anchor_left=ls.enumerate_land(self.pm(Tl0,Bl0),self.census_levels(Tl0,Bl0));al['left_graph']=anchor_left;persist()
            anchor_right=ls.enumerate_land(self.pm(Tr0,Br0),self.census_levels(Tr0,Br0));al['right_graph']=anchor_right;persist()
            continued,consistent=self.stationary_points(Tl0,Bl0,old)
            same=consistent and ls.gate.same_census(dict(points=continued,counts={k:sum(p['index']==k for p in continued) for k in range(3)}),dict(points=anchor_left.get('points',[]),counts={k:sum(p['index']==k for p in anchor_left.get('points',[])) for k in range(3)}))
            al['continued_old_roots_agree_with_full_left_census']=bool(same)
            if anchor_left['status']=='COMPUTED_VALID' and anchor_right['status']=='COMPUTED_VALID' and same and anchor_left['counts']!=anchor_right['counts']:
                al['status']='COMPUTED_VALID';break
            al.update(status='NUMERICAL_UNRESOLVED',reason='LATERAL_CENSUS_CONNECTION_OR_TOPOLOGY_NOT_RESOLVED');persist()
        else:
            record['reason']='FINITE_LATERAL_ANCHORS_EXHAUSTED';persist()
            from a2_census_repair import recover_anchor_censuses
            repaired=recover_anchor_censuses(self,T0,B0,T1,B1,old,offsets,record,persist)
            if repaired is None:raise core.Unresolved(record['reason'])
            anchor_left,anchor_right,fl,fr,offset,Tl0,Tr0,Bl0,Br0=repaired
            if self.pm(Tl0,Bl0).identity()!=anchor_left['identity'] or self.pm(Tr0,Br0).identity()!=anchor_right['identity']:
                raise core.Unresolved('A2_SELECTED_ANCHOR_MODEL_IDENTITY_MISMATCH')
            record['A2_selected_anchor_model_identity_verified']=True
            record.pop('reason',None)
        record['selected_anchor_offset']=offset
        def continue_side(anchor,f):
            # A finite geometric continuation to the requested one-sided limit.
            af=fl if anchor is anchor_left else fr;land=anchor;checks=[]
            distance=abs(af-fc);target=abs(f-fc);side=-1 if af<fc else 1;fractions=[]
            for j in range(23):
                if distance<=2*target:break
                distance/=2;fractions.append(fc+side*distance)
            if distance>2*target:raise core.Unresolved('B2_LATERAL_24_STEP_CAP')
            fractions.append(f)
            for frac in fractions:
                pm=self.pm(T0+frac*(T1-T0),B0+frac*(B1-B0));pts,ok=ls.continuous_points(pm,land['points'])
                checks.append(dict(fraction=frac,valid=ok,counts=[sum(p['index']==k for p in pts) for k in range(3)] if ok else None,
                    maximum_gradient_residual=max(p['gradient_residual'] for p in pts) if ok else None,minimum_absolute_curvature=min(abs(v) for p in pts for v in p['eig']) if ok else None,
                    minimum_point_separation=min(np.linalg.norm(p['m']-q['m']) for j,p in enumerate(pts) for q in pts[:j]) if ok else None))
                record['lateral_continuation_in_progress']=checks;persist()
                if not ok:raise core.Unresolved('LATERAL_ROOT_CONTINUATION_UNRESOLVED')
                land=ls.from_points(pm,pts,land)
            return land,checks
        try:
            left,lchecks=continue_side(anchor_left,lo);right,rchecks=continue_side(anchor_right,hi)
        except core.Unresolved as exc:
            record['reason']=str(exc);persist();raise
        record['lateral_continuation']={'left':lchecks,'right':rchecks}
        Tl=T0+lo*(T1-T0);Bl=B0+lo*(B1-B0);Tr=T0+hi*(T1-T0);Br=B0+hi*(B1-B0)
        self.cache[self.key(Tl,Bl)]=left;self.cache[self.key(Tr,Br)]=right
        record.update(pre_graph=left,post_graph=right);persist()
        pmr=self.pm(Tr,Br);oldmin=left['minima'];newmin=right['minima'];U=np.array([p['m'] for p in oldmin]);V=np.array([p['m'] for p in newmin]);P=np.zeros((len(V),len(U)))
        zero_exact=geom.applicable(self.pm(Tl,Bl)) and geom.applicable(pmr)
        if zero_exact:
            P,maps,zero_proof=geom.zero_field_map(self.pm(Tl,Bl),pmr,left,right);born=[]
            record['zero_field_map_proof']=zero_proof
        else:
            cont,ok=ls.batch_newton(pmr,U);survive={};dead=[]
            record['minimum_continuation_checks']=[]
            for i,u in enumerate(cont):
                check=dict(old=i,Newton_converged=bool(ok[i]),candidate_direction=u.copy())
                record['minimum_continuation_checks'].append(check)
                if not ok[i]:
                    # A failed continuation is not a stationary root. The
                    # destination still requires the two verified lateral flows.
                    check['disposition']='requires_verified_lateral_destination'
                    dead.append(i);continue
                p=ls.point(pmr,u);u=p['m'];dist=np.linalg.norm(V-u,axis=1);j=int(np.argmin(dist))
                check.update(corrected_direction=u,index=p['index'],nearest_distance=float(dist[j]))
                if p['index']==0 and dist[j]<1e-6 and np.linalg.norm(u-U[i])<.15:survive[i]=j
                else:dead.append(i)
            for i,j in survive.items():P[j,i]=1
            born=[j for j in range(len(V)) if j not in survive.values()];maps=[]
            # Symmetry reverse: a surviving stationary parent can become unstable;
            # daughter roots must form an orbit of verified energy symmetries.
            for i in dead:
                distances=np.linalg.norm(V-U[i],axis=1);near=[j for j in born if distances[j]<.1]
                symmetric,group=ls.cubic_symmetries(pmr,U[i],[V[j] for j in near]) if near else (False,[])
                if symmetric:
                    for j in near:P[j,i]=1/len(near)
                    maps.append(dict(old=i,kind='verified_symmetric_reverse',destinations=near,symmetries=group))
                else:
                    left_byid={p['point_id']:p for p in anchor_left['minima']}
                    start=left_byid[oldmin[i]['point_id']]['m']
                    flows=[ls.flow(self.pm(Tr0,Br0),start,anchor_right['minima'],tolr) for tolr in [1e-9,1e-11]]
                    if flows[0]['destination'] is None or flows[0]['destination']!=flows[1]['destination'] or not all(f['energy_monotone_decreasing'] for f in flows):
                        record['reason']='DISAPPEARING_MINIMUM_FLOW_DESTINATION_UNRESOLVED';record['failed_old_minimum']=i;record['flows']=flows
                        raise core.Unresolved(record['reason'])
                    j=flows[-1]['destination'];P[j,i]=1;maps.append(dict(old=i,kind='death_flow_at_verified_lateral_reference',destinations=[j],flows=flows,reference_fraction=fr))
        if np.max(abs(P.sum(axis=0)-1))>1e-12 or np.min(P)<-1e-12:raise core.Unresolved('EVENT_MAP_NOT_STOCHASTIC')
        # Evidence for the critical root: tangential eigenvalue bracket of a
        # continued root, or a located fold KKT with opposite min/saddle sides.
        evidence=[]
        pmleft=self.pm(T0+lo*(T1-T0),B0+lo*(B1-B0))
        for a in left['points']:
            v,okay=ls.batch_newton(pmr,[a['m']])
            if not okay[0]:continue
            try:b=ls.point(pmr,v[0])
            except core.Unresolved:continue
            if okay[0] and np.linalg.norm(v[0]-a['m'])<.1:
                ea=np.array(a['eig']);eb=np.array(b['eig'])
                if np.any(ea*eb<0):evidence.append(dict(kind='signed_tangent_eigenvalue_bracket',before=ea,after=eb,direction=a['m']))
        if critical is not None and critical['kind']=='fold_stationary_KKT':
            direction=np.array(critical['root_direction']);existence=left if len(left['points'])>len(right['points']) else right
            near=sorted(existence['points'],key=lambda p:np.linalg.norm(p['m']-direction))[:2]
            if len(near)==2 and abs(near[0]['index']-near[1]['index'])==1 and max(np.linalg.norm(p['m']-direction) for p in near)<.01:
                signed=[float(min(p['eig'],key=abs)) for p in near]
                if signed[0]*signed[1]<0:evidence.append(dict(kind='fold_pair_opposite_tangent_curvatures_and_KKT_root',existence_side_eigenvalues=signed,temperature_enclosure_empirical=True))
        record.update(old_minima=U,new_minima=V,map=P,mapping_evidence=maps,isolated_births=born,curvature_evidence=evidence)
        if not evidence:
            record['reason']='EVENT_HAS_NO_DEMONSTRATED_SIGNED_CURVATURE_BRACKET'
            raise core.Unresolved(record['reason'])
        record['status']='COMPUTED_VALID';record.pop('reason',None);persist();self.cost['events']+=time.monotonic()-t
        if temperature and np.linalg.norm(B1-B0)==0 and not any(abs(x['T_C']-(T0+fc*(T1-T0)))<1e-6 and np.linalg.norm(np.array(x['B_sample_T'])-B0)<1e-15 for x in self.critical_catalog):
            self.critical_catalog.append(dict(T_C=T0+fc*(T1-T0),B_sample_T=B0.copy(),critical=copy.deepcopy(critical)))
        self.event_solutions[event_key]=(lo,hi,left,right,P.copy(),copy.deepcopy(record))
        return lo,hi,left,right,P

    def advance(self,p,old,T0,B0,T1,B1,seconds,label,depth=0):
        if depth>4:raise core.Unresolved('TOO_MANY_EVENTS_IN_ONE_PRESCRIBED_STEP')
        B0=np.asarray(B0);B1=np.asarray(B1);Tm=(T0+T1)/2;Bm=(B0+B1)/2
        input_p=np.array(p,copy=True)
        transaction=self.transaction_token()
        hint=None
        if T0!=T1 and np.linalg.norm(B1-B0)==0:
            candidates=[x for x in self.critical_catalog if min(T0,T1)<x['T_C']<max(T0,T1) and np.linalg.norm(np.array(x['B_sample_T'])-B0)<1e-15]
            if candidates:hint=min(candidates,key=lambda x:abs(x['T_C']-T0))
        self.pending_critical_hint=hint
        try:
            if hint is not None:raise core.Unresolved('STATIONARY_EVENT_CANDIDATE')
            mid=self.landscape(Tm,Bm,old);end=self.landscape(T1,B1,mid)
            p=self.transport(self.align(old,mid,p),p,label+':align_mid');p=self.apply(p,mid,Tm,seconds,label);p=self.transport(self.align(mid,end,p),p,label+':align_end');checked(p,label+':identity')
            return p,end
        except core.Unresolved as e:
            if str(e) not in ['STATIONARY_EVENT_CANDIDATE','IDENTITY_CONTINUATION_REQUIRES_EVENT','CHANNEL_TOPOLOGY_CHANGE','NONPOSITIVE_BARRIER_NO_CLIPPING']:raise
        self.rollback_transaction(transaction)
        p=input_p.copy()
        lo,hi,left,right,P=self.event(T0,B0,T1,B1,old)
        Tl=T0+lo*(T1-T0);Bl=B0+lo*(B1-B0);Tr=T0+hi*(T1-T0);Br=B0+hi*(B1-B0)
        event_left=left
        if lo>0:
            p,left=self.advance(p,old,T0,B0,Tl,Bl,seconds*lo,label+':pre',depth+1)
            p=self.transport(self.align(left,event_left,p),p,label+':align_event_left');left=event_left
        p=self.apply(p,left,Tl,seconds*(hi-lo)/2,label+':event_left')
        before_map=p.copy();p=self.transport(P,p,label+':event_map',self.events[-1]);map_check=checked(p,label+':event_map')
        self.events[-1].update(occupation_before_map=before_map,occupation_after_map=p.copy(),probability_check=map_check,physical_step_seconds=seconds,time_partition_seconds=[seconds*lo,seconds*(hi-lo)/2,seconds*(hi-lo)/2,seconds*(1-hi)],event_segment_label=label)
        p=self.apply(p,right,Tr,seconds*(hi-lo)/2,label+':event_right')
        if hi<1:p,end=self.advance(p,right,Tr,Br,T1,B1,seconds*(1-hi),label+':post',depth+1)
        else:end=right
        return p,end

    def ramp(self,p,land,T0,B0,T1,B1,seconds,label):
        t=time.monotonic()
        if T1!=T0:n=max(1,int(math.ceil(abs(T1-T0)/self.level['max_temperature_step_C'])))
        else:n=self.level['field_ramp_substeps']
        for k in range(n):
            a=k/n;b=(k+1)/n
            p,land=self.advance(p,land,T0+a*(T1-T0),np.array(B0)+a*(np.array(B1)-B0),T0+b*(T1-T0),np.array(B0)+b*(np.array(B1)-B0),seconds/n,label)
            if k%16==0:self.snapshot(p,land,label,k,n)
        self.cost['ramps']+=time.monotonic()-t
        self.trace.append(dict(segment=label,T_start=T0,T_end=T1,seconds=seconds,substeps=n,state=p,states=len(p)))
        self.snapshot(p,land,label,n,n);return p,land

    def hold(self,p,land,T,seconds,label):
        t=time.monotonic();p=self.apply(p,land,T,seconds,label);self.cost['holds']+=time.monotonic()-t
        self.trace.append(dict(segment=label,T_C=T,seconds=seconds,state=p));self.snapshot(p,land,label,1,1);return p

    def snapshot(self,p,land,label,k,n):
        self.current=dict(segment=label,substep=k,substeps=n,T_C=land['identity']['T_C'],B_crystal_T=land['identity']['B_crystal_T'],p=p,minima=land['minima'],channels=land['channels'],steps=self.steps,readings=getattr(self,'readings',[]),events=self.events,elapsed_s=time.monotonic()-self.start)
        if self.checkpoint:self.checkpoint(self.current)

    def read(self,p,land,label):
        t=time.monotonic();M=1e-18*land['identity']['Ms']*(self.R@(p@np.array([q['m'] for q in land['minima']])))
        Y=M/S;self.cost['readings']+=time.monotonic()-t;return dict(label=label,M_Am2=M,Y=Y,p=p,temperature_C=20.)

    def run(self,protocol,prefix=None,on_prefix=None):
        zero=np.zeros(3);B=np.asarray(self.protocols['event']['field_T'])
        self.prefix_reused=prefix is not None
        if prefix is None:
            zero=np.zeros(3);B=np.asarray(self.protocols['event']['field_T']);T=20.;land=self.landscape(T,zero)
            p=core.prepare(self.pm(T,zero),land,self.R,self.Rspec);checked(p,'preparation');self.readings=[]
            self.trace.append(dict(segment='preparation',state=p,R=self.R,rule='maximum_sample_positive_z_exact_ties_equal'))
            p,land=self.ramp(p,land,20.,zero,20.,B,1.,'field_on')
            p,land=self.ramp(p,land,20.,B,200.,B,540.,'event_heating')
            p=self.hold(p,land,200.,1e6,'event_hold')
            p,land=self.ramp(p,land,200.,B,20.,B,540.,'event_cooling')
            p,land=self.ramp(p,land,20.,B,20.,zero,1.,'field_off')
            p=self.hold(p,land,20.,120.,'event_wait');self.readings.append(self.read(p,land,'NRM'))
            if on_prefix:
                fields=['cache','operators','events','trace','readings','critical_catalog','event_solutions','max_mass_error','min_probability','min_barrier_kBT','field_connection_checks']
                on_prefix(copy.deepcopy(dict(p=p,land=land,fields={k:getattr(self,k) for k in fields},key=dict(point=self.point_id,level=self.level))))
        else:
            data=copy.deepcopy(prefix)
            if core.plain(data['key'])!=core.plain(dict(point=self.point_id,level=self.level)):raise core.Unresolved('PREFIX_IDENTITY_MISMATCH')
            p=data['p'];land=data['land']
            for k,v in data['fields'].items():setattr(self,k,v)
            self.trace.append(dict(segment='prefix_reused',source='identical_post_event_state_before_laboratory',forked_probability=p.copy()))
        for plateau in self.protocols['laboratory'][protocol]['plateaus_C']:
            p,land=self.ramp(p,land,20.,zero,float(plateau),zero,(plateau-20)*3.,'lab_heat_'+str(plateau))
            p=self.hold(p,land,float(plateau),1200.,'lab_hold_'+str(plateau))
            p,land=self.ramp(p,land,float(plateau),zero,20.,zero,(plateau-20)*3.,'lab_cool_'+str(plateau))
            p=self.hold(p,land,20.,120.,'lab_wait_'+str(plateau));self.readings.append(self.read(p,land,str(plateau)))
        return self.result('COMPUTED_VALID')

    def result(self,status,reason=None):
        return dict(status=status,reason=reason,point_identity=self.point_id,resolution=self.level,readings=getattr(self,'readings',[]),
            trace=self.trace,events=self.events,checkpoint=getattr(self,'current',None),diagnostic=getattr(self,'diagnostic',None),
            wall_s=time.monotonic()-self.start,cost_s=self.cost,advance_operators=self.steps,
            cached_landscapes=len(self.cache),cached_operators=len(self.operators),maximum_mass_error=self.max_mass_error,
            minimum_probability=self.min_probability,min_barrier_kBT=self.min_barrier_kBT if math.isfinite(self.min_barrier_kBT) else None,
            S_Am2=S,sigma_Y_min=.0025/math.sqrt(2),noise_generated=False,
            support='conditional inherited SD envelope; bulk rates, intrawell relaxation and low barriers not experimentally qualified',
            channel_continuation_limit='unobserved heteroclinic saddle-connection changes are not globally excluded by finite stationary continuation',
            alignment_failures=self.alignment_failures,alignment_calls=self.alignment_calls,maximum_alignment_distance=self.maximum_alignment_distance,field_connection_checks=self.field_connection_checks,geometric_continuation_steps=getattr(self,'geometric_continuation_steps',0),
            population_curve=False,prefix_reused=getattr(self,'prefix_reused',False),critical_catalog=self.critical_catalog)
