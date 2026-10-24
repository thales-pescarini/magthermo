"""Verified-method axial tensor enumerator and finite local E1 kernels.

c is an unoriented symmetry axis in crystal coordinates; R maps crystal
vectors to the sample. For oblates c is the SHORT axis. No triaxial model.
"""
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,math,json,collections
import numpy as np
import mpmath as mp
from scipy.optimize import linear_sum_assignment,root
from environment import base_root
import gate1_verified_methods as gate

ROOT=Path(__file__).resolve().parents[1]
BASE=base_root()
sys.path.insert(0,str(BASE/'legacy_discrete/src'))
import s5_model as legacy
from population_protocol import haar_orientations,high_precision_row

COUNTS=collections.Counter()
KB=1.380649e-23
NU0=1e9

class Unresolved(RuntimeError):pass

class PointModel:
    def __init__(self,s,c,T,B=(0.,0.,0.),TM=0):
        self.s=float(s);self.c=np.array(c,float,copy=True);self.c/=np.linalg.norm(self.c)
        self.T=float(T);self.B=np.asarray(B,float);self.TM=TM
        if abs(self.s)>math.log(2.5)+1e-14:raise Unresolved('SHAPE_OUTSIDE_E1_AXIAL_DOMAIN')
        legacy.configure_material(TMx_percent=TM)
        _,self.k1,self.k2,self.Ms=legacy.material(self.T)
        if self.s>=0:
            N=np.asarray(legacy.source.demag_factors(math.exp(self.s),1.),float);parallel=N[0];perp=N[1]
        else:
            N=np.asarray(legacy.source.demag_factors(1.,math.exp(-self.s)),float);parallel=N[2];perp=N[0]
        self.N=perp*np.eye(3)+(parallel-perp)*np.outer(self.c,self.c)
        self.D=.5*legacy.MU0*self.Ms**2*self.N
        self.h=self.Ms*self.B
        self.arrays=(self.k1,self.k2,self.D,self.h,np.eye(3))
        self.escale=abs(self.k1)+float(np.max(np.diag(self.D)))
        COUNTS['point_models']+=1

    def np_arrays(self):return self.arrays

    def energy_np(self,u):
        COUNTS['energy_evaluations']+=1
        q=u*u
        return self.k1*(q[0]*q[1]+q[0]*q[2]+q[1]*q[2])+self.k2*np.prod(q)+u@self.D@u-self.h@u

    def grad_np(self,u):
        COUNTS['gradient_evaluations']+=1
        x,y,z=u;xx,yy,zz=x*x,y*y,z*z
        return np.array([2*x*(self.k1*(yy+zz)+self.k2*yy*zz),
            2*y*(self.k1*(xx+zz)+self.k2*xx*zz),
            2*z*(self.k1*(xx+yy)+self.k2*xx*yy)])+2*self.D@u-self.h

    def identity(self):return dict(s=self.s,c=self.c.tolist(),T_C=self.T,B_crystal_T=self.B.tolist(),TM=self.TM,
        N=self.N.tolist(),K1=self.k1,K2=self.k2,Ms=self.Ms,source=legacy.filehash(legacy.source.__file__))


def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def landscape(pm,seed_levels=(6,12),with_connections=True):
    censuses=[]
    for n in seed_levels:
        COUNTS['cube_sphere_seeds']+=6*n*n
        cen=gate.census(pm,n)
        COUNTS['Newton_seed_failures']+=cen['newton_failures']
        censuses.append(cen)
    result=landscape_from_censuses(pm,censuses,seed_levels,with_connections)
    if result['status']=='NUMERICAL_UNRESOLVED':
        return repair_saved_landscape(pm,result,with_connections)
    return result


def landscape_from_censuses(pm,censuses,seed_levels,with_connections=True):
    valid=gate.same_census(*censuses) and all(x['euler']==2 and x['all_morse'] for x in censuses)
    result=dict(identity=pm.identity(),seed_levels=list(seed_levels),censuses=censuses,
        census_agreement=gate.same_census(*censuses),status='COMPUTED_VALID' if valid else 'NUMERICAL_UNRESOLVED',
        numerical_verification_not_certification=True,channels=[],connections=[])
    if not valid or not with_connections:return result
    points=censuses[-1]['points'];minima=[p for p in points if p['index']==0]
    for p in [x for x in points if x['index']==1]:
        _,H,Z=gate.riem(pm,p['m']);_,V=np.linalg.eigh(H);v=Z@V[:,0]
        branches=[];ends=[]
        for sign in [-1,1]:
            start=p['m']+sign*1e-4*v;start/=np.linalg.norm(start)
            destinations=[];details=[]
            for tol in [1e-9,1e-11]:
                COUNTS['gradient_branches_integrated']+=1
                r=gate.integrate_branch(pm,start,t_max=4000.,rtol=tol,atol=tol*1e-2,escale=pm.escale)
                distances=[float(np.linalg.norm(np.asarray(r['m_end'])-q['m'])) for q in minima]
                k=int(np.argmin(distances));dest=k if distances[k]<1e-6 else None
                destinations.append(dest)
                COUNTS['flow_steps']+=r['n_steps']
                details.append(dict(**r,destination=dest,destination_direction_distance=distances[k]))
            agree=destinations[0]==destinations[1] and destinations[0] is not None
            branches.append(dict(agree=agree,details=details));ends.append(destinations[-1])
        barriers=[float(p['E']-minima[i]['E']) for i in ends] if None not in ends else []
        okay=all(b['agree'] for b in branches) and len(set(ends))==2 and min(barriers,default=-1)>0
        result['connections'].append(dict(saddle=p['m'],branches=branches,valid=okay))
        if okay:result['channels'].append(dict(endpoints=ends,saddle_direction=p['m'],saddle_E=p['E'],barrier_density=barriers))
        else:result['status']='NUMERICAL_UNRESOLVED'
    result['minima']=minima
    result['actual_state_count']=len(minima)
    result['min_barrier_density']=min((min(ch['barrier_density']) for ch in result['channels']),default=None)
    return result


def polish_census(pm,cen):
    """Tighten roots before deduplication, using saved candidates, not new seeds.

    The old absolute residual does not control direction error near small
    curvature. This is one finite corrector, with stricter residual and the
    original 1e-6-radian deduplication. No grain/occupation is deleted.
    """
    scale=pm.escale;points=[];records=[]
    def F(x):
        COUNTS['polish_KKT_residual_evaluations']+=1
        return np.r_[pm.grad_np(x[:3])/scale-x[3]*x[:3],.5*(x[:3]@x[:3]-1)]
    def J(x):
        COUNTS['polish_KKT_Jacobian_evaluations']+=1
        H=gate.hess_amb(pm,x[:3]);A=np.zeros((4,4));A[:3,:3]=H/scale-x[3]*np.eye(3)
        A[:3,3]=-x[:3];A[3,:3]=x[:3];return A
    for i,p in enumerate(cen['points']):
        u=np.asarray(p['m']);initial=np.r_[u,u@pm.grad_np(u)/scale]
        COUNTS['saved_candidates_polished']+=1
        sol=root(F,initial,jac=J,tol=1e-12,options={'maxfev':100})
        residual=float(np.linalg.norm(F(sol.x)))
        v=sol.x[:3]/np.linalg.norm(sol.x[:3]);displacement=float(np.linalg.norm(v-u))
        valid=residual<=2e-13 and displacement<.3
        if valid:
            gt,H,Z=gate.riem(pm,v);eig=np.linalg.eigvalsh(H)
            duplicate=next((j for j,q in enumerate(points) if math.acos(max(-1.,min(1.,float(v@q['m']))))<=1e-6),None)
            if duplicate is None:
                points.append(dict(m=v,E=float(pm.energy_np(v)),eig=eig.tolist(),index=int(np.sum(eig<0)),
                    morse=bool(np.all(abs(eig)>1e-8*scale))))
        else:duplicate=None
        records.append(dict(original_candidate=i,KKT_scaled_residual=residual,
            direction_change=displacement,accepted=valid,duplicate_of=duplicate,
            scipy_success=bool(sol.success),message=str(sol.message)))
    counts={i:sum(p['index']==i for p in points) for i in [0,1,2]}
    return dict(points=points,counts=counts,euler=counts[0]-counts[1]+counts[2],
        all_morse=all(p['morse'] for p in points),newton_failures=cen['newton_failures'],seeds=cen['seeds'],
        polish=records,all_candidate_polishes_resolved=all(r['accepted'] for r in records))


def repair_saved_landscape(pm,previous,with_connections=True):
    censuses=[polish_census(pm,c) for c in previous['censuses']]
    result=landscape_from_censuses(pm,censuses,previous['seed_levels'],with_connections)
    if not all(c['all_candidate_polishes_resolved'] for c in censuses):result['status']='NUMERICAL_UNRESOLVED'
    result['material_bug_correction']=dict(id='E1_BUG1_ROOT_ACCURACY_BEFORE_DEDUP',
        only_saved_candidates=True,new_seeds=0,dedup_angle_unchanged_rad=1e-6,
        residual_stricter_than_original=True,previous_status=previous['status'],
        occupations_or_grains_removed=False,automatic_second_correction=False)
    return result


def prepare(pm,land,R,spec=None):
    minima=land['minima'];u=np.array([p['m'] for p in minima]);projection=np.asarray(R)[2]@u.T
    selected=np.flatnonzero(projection.max()-projection<=1e-10)
    if len(selected)>1:
        COUNTS['MP_preparation_tie_refinements']+=1
        with mp.workdps(70):
            row=high_precision_row(R,spec)
            k1,k2=mp.mpf(pm.k1),mp.mpf(pm.k2);D=mp.matrix(pm.D.tolist());h=list(map(mp.mpf,pm.h))
            def F(x,y,z,lag):
                v=[x,y,z];sq=[a*a for a in v]
                g=[2*v[i]*(k1*(sq[(i+1)%3]+sq[(i+2)%3])+k2*sq[(i+1)%3]*sq[(i+2)%3])+2*sum(D[i,j]*v[j] for j in range(3))-h[i]-lag*v[i] for i in range(3)]
                return (*g,sum(a*a for a in v)-1)
            vals=[]
            for p in minima:
                v=p['m'];g=pm.grad_np(v)
                solution=mp.findroot(F,tuple(map(float,v))+(float(v@g),),tol=mp.mpf('1e-55'),maxsteps=60)
                vals.append(mp.fsum(row[i]*solution[i] for i in range(3)))
            top=max(vals);selected=np.array([i for i,v in enumerate(vals) if abs(top-v)<mp.mpf('1e-50')])
    p=np.zeros(len(u));p[selected]=1/len(selected)
    return p


def propagate(land,T,d,h,p0):
    if land['status']!='COMPUTED_VALID':raise Unresolved('UNQUALIFIED_LANDSCAPE_NO_PROPAGATION')
    energies=[p['E'] for p in land['minima']];N=len(energies);V=math.pi/6*(d*1e-9)**3
    lrs=[]
    for ch in land['channels']:
        for i in ch['endpoints']:
            barrier=ch['saddle_E']-energies[i]
            if barrier<=0:raise Unresolved('NONPOSITIVE_BARRIER_NO_CLIPPING')
            lrs.append(math.log(NU0)-V*barrier/(KB*(T+273.15)))
    span=(max(lrs)-min(lrs))/math.log(10)
    dps=int(60+1.3*span)
    if dps>2048:raise Unresolved('PRECISION_REQUIRED_ABOVE_FINITE_2048_DPS_RESOURCE_CAP')
    with mp.workdps(dps):
        e=[mp.mpf(v) for v in energies];Vm=mp.mpf(V);kT=mp.mpf(KB)*mp.mpf(T+273.15)
        Q=mp.zeros(N);S=mp.zeros(N)
        for ch in land['channels']:
            a,b=ch['endpoints'];es=mp.mpf(ch['saddle_E'])
            Q[b,a]+=mp.mpf(NU0)*mp.exp(-Vm*(es-e[a])/kT)
            Q[a,b]+=mp.mpf(NU0)*mp.exp(-Vm*(es-e[b])/kT)
            symmetric=mp.mpf(NU0)*mp.exp(-Vm*(es-(e[a]+e[b])/2)/kT)
            S[a,b]+=symmetric;S[b,a]+=symmetric
        for j in range(N):Q[j,j]=-mp.fsum(Q[i,j] for i in range(N) if i!=j);S[j,j]=Q[j,j]
        ex=[-Vm*v/kT for v in e];maximum=max(ex);z=mp.fsum(mp.exp(a-maximum) for a in ex)
        sq=[mp.sqrt(mp.exp(a-maximum)/z) for a in ex]
        eigen,vectors=mp.eigsy(S)
        initial=[mp.mpf(float(p0[i]))/sq[i] for i in range(N)]
        amplitude=[mp.fsum(vectors[i,k]*initial[i] for i in range(N)) for k in range(N)]
        output=[sq[i]*mp.fsum(vectors[i,k]*mp.exp(eigen[k]*mp.mpf(h))*amplitude[k] for k in range(N)) for i in range(N)]
        mass=float(abs(mp.fsum(output)-1));minimum=float(min(output))
        COUNTS['MP_spectral_propagations']+=1;COUNTS['MP_modes_retained']+=N
        if mass>1e-12 or minimum<-1e-12:raise Unresolved('MASS_OR_POSITIVITY_UNRESOLVED_NO_RESET')
    return np.array(list(map(float,output))),dict(dps=dps,log10_rate_span=span,mass_residual=mass,
        minimum_probability=minimum,n_modes=N,min_barrier_over_kBT=V*land['min_barrier_density']/(KB*(T+273.15)))


def perturbative(zero,pm,R,B_sample):
    if zero['status']!='COMPUTED_VALID':raise Unresolved('ZERO_FIELD_LANDSCAPE_UNRESOLVED')
    B=np.asarray(R).T@np.asarray(B_sample);Ms=pm.Ms
    ratio=float(np.linalg.norm(B)*Ms/(2*abs(pm.k1)))
    from copy import deepcopy
    result=deepcopy(zero)
    curv=min(min(abs(np.asarray(p['eig']))) for p in zero['censuses'][-1]['points'])
    # This curvature guard requests exact calculation; it is not an error bound.
    if ratio>.05 or Ms*np.linalg.norm(B)/curv>.05:raise Unresolved('EXACT_FIELD_REQUIRED_BY_FIELD_OR_CURVATURE_GUARD')
    for point in result['minima']:point['E']-=Ms*B@point['m']
    for ch in result['channels']:
        ch['saddle_E']-=Ms*B@ch['saddle_direction']
        ch['barrier_density']=[ch['saddle_E']-result['minima'][i]['E'] for i in ch['endpoints']]
        if min(ch['barrier_density'])<=0:raise Unresolved('EXACT_FIELD_REQUIRED_NONPOSITIVE_PERTURBED_BARRIER')
    result['min_barrier_density']=min(min(ch['barrier_density']) for ch in result['channels'])
    result['perturbation']=dict(field_ratio=ratio,curvature_ratio=Ms*np.linalg.norm(B)/curv,
        direction_change_ignored_first_order=True,observable_error_bound=None)
    return result


def explicit_event_map(old_minima,new_minima,kind,groups=None,disappearing=None,born=None,flow_destinations=None,curvature_bracket=None,symmetry_verified=False):
    """Conservative maps require a located event and explicit branch evidence.

    For generic broken symmetry use continuation plus isolated saddle-node
    birth/death, never apply an equal split without exact reflection evidence.
    """
    if not curvature_bracket:raise Unresolved('EVENT_NOT_LOCATED_BY_CURVATURE')
    a=np.asarray(old_minima);b=np.asarray(new_minima);P=np.zeros((len(b),len(a)))
    if kind in ['pitchfork_merge','pitchfork_reverse']:
        if not symmetry_verified or groups is None:raise Unresolved('PITCHFORK_BRANCH_SYMMETRY_NOT_VERIFIED')
        for old_ids,new_ids in groups:
            if len(new_ids)>1 and kind!='pitchfork_reverse':raise Unresolved('UNCLASSIFIED_EVENT_BRANCHING')
            for i in old_ids:
                for j in new_ids:P[j,i]=1/len(new_ids)
    elif kind in ['saddle_node_death','isolated_birth']:
        disappeared=set(disappearing or []);newborn=set(born or [])
        live_old=[i for i in range(len(a)) if i not in disappeared];live_new=[j for j in range(len(b)) if j not in newborn]
        if len(live_old)!=len(live_new):raise Unresolved('EVENT_IDENTITY_REMAINS_AMBIGUOUS')
        rows,cols=linear_sum_assignment(1-a[live_old]@b[live_new].T)
        for i,j in zip(rows,cols):P[live_new[j],live_old[i]]=1.
        for i in disappeared:
            destinations=(flow_destinations or {}).get(i)
            if not destinations or destinations[0]!=destinations[1]:raise Unresolved('TWO_TOLERANCE_EVENT_FLOW_DESTINATION_REQUIRED')
            P[destinations[0],i]=1.
    else:raise Unresolved('EVENT_KIND_NOT_IMPLEMENTED')
    if np.max(abs(P.sum(axis=0)-1))>1e-14 or np.min(P)<0:raise Unresolved('EVENT_MAP_DOES_NOT_CONSERVE_MASS')
    return P
