"""Bounded geometric correction for the inherited axial-energy consumer.

Exact axial c100/B0 branches are selected by mathematical conditions, with
no sentinel identifier or prescribed value of s. They are not extrapolated
to other fields or axes. Historical g1 branch labels are identifiers only.
MP coefficients are the exact binary64 energy coefficients, promoted directly;
in particular Delta is subtracted after promotion of Dperp and Dparallel.
"""
from pathlib import Path
import math,itertools,hashlib,json,functools
import numpy as np
import mpmath as mp
import e1_core as core
TARGET=1e-9
SOURCE_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def precision_supported(pm):
    """Accuracy criterion for the inherited axial-energy TM00 consumer."""
    return pm.TM==0 and abs(pm.s)<=math.log(2.5)+1e-14

def applicable(pm):
    return bool(precision_supported(pm) and np.array_equal(abs(pm.c),np.array([1.,0.,0.]))
        and np.array_equal(pm.B,np.zeros(3)) and np.array_equal(pm.h,np.zeros(3))
        and np.array_equal(pm.D,np.diag(np.diag(pm.D))) and pm.D[1,1]==pm.D[2,2]
        and pm.D[1,1]>pm.D[0,0] and pm.k1<0 and pm.k2<0)

def coefficient_key(pm):
    return (SOURCE_SHA,pm.TM,pm.s,tuple(pm.c),pm.T,tuple(pm.B),pm.k1,pm.k2,tuple(pm.D.ravel()),tuple(pm.h))

def coefficients(pm):
    return mp.mpf(pm.k1),mp.mpf(pm.k2),mp.matrix(pm.D.tolist()),mp.matrix(pm.h.tolist())

def axial_A(pm):
    if not applicable(pm):raise core.Unresolved('G1_AXIAL_A_CONDITIONS')
    with mp.workdps(85):return float(mp.mpf(pm.k1)+mp.mpf(float(pm.D[1,1]))-mp.mpf(float(pm.D[0,0])))

def polynomial(k1,k2,D,h,u):
    q=[v*v for v in u]
    E=k1*(q[0]*q[1]+q[0]*q[2]+q[1]*q[2])+k2*q[0]*q[1]*q[2]+(u.T*D*u)[0]-(h.T*u)[0]
    g=mp.matrix([2*u[a]*(k1*(q[(a+1)%3]+q[(a+2)%3])+k2*q[(a+1)%3]*q[(a+2)%3]) for a in range(3)])+2*D*u-h
    H=mp.matrix(3)
    for a in range(3):
        b,c=[x for x in range(3) if x!=a]
        H[a,a]=2*(k1*(q[b]+q[c])+k2*q[b]*q[c])+2*D[a,a]
        for b in range(3):
            if a!=b:H[a,b]=4*u[a]*u[b]*(k1+k2*q[3-a-b])+2*D[a,b]
    return E,g,H

def mp_point(pm,direction,dps,branch=None):
    with mp.workdps(dps):
        k1,k2,D,h=coefficients(pm);u=mp.matrix(direction)
        E,g,H=polynomial(k1,k2,D,h,u);lag=(u.T*g)[0];gt=g-lag*u
        axis=min(range(3),key=lambda a:abs(u[a]));v=mp.matrix([int(a==axis) for a in range(3)])
        v-=u[axis]*u;v/=mp.sqrt((v.T*v)[0])
        w=mp.matrix([u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]])
        Z=mp.matrix([[v[a],w[a]] for a in range(3)]);Ht=Z.T*(H-lag*mp.eye(3))*Z
        gap=mp.sqrt((Ht[0,0]-Ht[1,1])**2+4*Ht[0,1]**2)
        eig=[(Ht[0,0]+Ht[1,1]-gap)/2,(Ht[0,0]+Ht[1,1]+gap)/2]
        residual=mp.sqrt(mp.fsum(x*x for x in gt));normal=abs((u.T*u)[0]-1)
        scale=max(mp.mpf(1),abs(k1),abs(k2),max(abs(v) for v in D))
        morse=min(abs(e) for e in eig)>mp.mpf(10)**(-dps+20)*scale
        eta=mp.sqrt(mp.fsum(x*x for x in mp.lu_solve(Ht,Z.T*gt))) if morse else mp.inf
        f=np.array([float(x) for x in u]);rounding=mp.sqrt(mp.fsum((mp.mpf(float(f[a]))-u[a])**2 for a in range(3)))
        if residual>mp.mpf('1e-50')*scale or normal>mp.mpf('1e-50') or not morse or eta>mp.mpf('1e-9') or rounding>mp.mpf('1e-9'):
            raise core.Unresolved('G1_MP_GEOMETRY_NOT_RESOLVED')
        return dict(m=f,E=float(E),E_mp=mp.nstr(E,dps),eig=np.array([float(x) for x in eig]),
            eig_mp=[mp.nstr(x,dps) for x in eig],index=sum(x<0 for x in eig),morse=True,
            gradient_residual=float(residual),normal_residual=float(normal),angular_correction_eta=float(eta),
            float_rounding_chord=float(rounding),direction_mp=[mp.nstr(x,dps) for x in u],
            geometry_precision=dps,geometry_source='G1_MP_canonical',analytic_branch=branch,
            coefficient_promotion='exact binary64 K1 K2 D h; no material or temperature change',
            geometry_code_sha256=SOURCE_SHA)

def algebra(pm,dps=85):
    if dps not in [85,120]:raise ValueError('TWO_AUTHORIZED_PRECISION_LEVELS_ONLY')
    if not applicable(pm):raise core.Unresolved('G1_ALGEBRA_CONDITIONS_NOT_MET')
    return _algebra_cached(coefficient_key(pm),dps,pm)

class _ModelCacheProxy:
    pass

# The key contains every energy coefficient; pm is used only on a cache miss.
_ALGEBRA={}
def _algebra_cached(key,dps,pm):
    cachekey=(key,dps)
    if cachekey in _ALGEBRA:return _ALGEBRA[cachekey]
    with mp.workdps(dps):
        k1,k2,D,h=coefficients(pm);delta=D[1,1]-D[0,0];A=k1+delta;b=(k2-3*k1)/2;c=-3*k2/4
        if not b>0 or not c>0:raise core.Unresolved('G1_QUADRATIC_BRANCH_ASSUMPTIONS')
        roots=[]
        def add(u,label):
            p=mp_point(pm,u,dps,label);p['point_id']='g1:'+label;roots.append(p)
        for axis in range(3):
            for sign in [-1,1]:add([mp.mpf(sign if a==axis else 0) for a in range(3)],f'axis_{axis}_{sign:+d}')
        for sy,sz in itertools.product([-1,1],repeat=2):add([mp.mpf(0),sy/mp.sqrt(2),sz/mp.sqrt(2)],f'equator_{sy:+d}_{sz:+d}')
        qp=A/(2*k1)
        if 0<qp<1:
            for axis in [1,2]:
                for sx,st in itertools.product([-1,1],repeat=2):
                    u=[sx*mp.sqrt(1-qp),mp.mpf(0),mp.mpf(0)];u[axis]=st*mp.sqrt(qp)
                    add(u,f'plane_{axis}_{sx:+d}_{st:+d}')
        disc=b*b-4*c*A
        if disc>=0:
            q=-2*A/(b+mp.sqrt(disc))
            if 0<q<1:
                for sx,sy,sz in itertools.product([-1,1],repeat=3):add([sx*mp.sqrt(1-q),sy*mp.sqrt(q/2),sz*mp.sqrt(q/2)],f'bulk_{sx:+d}_{sy:+d}_{sz:+d}')
        counts={i:sum(p['index']==i for p in roots) for i in range(3)}
        if counts[0]-counts[1]+counts[2]!=2:raise core.Unresolved('G1_ALGEBRA_EULER_FAILURE')
        if len(_ALGEBRA)>=3000:_ALGEBRA.pop(next(iter(_ALGEBRA)))
        _ALGEBRA[cachekey]=roots
        core.COUNTS['G1_algebraic_censuses']+=1;core.COUNTS['G1_canonical_roots']+=len(roots)
        return roots

def signature(u):
    u=np.asarray(u);return tuple(0 if abs(x)<1e-10 else (1 if x>0 else -1) for x in u)

def canonical_from_candidate(pm,u,maximum_move=1e-4):
    matches=[p for p in algebra(pm) if signature(p['m'])==signature(u)]
    if len(matches)!=1:raise core.Unresolved('G1_BRANCH_SIGNATURE_NOT_UNIQUE')
    p=matches[0];move=float(np.linalg.norm(p['m']-u))
    if move>maximum_move:raise core.Unresolved('G1_CANONICAL_CORRECTION_TOO_LARGE:'+str(move))
    return dict(p,m=np.array(p['m'],copy=True),canonical_correction_from_candidate=move)

def diagonal_plane_candidate(pm, u, dps=85, maximum_move=1e-4):
    """Exact stationary family x=y for a verified uncoupled zero-field plane.

    On m=(sx*sqrt((1-q)/2),sx*sqrt((1-q)/2),sz*sqrt(q)),
    dE/dq=A+B*q+C*q**2. Coefficients are promoted BEFORE subtraction.
    Every returned root still passes the full three-dimensional MP KKT,
    tangent-Hessian, correction and seed-movement guards. No state is added.
    """
    if dps not in [85,120]:raise ValueError('TWO_AUTHORIZED_PRECISION_LEVELS_ONLY')
    u=np.asarray(u,float);D=pm.D;eps=8*np.finfo(float).eps
    applicable=(precision_supported(pm) and np.array_equal(pm.B,np.zeros(3))
        and np.array_equal(pm.h,np.zeros(3)) and np.array_equal(D,D.T)
        and D[0,0]==D[1,1] and D[0,2]==D[1,2]==0.
        and abs(u[0]-u[1])<=eps and abs(u[0])>eps)
    if not applicable:return None
    with mp.workdps(dps):
        k1,k2,dm,h=coefficients(pm)
        A=k1/2+k2/4+dm[2,2]-dm[0,0]-dm[0,1]
        B=-3*k1/2-k2;C=3*k2/4
        sx=mp.mpf(1 if u[0]>0 else -1);sz=mp.mpf(1 if u[2]>0 else -1)
        directions=[]
        if abs(u[2])<=eps:
            directions.append(([sx/mp.sqrt(2),sx/mp.sqrt(2),mp.mpf(0)],'center'))
        else:
            if C==0:
                roots=[] if B==0 else [-A/B]
            else:
                disc=B*B-4*C*A
                if disc<0:return None
                radical=mp.sqrt(disc)
                # One cancellation-free root and its companion.
                denominator=-B-mp.sign(B if B else 1)*radical
                roots=[denominator/(2*C)] if denominator else [mp.mpf(0)]
                if denominator:roots.append(2*A/denominator)
            for i,q in enumerate(roots):
                if 0<q<1:
                    directions.append(([sx*mp.sqrt((1-q)/2),sx*mp.sqrt((1-q)/2),sz*mp.sqrt(q)],f'root_{i}'))
        # The center is an exact stationary root on both sides. If the q>0
        # family has disappeared, correct only a nearby seed to that parent;
        # its full tangent Hessian still determines its current Morse index.
        if not directions:
            directions.append(([sx/mp.sqrt(2),sx/mp.sqrt(2),mp.mpf(0)],'parent_after_disappearance'))
        matches=[]
        for direction,branch in directions:
            point=mp_point(pm,direction,dps,f'diagonal_plane_{branch}_x{int(sx):+d}_z{int(sz):+d}')
            move=float(np.linalg.norm(point['m']-u))
            if move<=maximum_move:matches.append((point,move))
        if len(matches)!=1:raise core.Unresolved('E_DIAGONAL_PLANE_CORRECTION_NOT_UNIQUE_OR_OUTSIDE_BOUND')
        point,move=matches[0]
        point.update(geometry_source='E_diagonal_plane_polynomial_KKT',KKT_seed_direction_change=move,
            reduced_stationary_polynomial=dict(A=mp.nstr(A,dps),B=mp.nstr(B,dps),C=mp.nstr(C,dps)),
            full_three_dimensional_stationary_and_curvature_checks=True)
        core.COUNTS['E_diagonal_plane_corrections']+=1
        return point


def refine_kkt(pm,u,dps=85,maximum_move=1e-4):
    if dps not in [85,120]:raise ValueError('TWO_AUTHORIZED_PRECISION_LEVELS_ONLY')
    core.COUNTS['bounded_KKT_MP_calls_'+str(dps)]+=1
    diagonal=diagonal_plane_candidate(pm,u,dps,maximum_move)
    if diagonal is not None:return diagonal
    with mp.workdps(dps):
        k1,k2,D,h=coefficients(pm);v=mp.matrix([mp.mpf(float(x)) for x in u]);v/=mp.sqrt((v.T*v)[0]);_,g,_=polynomial(k1,k2,D,h,v)
        sc=max(mp.mpf(1),abs(k1),abs(k2),max(abs(x) for x in D),max(abs(x) for x in h))
        def F(x,y,z,lag):
            a=mp.matrix([x,y,z]);_,grad,_=polynomial(k1,k2,D,h,a)
            return tuple(grad[j]/sc-lag*a[j] for j in range(3))+((a.T*a)[0]-1,)
        def J(x,y,z,lag):
            a=mp.matrix([x,y,z]);_,_,H=polynomial(k1,k2,D,h,a)
            out=mp.matrix(4)
            for j in range(3):
                for k in range(3):out[j,k]=H[j,k]/sc-(lag if j==k else 0)
                out[j,3]=-a[j];out[3,j]=2*a[j]
            return out
        # findroot's multidimensional verification squares the residual norm.
        sol=mp.findroot(F,tuple(v)+((v.T*g)[0]/sc,),J=J,tol=mp.mpf(10)**(-2*(dps-20)),maxsteps=100)
        point=mp_point(pm,sol[:3],dps);move=float(np.linalg.norm(point['m']-np.asarray(u)))
        if move>maximum_move:raise core.Unresolved('G1_KKT_CORRECTION_BRANCH_AMBIGUOUS:'+str(move))
        point['KKT_seed_direction_change']=move
        return point

def refine_if_needed(pm,p):
    if not precision_supported(pm):return p
    if applicable(pm):return canonical_from_candidate(pm,p['m'])
    # Only an already stationary invariant-direction candidate is identified
    # with the exact mirror branch. Nearby off-branch roots are not collapsed.
    u=np.asarray(p['m'])
    if mirror_applicable(pm) and abs(u[0]+u[1])<=8*np.finfo(float).eps and abs(u[2])<=8*np.finfo(float).eps:
        with mp.workdps(85):
            a=mp.mpf(1 if u[0]>0 else -1)/mp.sqrt(2)
            out=mp_point(pm,[a,-a,mp.mpf(0)],85,'mirror_antipode_'+str(1 if u[0]>0 else -1))
        out['geometry_source']='B2_exact_mirror_stationary_branch'
        return out
    gt,H,_=core.gate.riem(pm,p['m'])
    try:eta=float(np.linalg.norm(np.linalg.solve(H,gt)))
    except np.linalg.LinAlgError:eta=math.inf
    if eta<=TARGET and min(abs(x) for x in p['eig'])>=.001:
        return dict(p,angular_correction_eta=eta,geometry_source='G1_conditioned_float_corrector')
    for dps in [85,120]:
        try:return refine_kkt(pm,p['m'],dps)
        except (ValueError,ZeroDivisionError,core.Unresolved) as exc:last=exc
    raise core.Unresolved('G1_FINITE_MP_CORRECTOR_EXHAUSTED:'+str(last))

def mirror_applicable(pm):
    D=pm.D
    return bool(precision_supported(pm) and np.array_equal(pm.B,np.zeros(3)) and np.array_equal(pm.h,np.zeros(3))
        and np.array_equal(D,D.T) and D[0,0]==D[1,1] and D[0,2]==D[1,2])

def mirror_hessian(pm,dps=85):
    """Exact promoted binary coefficients; no approximate symmetry imposed."""
    if not mirror_applicable(pm):raise core.Unresolved('B2_MIRROR_CONDITIONS_ABSENT')
    with mp.workdps(dps):
        k1,k2,D,h=coefficients(pm)
        a=-2*k1+4*D[0,1];b=2*mp.sqrt(2)*D[0,2]
        c=k1+k2/2+2*(D[2,2]-D[0,0]+D[0,1])
        return mp.matrix([[a,b],[b,c]])

def mirror_identity_check(pm):
    with mp.workdps(85):
        k1,k2,D,h=coefficients(pm);a=1/mp.sqrt(2);u=mp.matrix([a,-a,0]);v=mp.matrix([a,a,0]);w=mp.matrix([0,0,1])
        E,g,H=polynomial(k1,k2,D,h,u);lag=(u.T*g)[0];Z=mp.matrix([[v[j],w[j]] for j in range(3)])
        general=Z.T*(H-lag*mp.eye(3))*Z;closed=mirror_hessian(pm)
        sc=max(mp.mpf(1),max(abs(x) for x in closed));error=max(abs(x) for x in general-closed)/sc
        residual=mp.sqrt(mp.fsum(x*x for x in g-lag*u))/sc
        if error>mp.mpf('1e-65') or residual>mp.mpf('1e-65'):raise core.Unresolved('B2_MIRROR_IDENTITY_FAILURE')
        eigen=mp.eigsy(closed,eigvals_only=True)
        return dict(relative_Hessian_difference=float(error),scaled_stationary_residual=float(residual),eigenvalues=[float(x) for x in eigen],
            exact_binary_coefficients_promoted_before_operations=True,K2_retained=True,conditions_verified=True)

def locate_mirror_critical(pm_at_fraction,T0,T1):
    """A physical determinant sign bracket proposes a candidate, not a map."""
    pa,pb=pm_at_fraction(0.),pm_at_fraction(1.)
    if not mirror_applicable(pa) or not mirror_applicable(pb):return None
    def determinant(f):
        H=mirror_hessian(pm_at_fraction(float(f)))
        with mp.workdps(85):return H[0,0]*H[1,1]-H[0,1]**2
    lo,hi=0.,1.;fa,fb=determinant(lo),determinant(hi)
    if fa*fb>=0:return None
    initial=[float(fa),float(fb)];history=[]
    for iteration in range(64):
        mid=(lo+hi)/2
        if mid==lo or mid==hi:break
        fm=determinant(mid);history.append(dict(fraction=mid,determinant=float(fm)))
        if fm==0:lo=hi=mid;break
        if fa*fm<0:hi=mid;fb=fm
        else:lo=mid;fa=fm
        if abs(T1-T0)*(hi-lo)<=1e-12:break
    fc=(lo+hi)/2;pm=pm_at_fraction(fc);proof=mirror_identity_check(pm)
    return dict(kind='B2_exact_mirror_tangent_eigenvalue_candidate',fraction=fc,root_direction=np.array([1.,-1.,0.])/math.sqrt(2),
        physical_determinant_initial_bracket=initial,physical_fraction_enclosure=[lo,hi],physical_temperature_width_C=abs(T1-T0)*(hi-lo),
        iterations=len(history),location_history=history,identity_check=proof,
        candidate_only_until_lateral_roots_channels_maps_verified=True)

def zero_field_map(pmleft,pmright,left,right):
    """The simultaneous axial event, with verified parent-specific symmetry."""
    if not applicable(pmleft) or not applicable(pmright):raise core.Unresolved('ZERO_MAP_OUTSIDE_EXACT_SUBCASE')
    A0,A1=axial_A(pmleft),axial_A(pmright)
    forward=A0<0<A1;reverse=A1<0<A0
    if not (forward or reverse):raise core.Unresolved('ZERO_MAP_NO_SIGNED_CRITICAL_BRACKET')
    expected=([8,12,6],[2,4,4]) if forward else ([2,4,4],[8,12,6])
    if (left['counts'],right['counts'])!=expected:raise core.Unresolved('ZERO_MAP_ACTUAL_CENSUS_NOT_SUPPORTED')
    old=left['minima'];new=right['minima'];P=np.zeros((len(new),len(old)));evidence=[]
    group=[np.diag([1.,sy,sz]) for sy,sz in itertools.product([-1,1],repeat=2)]
    for i,parent in enumerate(old):
        sign=1 if parent['m'][0]>0 else -1
        targets=[j for j,p in enumerate(new) if (1 if p['m'][0]>0 else -1)==sign]
        if forward:
            if len(targets)!=1 or not parent['analytic_branch'].startswith('bulk_') or not new[targets[0]]['analytic_branch'].startswith('axis_0_'):
                raise core.Unresolved('ZERO_FORWARD_BRANCH_ASSOCIATION')
            P[targets[0],i]=1.;evidence.append(dict(old=i,kind='four_branch_axial_merge',destinations=targets,sign_x=sign))
        else:
            if len(targets)!=4 or not parent['analytic_branch'].startswith('axis_0_') or not all(new[j]['analytic_branch'].startswith('bulk_') for j in targets):
                raise core.Unresolved('ZERO_REVERSE_FOUR_DAUGHTERS_NOT_DEMONSTRATED')
            pm=pmright
            if not all(np.array_equal(G.T@pm.D@G,pm.D) and np.array_equal(G.T@pm.h,pm.h) and np.array_equal(G@parent['m'],parent['m']) for G in group):
                raise core.Unresolved('ZERO_REVERSE_ENERGY_SYMMETRY_FAILURE')
            U=np.array([new[j]['m'] for j in targets]);orbit=np.array([G@U[0] for G in group])
            distances=np.linalg.norm(orbit[:,None]-U[None,:],axis=2)
            if np.max(np.min(distances,axis=1))>1e-9 or len({int(np.argmin(row)) for row in distances})!=4:
                raise core.Unresolved('ZERO_REVERSE_ORBIT_NOT_TRANSITIVE')
            P[targets,i]=.25;evidence.append(dict(old=i,kind='verified_symmetric_reverse',destinations=targets,sign_x=sign,symmetries=group,orbit_distance_matrix=distances))
    if np.max(abs(P.sum(axis=0)-1))>1e-12 or P.min()<0:raise core.Unresolved('ZERO_MAP_NOT_STOCHASTIC')
    proof=dict(kind='exact_axial_simultaneous_event',A_bracket=[A0,A1],two_tangent_curvatures_vanish_at_A0=True,
        actual_census=[left['counts'],right['counts']],global_equilibrium_imposed=False,all_parent_columns_tested=True)
    return P,evidence,proof
