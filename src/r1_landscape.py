"""Finite axial-energy root enumeration, continuation, and bounded descent.

No universal angular solver is introduced. The stationary energy and all-mode
propagator are inherited from E1. Numerical root seeds and flow stopping are
the repairs authorized in E1-R1.
"""
import itertools,math,time,hashlib,json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root,linear_sum_assignment
import e1_core as core
import g1_geometry as geom
gate=core.gate

def symmetry_seeds():
    out=[]
    for a in itertools.product([-1.,0.,1.],repeat=3):
        if any(a):v=np.array(a);out.append(v/np.linalg.norm(v))
    return np.array(out)

def scale(pm):
    return max(1.,abs(pm.k1),abs(pm.k2),float(np.linalg.norm(pm.D-np.trace(pm.D)/3*np.eye(3))),float(np.linalg.norm(pm.h)))

def batch_quantities(pm,u):
    q=u*u;k1=pm.k1;k2=pm.k2
    D=pm.D-np.trace(pm.D)/3*np.eye(3)
    g=2*u*(k1*(q.sum(axis=1)[:,None]-q)+k2*np.stack([q[:,1]*q[:,2],q[:,0]*q[:,2],q[:,0]*q[:,1]],axis=1))+2*u@D-pm.h
    H=np.zeros((len(u),3,3))
    for i in range(3):
        j,k=[v for v in range(3) if v!=i]
        H[:,i,i]=2*(k1*(q[:,j]+q[:,k])+k2*q[:,j]*q[:,k])
        for j in range(3):
            if i!=j:H[:,i,j]=4*u[:,i]*u[:,j]*(k1+k2*q[:,3-i-j])
    return g,H+2*D

def _batch_newton_r2(pm,seeds,iters=100):
    u=np.array(seeds,float,copy=True);u/=np.linalg.norm(u,axis=1)[:,None]
    alive=np.ones(len(u),bool);done=np.zeros(len(u),bool);sc=scale(pm)
    for _ in range(iters):
        ids=np.flatnonzero(alive&~done)
        if not len(ids):break
        v=u[ids];g,H=batch_quantities(pm,v);lag=np.einsum('ij,ij->i',v,g)
        gt=g-lag[:,None]*v
        small=np.linalg.norm(gt,axis=1)<=2e-13*sc
        done[ids[small]]=True
        active=~small
        if not np.any(active):continue
        ia=ids[active];v=v[active];gt=gt[active];H=H[active];lag=lag[active]
        normal=np.einsum('ij,ik->ijk',v,v);P=np.eye(3)-normal
        A=P@(H-lag[:,None,None]*np.eye(3))@P+sc*normal
        try:step=np.linalg.solve(A,-gt[...,None])[...,0]
        except np.linalg.LinAlgError:
            step=np.array([np.linalg.lstsq(a,-b,rcond=1e-14)[0] for a,b in zip(A,gt)])
        step-=np.einsum('ij,ij->i',step,v)[:,None]*v
        norm=np.linalg.norm(step,axis=1);step*=np.minimum(1.,.3/np.maximum(norm,1e-300))[:,None]
        angle=np.linalg.norm(step,axis=1)
        update=np.cos(angle)[:,None]*v+np.sinc(angle/np.pi)[:,None]*step
        u[ia]=update/np.linalg.norm(update,axis=1)[:,None]
    g,_=batch_quantities(pm,u);gt=g-np.einsum('ij,ij->i',u,g)[:,None]*u
    done=np.linalg.norm(gt,axis=1)<=2e-11*sc
    core.COUNTS['vector_Newton_seeds']+=len(u)
    return u,done

def _point_r2(pm,u):
    gt,H,_=gate.riem(pm,u);eig=np.linalg.eigvalsh(H)
    return dict(m=np.asarray(u),E=float(pm.energy_np(u)),eig=eig,index=int(np.sum(eig<0)),
        morse=bool(np.min(abs(eig))>1e-9*scale(pm)),gradient_residual=float(np.linalg.norm(gt)))

def _census_r2(pm,n,extra=True):
    seeds=gate.cube_sphere_seeds(n)
    if extra:seeds=np.r_[seeds,symmetry_seeds()]
    u,ok=batch_newton(pm,seeds);unique=[]
    for v in u[ok]:
        if not unique or np.min(np.linalg.norm(np.array(unique)-v,axis=1))>1e-6:unique.append(v)
    # One tight KKT polish followed by the same angular deduplication.
    raw=dict(points=[point(pm,v) for v in unique],newton_failures=int((~ok).sum()),seeds=len(seeds))
    polished=core.polish_census(pm,raw)
    polished['extra_high_symmetry_seeds']=26 if extra else 0
    return polished

def batch_newton(pm,seeds,iters=100):
    u,ok=_batch_newton_r2(pm,seeds,iters)
    if geom.applicable(pm):
        for j in np.flatnonzero(ok):
            try:u[j]=geom.canonical_from_candidate(pm,u[j])['m']
            except core.Unresolved:ok[j]=False
    return u,ok

def point(pm,u):
    return geom.refine_if_needed(pm,_point_r2(pm,u))

def census(pm,n,extra=True):
    raw=_census_r2(pm,n,extra)
    if not geom.precision_supported(pm):return raw
    fixed=[]
    for p in raw['points']:
        q=geom.refine_if_needed(pm,p);fixed.append(q)
    if geom.applicable(pm):
        reference=geom.algebra(pm)
        if len(fixed)!=len(reference) or {p['analytic_branch'] for p in fixed}!={p['analytic_branch'] for p in reference}:
            raise core.Unresolved('G1_SEEDED_CENSUS_INCOMPLETE_VERSUS_ALGEBRA')
    counts={k:sum(p['index']==k for p in fixed) for k in range(3)}
    return dict(raw,points=fixed,counts=counts,euler=counts[0]-counts[1]+counts[2],all_morse=all(p['morse'] for p in fixed),
        geometric_correction='same seeded census candidates, corrected before acceptance; G1 exact algebra only when conditions hold')

def _continuous_points_reference(pm,old):
    if geom.applicable(pm) and all(p.get('analytic_branch') for p in old):
        reference=geom.algebra(pm);bybranch={p['analytic_branch']:p for p in reference}
        if len(reference)!=len(old) or set(bybranch)!={p['analytic_branch'] for p in old}:
            return [dict(p) for p in reference],False
        pts=[dict(bybranch[p['analytic_branch']],point_id=p.get('point_id',bybranch[p['analytic_branch']]['point_id'])) for p in old]
        valid=all(a['index']==b['index'] and b['morse'] and b['angular_correction_eta']<=1e-9 for a,b in zip(old,pts))
        travel=max(np.linalg.norm(a['m']-b['m']) for a,b in zip(old,pts))
        return pts,bool(valid and travel<.15)
    u,ok=batch_newton(pm,[p['m'] for p in old])
    if not np.all(ok):
        # These are the last valid roots, explicitly predictions only at the
        # requested new parameters. Failed Newton candidates are not evaluated.
        return [dict(p,continuation_prediction_only=True) for p in old],False
    try:pts=[dict(point(pm,v),point_id=a.get('point_id',f'cp{k:03d}')) for k,(a,v) in enumerate(zip(old,u))]
    except core.Unresolved:
        return [_point_r2(pm,v) for v in u],False
    u=np.array([p['m'] for p in pts])
    unique=all(np.linalg.norm(u[i]-u[j])>1e-6 for i in range(len(u)) for j in range(i))
    same=all(a['index']==b['index'] and b['morse'] for a,b in zip(old,pts))
    travel=float(np.max(np.linalg.norm(u-np.array([p['m'] for p in old]),axis=1)))
    return pts,bool(np.all(ok) and unique and same and travel<.15)


def continuous_points(pm,old):
    """Reference continuation, then bounded same-branch MP KKT recovery.

    An unsuccessful binary64 Newton iterate is never treated as a root. The
    fallback solves the same stationary equations from every previous root;
    indices, pairwise uniqueness and travel must all survive. A lost/born
    state still requests the approved event transport, with no reset/split.
    """
    points,valid=_continuous_points_reference(pm,old)
    if valid or geom.applicable(pm):return points,valid
    previous=np.array([p['m'] for p in old]);corrected=[]
    core.COUNTS['E_MP_continuation_fallback_calls']+=1
    for i,p in enumerate(old):
        separation=min((np.linalg.norm(previous[i]-v) for j,v in enumerate(previous) if j!=i),default=2.)
        maximum_move=min(.1,.25*separation)
        for dps in [85,120]:
            try:
                q=geom.refine_kkt(pm,p['m'],dps,maximum_move=maximum_move)
                if q['index']!=p['index'] or not q['morse']:
                    raise core.Unresolved('E_MP_CONTINUATION_INDEX_CHANGED_REQUIRES_EVENT')
                q=dict(q,point_id=p.get('point_id',f'cp{i:03d}'),
                       E_continuation_corrector='same polynomial KKT; previous-root seed; bounded by quarter root separation',
                       E_maximum_move=maximum_move)
                corrected.append(q)
                break
            except (ValueError,ZeroDivisionError,core.Unresolved):
                pass
        else:
            core.COUNTS['E_MP_continuation_fallback_unresolved']+=1
            return points,False
    U=np.array([q['m'] for q in corrected])
    unique=all(np.linalg.norm(U[i]-U[j])>1e-6 for i in range(len(U)) for j in range(i))
    travel=float(np.max(np.linalg.norm(U-previous,axis=1)))
    accepted=unique and travel<.15
    core.COUNTS['E_MP_continuation_fallback_accepted']+=int(accepted)
    return corrected,bool(accepted)

def _flow_r3(pm,start,minima,rtol=1e-9):
    """Same negative-gradient path, Radau for soft/stiff curvature.

    The positive scalar rescaling is constant. Horizons are cumulative and the
    final state of each horizon is the initial state of the next, never a new
    launch from the saddle. A local convex-ball test permits early termination.
    Its conservative Hessian Lipschitz bound is finite on this cubic polynomial
    sphere. This is local basin evidence, not a global completeness proof.
    """
    U=np.array([p['m'] for p in minima]);sc=scale(pm)
    ev=np.array([min(p['eig']) for p in minima])
    # Overestimate of third derivative plus moving tangent-frame terms.
    L=200*(abs(pm.k1)+abs(pm.k2)+np.linalg.norm(pm.D-np.trace(pm.D)/3*np.eye(3))+np.linalg.norm(pm.h)+1.)
    sep=np.array([min([np.linalg.norm(a-b) for j,b in enumerate(U) if j!=i],default=2.) for i,a in enumerate(U)])
    radius=np.minimum(2e-7,np.minimum(ev/(8*L),sep/10))
    def rhs(t,y):return gate.flow_rhs(pm,y,sc)
    def arrived(t,y):
        v=y/np.linalg.norm(y)
        return float(np.min(np.linalg.norm(U-v,axis=1)-radius))
    arrived.terminal=True;arrived.direction=-1
    y=np.array(start,float);y/=np.linalg.norm(y);previous=0.;allE=[];normerr=0.;horizons=[];steps=0
    for horizon in [4000.,32000.,256000.]:
        sol=solve_ivp(rhs,(previous,horizon),y,method='Radau',rtol=rtol,atol=rtol*.01,events=arrived)
        Y=sol.y.T;norms=np.linalg.norm(Y,axis=1);normerr=max(normerr,float(np.max(abs(norms-1))))
        Y=Y/norms[:,None];allE.extend(float(pm.energy_np(v)) for v in Y);steps+=len(Y)
        y=sol.y[:,-1];previous=float(sol.t[-1]);v=y/np.linalg.norm(y)
        distances=np.linalg.norm(U-v,axis=1);j=int(np.argmin(distances));gt,H,_=gate.riem(pm,v)
        local=bool(distances[j]<=radius[j]*1.01 and min(np.linalg.eigvalsh(H))>0)
        horizons.append(dict(horizon=horizon,s_end=previous,distance=float(distances[j]),gradient_residual=float(np.linalg.norm(gt)),local_convex_ball_entered=local,solver_success=bool(sol.success)))
        if not sol.success:break
        if local:break
    v=y/np.linalg.norm(y);distances=np.linalg.norm(U-v,axis=1);j=int(np.argmin(distances));gt,H,_=gate.riem(pm,v)
    residual=float(np.linalg.norm(gt));curv=float(min(np.linalg.eigvalsh(H)))
    dest=j if distances[j]<1e-6 and curv>0 else None
    E=np.array(allE);increase=float(max(np.diff(E),default=0.))
    monotone=increase<=1e-9*max(1.,abs(E[0]))
    core.COUNTS['bounded_gradient_flows']+=1;core.COUNTS['flow_accepted_points']+=steps
    return dict(destination=dest,m_end=v,destination_direction_distance=float(distances[j]),
        tangential_gradient_end=residual,positive_curvature_end=curv,n_steps=steps,
        s_end=previous,rtol=rtol,atol=rtol*.01,energy_start=float(E[0]),energy_end=float(E[-1]),
        energy_max_increase=increase,energy_monotone_decreasing=bool(monotone),max_norm_deviation=normerr,
        horizons=horizons,local_ball_radius=float(radius[j]),Hessian_Lipschitz_overestimate=float(L),
        termination='LOCAL_CONVEX_BALL' if horizons[-1]['local_convex_ball_entered'] else 'HORIZON_OR_SOLVER_LIMIT',
        trajectories_preserved_by_positive_scalar_rescaling=True,scale=sc)

def frontier_connection(pm,p,points):
    """Exact invariant-plane test for c=[110], B=0; no probabilistic split."""
    D=pm.D;tol=1e-10
    symmetric=(np.linalg.norm(pm.h)==0 and abs(D[0,0]-D[1,1])<tol
        and abs(D[0,2])+abs(D[1,2])<tol and abs(D[2,2]-(D[0,0]-D[0,1]))<tol)
    u=np.asarray(p['m'])
    if not symmetric or abs(u[0]+u[1])>1e-8 or abs(u[2])>1e-8:return None
    factor=[pm.k1/2+pm.k2/4,pm.k1/2+pm.k2/6]
    if max(factor)>=0:return None
    normal=np.array([1.,1.,0.])/math.sqrt(2);_,H,Z=gate.riem(pm,u)
    upper_normal=float((Z.T@normal)@H@(Z.T@normal))
    upper_plane=float(np.array([0.,0.,1.])@Z@H@Z.T@np.array([0.,0.,1.]))
    targets=[]
    for sign in [-1.,1.]:
        target=np.array([math.copysign(1.,u[0]),-math.copysign(1.,u[0]),sign])/math.sqrt(3)
        q=min(points,key=lambda q:np.linalg.norm(q['m']-target))
        _,h,z=gate.riem(pm,q['m']);v=np.cross(normal,q['m']);v/=np.linalg.norm(v)
        transverse=float((z.T@normal)@h@(z.T@normal));inplane=float((z.T@v)@h@(z.T@v))
        if q['index']!=1 or np.linalg.norm(q['m']-target)>1e-7 or transverse>=0 or inplane<=0:return None
        targets.append(dict(point_id=q['point_id'],direction=q['m'],transverse_curvature=transverse,in_plane_curvature=inplane))
    if upper_normal<=0 or upper_plane>=0:return None
    return dict(classification='SADDLE_TO_SADDLE',saddle_id=p['point_id'],saddle=p['m'],
        targets=targets,plane='u_x+u_y=0',plane_invariance_verified=True,
        energy_restriction='K1*(1+2*w-3*w*w)/4+K2*w*(1-w)**2/4+constant',
        derivative='(1-3*w)*(K1/2+K2*(1-w)/4)',derivative_factor_endpoints=factor,
        upper_transverse_curvature=upper_normal,upper_in_plane_curvature=upper_plane,
        boundary_connection_no_autonomous_direct_channel=True,valid=True)

def attach_ids(points):
    for k,p in enumerate(points):p.setdefault('point_id',f'cp{k:03d}')
    return points

def connect(pm,points):
    attach_ids(points)
    minima=[p for p in points if p['index']==0];saddles=[p for p in points if p['index']==1]
    if not minima:return [],[],[],False
    channels=[];details=[];valid=True
    for p in saddles:
        special=frontier_connection(pm,p,points)
        if special is not None:details.append(special);continue
        _,H,Z=gate.riem(pm,p['m']);_,V=np.linalg.eigh(H);v=Z@V[:,0]
        separation=min(np.linalg.norm(p['m']-q['m']) for q in points if q is not p)
        eps=min(1e-4,.01*separation);ends=[];branches=[]
        for sign in [-1,1]:
            start=p['m']+sign*eps*v;start/=np.linalg.norm(start)
            runs=[flow(pm,start,minima,tol) for tol in [1e-9,1e-11]]
            ok=runs[0]['destination'] is not None and runs[0]['destination']==runs[1]['destination'] and all(r['energy_monotone_decreasing'] for r in runs)
            ends.append(runs[-1]['destination']);branches.append(dict(agree=ok,details=runs))
        barriers=[float(p['E']-minima[i]['E']) for i in ends] if None not in ends else []
        okay=all(b['agree'] for b in branches) and min(barriers,default=-1)>0
        cls=('DIRECT_MINIMUM_TRANSITION' if len(set(ends))==2 else 'SAME_MINIMUM_RETURN') if okay else 'UNRESOLVED'
        details.append(dict(saddle_id=p['point_id'],saddle=p['m'],classification=cls,branches=branches,endpoints=ends,valid=okay))
        if okay and cls=='DIRECT_MINIMUM_TRANSITION':
            channels.append(dict(saddle_id=p['point_id'],endpoint_point_ids=[minima[i]['point_id'] for i in ends],endpoints=ends,saddle_direction=p['m'],saddle_E=p['E'],barrier_density=barriers))
        valid &= okay
    return minima,channels,details,bool(valid and len(channels)>0)

def from_points(pm,points,old=None):
    """Explicit critical-point and channel identities, including parallel paths."""
    attach_ids(points);counts=[sum(p['index']==i for p in points) for i in [0,1,2]]
    if counts[0]-counts[1]+counts[2]!=2 or not all(p['morse'] for p in points):raise core.Unresolved('NON_MORSE_OR_EULER_CENSUS')
    if old is None:
        minima,channels,details,ok=connect(pm,points)
        if not ok:
            error=core.Unresolved('TWO_TOLERANCE_SADDLE_CLASSIFICATION_UNRESOLVED')
            error.evidence=dict(minima=minima,channels=channels,connections=details,points=points)
            raise error
        source='two_tolerance_flow_and_verified_invariant_geometry';assumption=False
    else:
        minima=[p for p in points if p['index']==0];byid={p['point_id']:p for p in points}
        mind={p['point_id']:i for i,p in enumerate(minima)}
        if set(byid)!=set(p['point_id'] for p in old['points']):raise core.Unresolved('CRITICAL_IDENTITY_CHANGE')
        channels=[]
        for ch in old['channels']:
            sad=byid[ch['saddle_id']];ends=[mind[x] for x in ch['endpoint_point_ids']]
            barriers=[float(sad['E']-minima[i]['E']) for i in ends]
            # Cancellation is resolved downstream from refined roots with MP.
            if min(barriers)<=0 and min(barriers)<-1e-7*max(1.,abs(sad['E'])):raise core.Unresolved('NONPOSITIVE_BARRIER_NO_CLIPPING')
            channels.append(dict(saddle_id=sad['point_id'],endpoint_point_ids=ch['endpoint_point_ids'],endpoints=ends,saddle_direction=sad['m'],saddle_E=sad['E'],barrier_density=barriers))
        details=[dict(saddle_id=x['saddle_id'],classification=x['classification'],valid=x['valid'],classification_source='continued_critical_and_channel_identity') for x in old['connections']]
        source='explicit_stationary_channel_identity_continuation';assumption=True
    result=dict(status='COMPUTED_VALID',identity=pm.identity(),points=points,minima=minima,channels=channels,
        actual_state_count=len(minima),min_barrier_density=min(min(c['barrier_density']) for c in channels),
        connections=details,counts=counts,connection_source=source,
        no_unobserved_heteroclinic_switch_assumption=assumption)
    return canonicalize_rates_geometry(pm,result)

def canonicalize_rates_geometry(pm,land):
    if not geom.precision_supported(pm):return land
    small=min(min(c['barrier_density']) for c in land['channels'])<1e-7*max(1.,max(abs(p['E']) for p in land['minima']))
    if not (small or geom.applicable(pm)):return land
    needed={p['point_id'] for p in land['minima']}|{c['saddle_id'] for c in land['channels']}
    for p in land['points']:
        if p['point_id'] not in needed or p.get('E_mp') is not None:continue
        q=geom.refine_kkt(pm,p['m'],85)
        if q['index']!=p['index'] or np.linalg.norm(q['m']-p['m'])>1e-9:raise core.Unresolved('G1_RATE_GEOMETRY_WAS_NOT_CANONICAL')
        pid=p['point_id'];branch=p.get('analytic_branch');p.update(q);p['point_id']=pid
        if branch:p['analytic_branch']=branch
    import mpmath as mp
    byid={p['point_id']:p for p in land['points']}
    land['minima']=[byid[p['point_id']] for p in land['minima']]
    with mp.workdps(85):
        for c in land['channels']:
            sad=byid[c['saddle_id']];c['saddle_direction']=sad['m'];c['saddle_E']=sad['E'];c['saddle_E_mp']=sad['E_mp']
            values=[mp.mpf(sad['E_mp'])-mp.mpf(land['minima'][i]['E_mp']) for i in c['endpoints']]
            if min(values)<=0:raise core.Unresolved('G1_CANONICAL_NONPOSITIVE_BARRIER')
            c['barrier_density']=[float(v) for v in values]
    land['min_barrier_density']=min(min(c['barrier_density']) for c in land['channels'])
    land['canonical_geometry_and_energy']=True
    return land

def enumerate_land(pm,levels=(6,12)):
    cens=[census(pm,n) for n in levels]
    agree=gate.same_census(*cens)
    result=dict(identity=pm.identity(),seed_levels=list(levels),censuses=cens,census_agreement=agree,status='NUMERICAL_UNRESOLVED')
    if agree and all(c['euler']==2 and c['all_morse'] and c['counts'][0]>0 and c['counts'][2]>0 for c in cens):
        try:result.update(from_points(pm,cens[-1]['points']))
        except core.Unresolved as e:
            result['reason']=str(e)
            result.update(getattr(e,'evidence',{}))
    else:result['reason']='CENSUS_DISAGREEMENT_OR_NON_MORSE'
    return result

def cubic_symmetries(pm,parent,daughters):
    """Verify energy symmetries and transitivity on actual daughter roots."""
    accepted=[]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1.,1.],repeat=3):
            A=np.eye(3)[list(perm)]*np.array(signs)[:,None]
            if np.max(abs(A.T@pm.D@A-pm.D))>1e-9 or np.max(abs(A.T@pm.h-pm.h))>1e-10:continue
            if np.linalg.norm(A@parent-parent)>1e-4:continue
            accepted.append(A)
    U=np.array(daughters)
    transitive=all(any(np.linalg.norm(A@U[0]-v)<2e-5 for A in accepted) for v in U)
    return bool(transitive and len(U)>1),[A.tolist() for A in accepted]


def flow(pm,start,minima,rtol=1e-9):
    begin=time.monotonic();core.COUNTS["B1_flow_calls"]+=1
    try:return _flow_r3(pm,start,minima,rtol)
    finally:core.COUNTS["B1_flow_wall_s"]+=time.monotonic()-begin
