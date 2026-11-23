"""Magnetite and titanomagnetite properties and the SDCC single-domain energy model.

The solver enumerates zero-field stationary points for axisymmetric grains with
the shape axis parallel to a hard crystal direction. It follows and classifies
the stationary points as the field changes.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys, math, json, hashlib, itertools, warnings
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.optimize import root, minimize, linear_sum_assignment, brentq
from scipy.special import ndtr, ndtri
from scipy.linalg import expm

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'upstream/src'))
from sdcc import energy as source

KB=1.380649e-23
MU0=4*np.pi*1e-7
NU0=1e9
POLICY={'temperature_step_C':2.,'field_step_s':.25,'root_xtol':1e-11,
        'root_scaled_residual_max':2e-9,'root_merge_distance':1e-7,
        'morse_relative_threshold':1e-9,'preparation_projection_tolerance':1e-10,
        'preparation_high_precision_dps':70,'basin_start_angle_rad':.035,
        'basin_angular_residual_max':2e-5,'probability_residual_max':2e-9,
        'scipy_expm_norm_max':2e5,'mpmath_dps':60,
        'source_energy_semantics':'energy_ang, not energy_xyz',
        'kinetic_temperature_offset_K':273.15,'nu0_s_inverse':NU0,
        'zero_field_exponential':'symmetry characters with positive mode sums',
        'event_transport':'zero-field axial pitchfork symmetry only; other events rejected'}

class SupportError(RuntimeError):pass
class NumericError(RuntimeError):pass
def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def filehash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(p, x):
    import tempfile
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False).encode() + b'\n'
    fd, temporary_name = tempfile.mkstemp(prefix=p.name + '.', suffix='.tmp', dir=p.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            fd = None
            stream.write(payload)
        os.replace(temporary, p)
    finally:
        if fd is not None:
            os.close(fd)
        temporary.unlink(missing_ok=True)
import yaml
from population_protocol import population_quadrature, haar_orientations, high_precision_row
CORE=yaml.safe_load((BASE/'config/core.yaml').read_text())
GEOMETRIES=CORE['geometries']
IDENTITIES={'configuration_identities':[]}
MATERIAL={'TMx_percent':float(CORE['TMx_percent']),'K1_multiplier':1.,'Ms_multiplier':1.,'Tc_delta_C':0.}
CALIBRATION_MS20=float(source.get_material_parms(0,'hard',20.)[3])
TOTAL_VOLUME=float(CORE['total_magnetic_volume_m3'])
ORIENTATIONS={'orientations':haar_orientations(12)}
COMMON={'model':'axial_single_domain','program_version':'2.0','source_energy_sha256':filehash(source.__file__),
        'core_config_sha256':filehash(BASE/'config/core.yaml'),'material':dict(MATERIAL)}
CACHE_RECORDS={}
POLICY['event_transport']='explicit prolate zero-field 2-to-8 reflection pitchfork, bracketed; other count changes unsupported'

def configure_material(TMx_percent=0.,K1_multiplier=1.,Ms_multiplier=1.,Tc_delta_C=0.,nu0=1e9):
    global NU0
    if float(TMx_percent) not in [0.,5.]:raise SupportError('MATERIAL_OUTSIDE_DECLARED_TM00_TM05')
    if K1_multiplier<=0 or Ms_multiplier<=0 or nu0<=0:raise ValueError('INVALID_MATERIAL_PERTURBATION')
    MATERIAL.update(TMx_percent=float(TMx_percent),K1_multiplier=float(K1_multiplier),
                    Ms_multiplier=float(Ms_multiplier),Tc_delta_C=float(Tc_delta_C))
    NU0=float(nu0);COMMON['material']=dict(MATERIAL);COMMON['nu0_s_inverse']=NU0
    material.cache_clear();_landscape.cache_clear();CACHE_RECORDS.clear()

@lru_cache(maxsize=8192)
def material(T):
    T=float(T)
    if not 20.<=T<=350.:raise SupportError('MATERIAL_T_OUTSIDE_20_350C')
    x=MATERIAL['TMx_percent']/100
    Tc=372.37*x**3-691.52*x**2-413.85*x+580.
    Teff=T*Tc/(Tc+MATERIAL['Tc_delta_C'])
    r,k1,k2,ms=source.get_material_parms(MATERIAL['TMx_percent'],'hard',Teff)
    r=np.asarray(r,float)
    if not np.allclose(r,np.eye(3),atol=1e-14):
        raise SupportError('AXIAL_ENUMERATOR_REQUIRES_SOURCE_VERIFIED_IDENTITY_HARD')
    return r,float(k1)*MATERIAL['K1_multiplier'],float(k2),float(ms)*MATERIAL['Ms_multiplier']

def population(n,median_nm=50.,geometric_sd=1.05,support_nm=(35.,60.)):
    p=population_quadrature(n,median_nm,geometric_sd,support_nm)
    p.update(n=int(n),Vmag_total_m3=TOTAL_VOLUME,S_A_m2=CALIBRATION_MS20*TOTAL_VOLUME,
        number_scale=TOTAL_VOLUME/p['mean_grain_volume_m3'],
        quadrature_volume_m3=p['mean_grain_volume_m3'])
    return p

def coeffs(gid,T,B):
    g=GEOMETRIES[gid]
    if not (float(g['PRO'])==1. or float(g['OBL'])==1.) or g.get('rotation','identity')!='identity':raise SupportError('ONLY_AXIAL_HARD_IDENTITY_SUPPORTED')
    r,k1,k2,ms=material(float(T))
    n=np.asarray(source.demag_factors(float(g['PRO']),float(g['OBL'])),float)
    absolute=.5*MU0*ms**2*n
    return k1,k2,absolute-absolute.min(),-ms*np.asarray(B,float),float(absolute.min()),n

def evaluate(u,c):
    k1,k2,D,b,*_=c;x,y,z=u;xx,yy,zz=x*x,y*y,z*z
    e=k1*(xx*yy+xx*zz+yy*zz)+k2*xx*yy*zz+D@np.array([xx,yy,zz])+b@u
    grad=np.array([2*x*(k1*(yy+zz)+k2*yy*zz+D[0])+b[0],
                   2*y*(k1*(xx+zz)+k2*xx*zz+D[1])+b[1],
                   2*z*(k1*(xx+yy)+k2*xx*yy+D[2])+b[2]])
    h=np.array([[2*(k1*(yy+zz)+k2*yy*zz+D[0]),4*x*y*(k1+k2*zz),4*x*z*(k1+k2*yy)],
                [4*x*y*(k1+k2*zz),2*(k1*(xx+zz)+k2*xx*zz+D[1]),4*y*z*(k1+k2*xx)],
                [4*x*z*(k1+k2*yy),4*y*z*(k1+k2*xx),2*(k1*(xx+yy)+k2*xx*yy+D[2])]])
    return e,grad,h
def tangent(u):
    a=np.eye(3)[np.argmin(abs(u))];v=a-(a@u)*u;v/=np.linalg.norm(v)
    return np.column_stack([v,np.cross(u,v)])
def classify(u,c):
    e,g,h=evaluate(u,c);b=tangent(u);H=b.T@(h-(u@g)*np.eye(3))@b
    vals,vecs=np.linalg.eigh(H);scale=max(abs(c[0]),np.max(abs(c[2])),1.)
    residual=np.linalg.norm(g-(g@u)*u)/scale
    if residual>POLICY['root_scaled_residual_max']:raise NumericError('STATIONARITY_RESIDUAL')
    if np.min(abs(vals))<POLICY['morse_relative_threshold']*scale:raise SupportError('DEGENERATE_STATIONARY_POINT_REQUIRES_EVENT_LIMIT')
    return {'u':u,'energy':e,'eigenvalues':vals,'index':int((vals<0).sum()),
            'unstable':b@vecs[:,0],'stationarity_scaled_residual':float(residual)}

def zero_points(c):
    k1,k2,D,*_=c;points=[]
    for axis in range(3):
        for sign in [-1,1]:points.append(sign*np.eye(3)[axis])
    for i,j in itertools.combinations(range(3),2):
        t=.5+(D[i]-D[j])/(2*k1)
        if 0<t<1:
            for si,sj in itertools.product([-1,1],repeat=2):
                u=np.zeros(3);u[i]=si*np.sqrt(t);u[j]=sj*np.sqrt(1-t);points.append(u)
    pairs=[(i,j) for i,j in itertools.combinations(range(3),2) if abs(D[i]-D[j])<1e-10]
    if not pairs:raise SupportError('NON_AXIAL_ZERO_FIELD_ENUMERATION_NOT_IMPLEMENTED')
    i,j=pairs[0];k=3-i-j
    roots=np.roots([-3*k2,k2-3*k1,k1+D[i]-D[k]]) if k2 else [-(k1+D[i]-D[k])/(-3*k1)]
    for s in roots:
        if abs(np.imag(s))<1e-12 and 0<float(np.real(s))<.5:
            s=float(np.real(s))
            for signs in itertools.product([-1,1],repeat=3):
                u=np.empty(3);u[i]=signs[i]*np.sqrt(s);u[j]=signs[j]*np.sqrt(s);u[k]=signs[k]*np.sqrt(1-2*s);points.append(u)
    # The alternative interior factor k1+k2*v_k=0 cannot occur on this
    # simplex when -k1/k2 is outside [0,1]. Verify, do not assume it.
    if k2 and 0<=-k1/k2<=1:raise SupportError('ADDITIONAL_INTERIOR_BRANCH_REQUIRES_SUPPORT')
    return points

def refine(u,c):
    scale=max(abs(c[0]),np.max(abs(c[2])),1.)
    _,g,_=evaluate(u,c);initial=np.r_[u,(u@g)/scale]
    def f(x):
        _,g,_=evaluate(x[:3],c)
        return np.r_[g/scale-x[3]*x[:3],.5*(x[:3]@x[:3]-1)]
    def jac(x):
        _,g,h=evaluate(x[:3],c);J=np.zeros((4,4));J[:3,:3]=h/scale-x[3]*np.eye(3);J[:3,3]=-x[:3];J[3,:3]=x[:3]
        return J
    sol=root(f,initial,jac=jac,tol=POLICY['root_xtol'])
    if np.linalg.norm(f(sol.x))>POLICY['root_scaled_residual_max']:raise NumericError('FINITE_FIELD_ROOT_DID_NOT_CONVERGE')
    u=sol.x[:3];u=u/np.linalg.norm(u)
    return u

def descend(u,c,minima):
    scale=max(abs(c[0]),np.max(abs(c[2])),1.)
    def objective(v):
        norm=np.linalg.norm(v);q=v/norm;e,g,_=evaluate(q,c)
        return e/scale,(g-q*(q@g))/(scale*norm)
    result=minimize(objective,u,jac=True,method='BFGS',options={'gtol':2e-10,'maxiter':250})
    v=result.x/np.linalg.norm(result.x);d=np.linalg.norm(minima-v,axis=1);j=int(np.argmin(d))
    if d[j]>POLICY['basin_angular_residual_max']:raise NumericError('BASIN_DESCENT_UNRESOLVED')
    return j

@lru_cache(maxsize=12000)
def _landscape(context_hash,gid,T,Btuple):
    T=float(T);B=np.asarray(Btuple,float);c=coeffs(gid,T,B)
    zc=coeffs(gid,T,np.zeros(3));seeds=zero_points(zc)
    pts=[classify(refine(u,c) if np.any(B) else u,c) for u in seeds]
    U=np.array([p['u'] for p in pts]);dist=np.linalg.norm(U[:,None]-U[None,:],axis=2)+np.eye(len(U))*10
    if dist.min()<POLICY['root_merge_distance']:raise SupportError('CONTINUATION_ROOT_MERGE')
    minima=[p for p in pts if p['index']==0];saddles=[p for p in pts if p['index']==1];maxima=[p for p in pts if p['index']==2]
    if len(minima)-len(saddles)+len(maxima)!=2:raise NumericError('SPHERE_CRITICAL_CENSUS_EULER_CHECK')
    mins=np.array([p['u'] for p in minima]);channels=[]
    for saddle in saddles:
        if not np.any(B):
            # The energy is even in each coordinate. Coordinate planes are
            # invariant for its gradient flow. Enumeration above establishes
            # one minimum in each occupied reflection sector, so the two
            # descending branches retain the nonzero saddle coordinate signs.
            # This remains well conditioned as a pitchfork is approached.
            common=(abs(saddle['u'])>1e-8)&(np.max(abs(mins),axis=0)>1e-8)
            ends=np.flatnonzero(np.all(np.sign(mins[:,common])==np.sign(saddle['u'][common]),axis=1)).tolist()
            if len(ends)!=2:raise SupportError('ZERO_FIELD_REFLECTION_SECTOR_ENDPOINTS_NOT_UNIQUE')
        else:
            ends=[]
            for sign in [-1,1]:
                a=POLICY['basin_start_angle_rad'];u=saddle['u']*np.cos(a)+sign*saddle['unstable']*np.sin(a)
                ends.append(descend(u,c,mins))
        if ends[0]==ends[1]:raise SupportError('SADDLE_ENDPOINTS_NOT_DISTINCT')
        for j in ends:
            if saddle['energy']-minima[j]['energy']<=0:raise SupportError('NONPOSITIVE_BARRIER_NO_CLIPPING')
        channels.append({'endpoints':ends,'saddle_direction':saddle['u'].tolist(),'saddle_density_J_m3':float(saddle['energy']),
                         'barrier_density_J_m3':[float(saddle['energy']-minima[j]['energy']) for j in ends]})
    result={'gid':gid,'material':dict(MATERIAL),'T_C':T,'B_shape_T':B.tolist(),'minima':mins,'energies':np.array([p['energy'] for p in minima]),
            'channels':channels,'critical_count':[len(minima),len(saddles),len(maxima)],
            'max_stationarity_scaled_residual':max(p['stationarity_scaled_residual'] for p in pts),
            'minimum_absolute_hessian_eigenvalue':float(min(np.min(abs(p['eigenvalues'])) for p in pts)),
            'energy_offset_J_m3':c[4],'N':c[5],'Ms_A_m':material(T)[3],
            'all_critical_directions':U,'all_critical_indices':np.array([p['index'] for p in pts])}
    result['cache_identity']={**COMMON,'key':context_hash,'geometry':GEOMETRIES[gid],
        'configuration_geometry_closures':[x['configuration_geometry_closure_sha256'] for x in IDENTITIES['configuration_identities'] if gid in x['payload']['configuration']['number_fractions']],
        'R_cf_sha256':digest(material(T)[0].T.tolist()),'numerics_sha256':digest(POLICY)}
    CACHE_RECORDS[context_hash]=result
    return result

def landscape_key(gid,T,Btuple=(0.,0.,0.),overrides=None):
    payload={'identity':COMMON,'geometry':GEOMETRIES[gid],'R_cf':material(float(T))[0].T.tolist(),
             'T_C':float(T),'B_shape_T':list(Btuple),'landscape_policy':POLICY,
             'model_code_sha256':filehash(__file__)}
    if overrides:payload.update(overrides)
    return digest(payload)
def landscape(gid,T,Btuple=(0.,0.,0.)):
    return _landscape(landscape_key(gid,T,Btuple),gid,float(T),tuple(Btuple))

def rates(land,diameters):
    d=np.asarray(diameters);v=np.pi/6*(d*1e-9)**3;m=len(land['minima'])
    q=np.zeros((len(d),m,m));channel_rates=[]
    for ch in land['channels']:
        i,j=ch['endpoints'];di,dj=ch['barrier_density_J_m3']
        if not (np.isfinite([di,dj]).all() and di>0 and dj>0):raise SupportError('NONPOSITIVE_BARRIER_NO_CLIPPING')
        a=NU0*np.exp(-v*di/(KB*(land['T_C']+273.15)));b=NU0*np.exp(-v*dj/(KB*(land['T_C']+273.15)))
        q[:,j,i]+=a;q[:,i,j]+=b;channel_rates.append((a,b))
    for j in range(m):q[:,j,j]=-np.sum(q[:,:,j],axis=1)
    return q,channel_rates

def zero_propagator(land,q,h):
    u=land['minima'];axes=np.flatnonzero(np.max(abs(u),axis=0)>1e-7);m=len(u)
    if 2**len(axes)!=m:raise SupportError('ZERO_FIELD_CHARACTER_BASIS_NOT_COMPLETE')
    chars=np.array([np.prod(np.sign(u[:,[axes[k] for k in range(len(axes)) if mask>>k&1]]),axis=1) if mask else np.ones(m) for mask in range(m)]).T
    H=chars/np.sqrt(m)
    if np.max(abs(H.T@H-np.eye(m)))>1e-12:raise NumericError('CHARACTER_BASIS_ORTHOGONALITY')
    lambdas=np.zeros((len(q),m))
    for k in range(1,m):
        vals=np.sum(q*(1-chars[:,k,None]/chars[None,:,k]),axis=1)
        lam=np.mean(vals,axis=1)
        if np.max(abs(vals-lam[:,None])/(np.max(abs(q),axis=(1,2))[:,None]+1e-300))>2e-10:
            raise SupportError('ZERO_FIELD_SYMMETRY_RATE_CHECK_FAILED')
        lambdas[:,k]=lam
    # Every lambda is a sum of nonnegative channel terms, avoiding loss
    # of a slow mode in subtraction of large eigenvalues.
    U=np.einsum('ik,dk,jk->dij',H,np.exp(-h*lambdas),H)
    return U,lambdas

EXP_STATS={'scipy_calls':0,'mpmath_calls':0,'symmetry_calls':0,'max_mass_residual':0.,'minimum_probability':1.,'max_semigroup_comparison':0.}
def propagator(land,diameters,h):
    q,_=rates(land,diameters)
    if not np.any(land['B_shape_T']):
        U,_=zero_propagator(land,q,h);EXP_STATS['symmetry_calls']+=len(q)
    else:
        U=[]
        for Q in q:
            norm=np.linalg.norm(Q*h,1)
            a=expm(Q*h) if norm<POLICY['scipy_expm_norm_max'] else None
            if a is not None and (np.max(abs(a.sum(axis=0)-1))>2e-10 or np.min(a)<-2e-12):a=None
            if a is None:
                import mpmath as mp
                with mp.workdps(POLICY['mpmath_dps']):
                    M=mp.matrix(Q.tolist())
                    for j in range(len(Q)):M[j,j]=-mp.fsum(M[i,j] for i in range(len(Q)) if i!=j)
                    a=np.array(mp.expm(M*mp.mpf(float(h))).tolist(),float)
                EXP_STATS['mpmath_calls']+=1
            else:EXP_STATS['scipy_calls']+=1
            U.append(a)
        U=np.array(U)
    residual=float(np.max(abs(U.sum(axis=1)-1)));EXP_STATS['max_mass_residual']=max(EXP_STATS['max_mass_residual'],residual)
    if residual>POLICY['probability_residual_max'] or U.min()<-2e-10:raise NumericError('PROPAGATOR_MASS_OR_POSITIVITY')
    return U

def transport(old,new):
    a,b=old['minima'],new['minima'];m,n=len(a),len(b)
    ga,gb=GEOMETRIES[old['gid']],GEOMETRIES[new['gid']]
    if any(float(ga[k])!=float(gb[k]) for k in ['PRO','OBL']) or old.get('material')!=new.get('material'):
        raise SupportError('TRANSPORT_BETWEEN_DIFFERENT_PHYSICS')
    if m==n:
        rows,cols=linear_sum_assignment(1-a@b.T);P=np.zeros((n,m));P[cols,rows]=1
        if np.max(np.linalg.norm(a[rows]-b[cols],axis=1))>.45:raise SupportError('STATE_CONTINUATION_TOO_LARGE')
        return P,'CONTINUATION'
    if np.any(old['B_shape_T']) or np.any(new['B_shape_T']) or {m,n}!={2,8}:
        raise SupportError('PHYSICAL_EVENT_TRANSPORT_NOT_SUPPORTED')
    if not (float(ga['PRO'])>1. and float(ga['OBL'])==1.):
        raise SupportError('EVENT_NOT_PROLATE_AXIAL_REFLECTION_PITCHFORK')
    event=event_temperature(old['gid'])
    if event is None or not min(old['T_C'],new['T_C'])<event<max(old['T_C'],new['T_C']):
        raise SupportError('PITCHFORK_NOT_BRACKETED')
    # Four reflection sectors per sign of the shape easy x axis merge into it.
    two=a if m==2 else b;eight=b if n==8 else a
    if np.max(abs(abs(two[:,0])-1))>1e-7 or np.any(abs(eight[:,0])<1e-7):
        raise SupportError('PITCHFORK_AXIS_OR_BRANCH_CHECK_FAILED')
    sectors={(int(np.sign(u[0])),int(np.sign(u[1])),int(np.sign(u[2]))) for u in eight}
    if len(sectors)!=8:raise SupportError('PITCHFORK_REFLECTION_SECTORS_INCOMPLETE')
    P=np.zeros((n,m))
    for j,u in enumerate(a):
        targets=np.flatnonzero(np.sign(b[:,0])==np.sign(u[0]))
        if len(targets)!=(4 if n==8 else 1):raise SupportError('PITCHFORK_BRANCH_MULTIPLICITY')
        P[targets,j]=1/len(targets)
    return P,'PROLATE_ZERO_FIELD_REFLECTION_PITCHFORK'


def prepare(land,R,orientation_index=None,orientation_spec=None,preparation_axis=None):
    axis=np.array([0.,0.,1.]) if preparation_axis is None else np.asarray(preparation_axis,float)
    if abs(np.linalg.norm(axis)-1)>1e-12:raise ValueError('PREPARATION_AXIS_MUST_BE_UNIT')
    projection=axis@R@land['minima'].T;best=projection.max();selected=np.flatnonzero(best-projection<=POLICY['preparation_projection_tolerance'])
    status='UNIQUE_WELL_SEPARATED';projections_hp=None
    if len(selected)>1:
        import mpmath as mp
        with mp.workdps(POLICY['preparation_high_precision_dps']):
            row=high_precision_row(R,orientation_spec) if preparation_axis is None else [mp.mpf(float(v)) for v in (axis@R)]
            k1,k2,D,*_=coeffs(land['gid'],land['T_C'],np.zeros(3));K1,K2=mp.mpf(k1),mp.mpf(k2);D=[mp.mpf(float(t)) for t in D]
            hp=[]
            for u in land['minima']:
                active=np.flatnonzero(abs(u)>1e-8);v=[mp.mpf(0)]*3
                if len(active)==1:v[int(active[0])]=mp.mpf(int(np.sign(u[active[0]])))
                elif len(active)==2:
                    a,b=map(int,active);t=mp.mpf('.5')+(D[a]-D[b])/(2*K1)
                    v[a]=int(np.sign(u[a]))*mp.sqrt(t);v[b]=int(np.sign(u[b]))*mp.sqrt(1-t)
                else:
                    a,b=next((a,b) for a,b in itertools.combinations(range(3),2) if abs(float(D[a]-D[b]))<1e-10);k=3-a-b
                    s=mp.findroot(lambda s:(1-3*s)*(K1+K2*s)+D[a]-D[k],mp.mpf(float(u[a]**2)))
                    v[a]=int(np.sign(u[a]))*mp.sqrt(s);v[b]=int(np.sign(u[b]))*mp.sqrt(s);v[k]=int(np.sign(u[k]))*mp.sqrt(1-2*s)
                hp.append(mp.fsum(row[j]*v[j] for j in range(3)))
            top=max(hp);selected=np.array([j for j,t in enumerate(hp) if abs(top-t)<mp.mpf('1e-50')],int)
            projections_hp=[mp.nstr(x,65) for x in hp];status='HIGH_PRECISION_SYMMETRY_TIE' if len(selected)>1 else 'NEAR_TIE_RESOLVED_HIGH_PRECISION'
    p=np.zeros(len(projection));p[selected]=1/len(selected)
    return p,{'projection_z':projection.tolist(),'occupied_indices':selected.tolist(),'tie_count':len(selected),'tie_high_precision_status':status,'high_precision_projections':projections_hp}

def event_temperature(gid):
    g=GEOMETRIES[gid]
    if not (float(g['PRO'])>1. and float(g['OBL'])==1.):return None
    def f(T):
        k1,k2,D,*_=coeffs(gid,float(T),np.zeros(3))
        if abs(D[1]-D[2])>1e-9:raise SupportError('EVENT_REQUIRES_AXIAL_TRANSVERSE_SYMMETRY')
        return k1+D[1]-D[0]
    return float(brentq(f,20.,350.,xtol=1e-11)) if f(20.)*f(350.)<0 else None
