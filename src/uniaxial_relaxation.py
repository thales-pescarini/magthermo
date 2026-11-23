"""Finite vector uniaxial comparators. No F geometry or F event maps used."""
import math
import time
import numpy as np
import mpmath as mp
import grain_energy as core
from relaxation_spectrum import Spectral


class NotApplicable(core.Unresolved):
    pass


def uniaxial(K, Ms, axis, B, identity=None):
    """Two fixed polynomial precision levels; exact axial branch first.

    In the (axis,B) plane, theta and its orthogonal tangent diagonalize the
    FULL constrained Hessian. The second eigenvalue is -u.g, not theta''.
    The two descending planar neighbours prove the reduced saddle linkage.
    """
    attempts=[]
    for dps in (80,110):
        start=time.monotonic()
        with mp.workdps(dps):
            k=mp.mpf(K);ms=mp.mpf(Ms)
            a=mp.matrix(axis);a/=mp.sqrt(mp.fsum(x*x for x in a))
            field=mp.matrix(B)
            par=mp.fsum(a[i]*field[i] for i in range(3))
            transverse=field-par*a;perp=mp.sqrt(mp.fsum(x*x for x in transverse))
            if k<=0:
                raise NotApplicable('NOT_APPLICABLE_TWO_WELL:K0_NONPOSITIVE')
            if perp==0:
                h=ms*par/(2*k)
                if abs(h)>=1:
                    raise NotApplicable('NOT_APPLICABLE_TWO_WELL:AXIAL_METASTABILITY_ABSENT')
                minima=[]
                for sign in (1,-1):
                    v=sign*a
                    minima.append(dict(m=list(map(float,v)),E=mp.nstr(-sign*ms*par,dps),
                                       gradient_residual='0',eig=[mp.nstr(2*k+sign*ms*par,dps)]*2,
                                       point_id='n0_axial_'+str(sign),index=0))
                es=k*(1+h*h)
                return dict(status='COMPUTED_VALID',minima=minima,saddle_E=mp.nstr(es,dps),
                            barrier_density=[mp.nstr(k*(1+sign*h)**2,dps) for sign in (1,-1)],
                            identity=identity or {},K0=mp.nstr(k,dps),Ms=float(ms),axis=list(map(float,a)),
                            B_crystal_T=list(map(float,field)),dps=dps,
                            verification=dict(branch='exact_axial_effective_barrier',h=mp.nstr(h,dps),
                                              no_channel_multiplicity=True,saddle_is_degenerate_circle=True,
                                              two_wells=True,attempts=attempts,geometry_wall_s=time.monotonic()-start))
            b=transverse/perp
            coefficients=[ms*perp,-4*k+2*ms*par,mp.mpf(0),4*k+2*ms*par,-ms*perp]
            try:
                # Polynomial companion roots are initialization, not an angular
                # search. No mesh, root seed expansion or third precision.
                scale=max(abs(x) for x in coefficients)
                initial=np.roots([float(x/scale) for x in coefficients])
                roots=[mp.mpc(str(v.real),str(v.imag)) for v in initial]
                monic=[v/coefficients[0] for v in coefficients]
                # V1: a relative update criterion avoids demanding an absolute
                # correction below MP epsilon for a root |t| >> 1. Precision,
                # polynomial, 100-iteration cap and physical tests are unchanged.
                converged=False
                for iteration in range(100):
                    relative_updates=[]
                    for i in range(4):
                        value=monic[0]
                        for coefficient in monic[1:]:value=value*roots[i]+coefficient
                        denominator=mp.mpf(1)
                        for j in range(4):
                            if i!=j:denominator*=roots[i]-roots[j]
                        if denominator==0:
                            raise core.Unresolved('N0_POLYNOMIAL_COINCIDENT_ITERATES')
                        correction=value/denominator
                        roots[i]-=correction
                        relative_updates.append(abs(correction)/max(1,abs(roots[i])))
                    if max(relative_updates)<mp.power(10,-dps//2):
                        converged=True;break
                if not converged:
                    raise core.Unresolved('N0_POLYNOMIAL_100_ITERATIONS_EXHAUSTED')
                points=[];residuals=[]
                real_tolerance=mp.power(10,-dps//2)
                for root in roots:
                    if abs(mp.im(root))>real_tolerance*max(1,abs(root)):
                        continue
                    theta=2*mp.atan(mp.re(root))
                    x=mp.cos(theta);y=mp.sin(theta);u=x*a+y*b
                    original=2*k*y*x+ms*par*y-ms*perp*x
                    residual=abs(original)/max(k,ms*abs(par),ms*perp)
                    if residual>mp.power(10,-dps//3):
                        raise core.Unresolved('N0_ORIGINAL_STATIONARITY_RESIDUAL')
                    energy=k*(1-x*x)-ms*(par*x+perp*y)
                    ev_plane=2*k*(x*x-y*y)+ms*(par*x+perp*y)
                    ev_orthogonal=2*k*x*x+ms*(par*x+perp*y)
                    if min(abs(ev_plane),abs(ev_orthogonal))<scale*mp.power(10,-dps//3):
                        raise core.Unresolved('N0_TANGENT_CLASSIFICATION_AMBIGUOUS')
                    index=int(ev_plane<0)+int(ev_orthogonal<0)
                    points.append(dict(theta=theta%(2*mp.pi),u=u,energy=energy,
                                       eigen=[ev_plane,ev_orthogonal],index=index,residual=residual))
                    residuals.append(residual)
                # theta=pi omitted by tan(theta/2) has f(pi)=Ms B_perp,
                # strictly nonzero on this branch. It is explicitly checked.
                theta_pi_residual=ms*perp
                points.sort(key=lambda v:v['theta'])
                if len(points) not in (2,4):
                    raise core.Unresolved('N0_REAL_POLYNOMIAL_ROOT_COUNT_UNRESOLVED')
                minima=[p for p in points if p['index']==0]
                saddles=[p for p in points if p['index']==1]
                maxima=[p for p in points if p['index']==2]
                counts=[len(minima),len(saddles),len(maxima)]
                if counts==[1,0,1]:
                    raise NotApplicable('NOT_APPLICABLE_TWO_WELL:VERIFIED_ONE_MINIMUM')
                if counts!=[2,1,1]:
                    raise core.Unresolved('N0_REDUCED_CENSUS_UNRESOLVED:'+str(counts))
                saddle=saddles[0];j=points.index(saddle)
                if points[(j-1)%len(points)]['index']!=0 or points[(j+1)%len(points)]['index']!=0:
                    raise core.Unresolved('N0_SADDLE_LINKAGE_UNRESOLVED')
                barriers=[saddle['energy']-p['energy'] for p in minima]
                if min(barriers)<=0:
                    raise core.Unresolved('N0_NONPOSITIVE_BARRIER_NO_CLIPPING')
                separation=min(mp.sqrt(mp.fsum((p['u'][i]-q['u'][i])**2 for i in range(3)))
                               for j,p in enumerate(points) for q in points[:j])
                rows=[dict(m=list(map(float,p['u'])),E=mp.nstr(p['energy'],dps),
                           eig=[mp.nstr(e,dps) for e in p['eigen']],index=0,
                           gradient_residual=mp.nstr(p['residual'],dps),point_id='n0_polynomial_'+str(j))
                      for j,p in enumerate(minima)]
                return dict(status='COMPUTED_VALID',minima=rows,saddle_E=mp.nstr(saddle['energy'],dps),
                            barrier_density=[mp.nstr(v,dps) for v in barriers],identity=identity or {},
                            K0=mp.nstr(k,dps),Ms=float(ms),axis=list(map(float,a)),B_crystal_T=list(map(float,field)),dps=dps,
                            verification=dict(branch='finite_polynomial_full_tangent_Hessian',counts=counts,
                                              original_equation_max_relative_residual=mp.nstr(max(residuals),dps),
                                              root_separation=mp.nstr(separation,dps),
                                              theta_pi_checked=True,theta_pi_stationarity_residual=mp.nstr(theta_pi_residual,dps),
                                              saddle_links='two adjacent planar minima in enumerated cyclic critical order',
                                              saddle_eigenvalues=[mp.nstr(e,dps) for e in saddle['eigen']],
                                              maximum_eigenvalues=[mp.nstr(e,dps) for e in maxima[0]['eigen']],
                                              antipodal_distance=float(np.linalg.norm(np.array(rows[0]['m'])+rows[1]['m'])),
                                              polynomial_iterations=iteration+1,
                                              maximum_relative_polynomial_update=mp.nstr(max(relative_updates),dps),
                                              attempts=attempts,geometry_wall_s=time.monotonic()-start))
            except NotApplicable:
                raise
            except Exception as exc:
                attempts.append(dict(dps=dps,reason=type(exc).__name__+':'+str(exc),wall_s=time.monotonic()-start))
    err=core.Unresolved('N0_NUMERICAL_UNRESOLVED_TWO_PRECISIONS_EXHAUSTED')
    err.evidence=attempts
    raise err


def geometry_N0(variant,grain,T,B_sample):
    R=np.asarray(grain['R'],float)
    c=np.asarray(grain['c'],float)
    pm=core.PointModel(grain['s'],c,T,R.T@np.asarray(B_sample,float),0)
    if variant=='N0_shape':
        parallel=float(c@pm.N@c)/(float(c@c))
        perpendicular=(float(np.trace(pm.N))-parallel)/2
        K=0. if grain['s']==0 else .5*core.legacy.MU0*pm.Ms**2*abs(perpendicular-parallel)
        axis=c.tolist()
    else:
        K=-pm.k1/12-pm.k2/27
        # Normalize the exact [111] axis inside each MP level.
        axis=[1,1,1]
    ident=dict(variant=variant,s=grain['s'],c=grain['c'],T_C=float(T),B_crystal_T=pm.B.tolist(),
               Ms=pm.Ms,K1=pm.k1,K2=pm.k2,TM=0,R=grain['R'],
               artificial_easy_axis_oblate=(variant=='N0_shape' and grain['s']<0),
               prefactor_per_s=1e9,multiplicity=1,energy_unit='J/m3',field_unit='T')
    graph=uniaxial(K,pm.Ms,axis,pm.B.tolist(),ident)
    graph['identity']['Ms']=pm.Ms
    return graph


def spectral_N0(graph,d,T):
    precision=graph['dps']
    with mp.workdps(precision):
        V=mp.mpf(math.pi/6*(d*1e-9)**3)
        kT=mp.mpf(core.KB)*mp.mpf(T+273.15)
        barriers=list(map(mp.mpf,graph['barrier_density']))
        rates=[mp.mpf(1e9)*mp.exp(-V*e/kT) for e in barriers]
        a,b=rates;total=a+b
        Q=mp.matrix([[-a,b],[a,-b]])
        pi=mp.matrix([b/total,a/total])
        s=Spectral(Q,pi,precision,identity=graph['identity'])
        s.diagnostics.update(N0_effective_saddle_multiplicity=1,
                             min_barrier_over_kBT=str(V*min(barriers)/kT))
        return s


def preparation(graph,grain):
    with mp.workdps(80):
        row=core.high_precision_row(np.asarray(grain['R']),grain.get('R_metadata'))
        # Preparation is at zero field, so its own minima are exactly +/-a.
        axis=mp.matrix([1,1,1] if graph['identity']['variant']=='N0_crystal' else grain['c'])
        axis/=mp.sqrt(mp.fsum(x*x for x in axis))
        value=mp.fsum(row[i]*axis[i] for i in range(3))
        if abs(value)<mp.mpf('1e-50'):
            return np.array([.5,.5])
        return np.array([1.,0.]) if value>0 else np.array([0.,1.])


def align(source,target):
    U=np.asarray([p['m'] for p in source['minima']]);V=np.asarray([p['m'] for p in target['minima']])
    direct=float(np.sum((U-V)**2));reverse=float(np.sum((U-V[::-1])**2))
    if abs(direct-reverse)<1e-20*max(1.,direct,reverse):
        raise core.Unresolved('N0_CONTINUOUS_PERMUTATION_AMBIGUOUS')
    return np.eye(2) if direct<reverse else np.array([[0.,1.],[1.,0.]])
