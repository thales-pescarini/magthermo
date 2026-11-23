"""Check relaxation operators against analytic Markov chains and saved energy landscapes."""
from pathlib import Path
import json
import math
import time
import numpy as np
import mpmath as mp
from scipy.linalg import expm
import grain_energy as core
import thermal_executor as old
from relaxation_spectrum import Spectral,spectral_F,physical_inputs,observer,maxabs
from uniaxial_relaxation import uniaxial,spectral_N0,NotApplicable


def binary(k12,k21):
    with mp.workdps(80):
        a=mp.mpf(str(k12));b=mp.mpf(str(k21))
        return Spectral([[-a,b],[a,-b]],[b/(a+b),a/(a+b)],80)


def manufactured_slow():
    with mp.workdps(80):
        K=mp.mpf('1000000');e=mp.mpf('0.000001')
        return Spectral([[-e,K,0],[e,-2*K,e],[0,K,-e]],[K/(2*K+e),e/(2*K+e),K/(2*K+e)],80)


def mathematical(spec):
    name=spec['id'];res={};tol=1e-10
    with mp.workdps(80):
        if name in ('A01','A02'):
            sp=binary(.0003,.0003) if name=='A01' else binary(spec['k12'],spec['k21'])
            W=np.array([[1.,-1.],[0.,0.],[0.,0.]])
            selected=sp.select(W);Q1=selected['rate']*(sp.projectors[0]-mp.eye(2))
            residual=float(maxabs(Q1-sp.Q));expected=mp.mpf(str(spec['expected_lambda1']))
            rel=float(abs(selected['rate']-expected)/expected)
            differences=[];minimum=1.;mass=0.
            for seconds in spec['seconds']:
                U,info=sp.propagator(seconds,'N1',selected)
                expected_U=expm(np.asarray(sp.Q.tolist(),float)*seconds)
                differences.append(float(np.max(abs(U-expected_U))))
                minimum=min(minimum,info['minimum']);mass=max(mass,info['mass_residual'])
            res=dict(generator_error=residual,lambda_relative_error=rel,propagator_max_errors=differences,
                     selected_rate=str(selected['rate']),pi=[str(x) for x in sp.pi],minimum=minimum,mass=mass,
                     pass_all=residual<tol and rel<tol and max(differences)<tol and minimum>=-tol and mass<tol)
        elif name=='A03':
            pi=mp.matrix([mp.mpf(1)/3]*3);P0=pi*mp.ones(1,3);Q=mp.mpf('.001')*(P0-mp.eye(3))
            sp=Spectral(Q,pi,80);sel=sp.select(spec['W'])
            P=sp.projectors[1]+sp.projectors[2]
            # Independent rotation of a concrete orthonormal basis for 1^perp.
            z1=mp.matrix([1,-1,0])/mp.sqrt(2);z2=mp.matrix([1,1,-2])/mp.sqrt(6)
            theta=mp.mpf(str(spec['basis_rotation_radians']))
            r1=mp.cos(theta)*z1+mp.sin(theta)*z2;r2=-mp.sin(theta)*z1+mp.cos(theta)*z2
            rotated=r1*r1.T+r2*r2.T;C=mp.eye(3)-mp.ones(3)/3;W=mp.matrix(spec['W'])
            a=mp.sqrt(mp.fsum(x*x for x in W*P*C));b=mp.sqrt(mp.fsum(x*x for x in W*rotated*C))
            error=float(maxabs(P-rotated));visibility_error=float(abs(a-b)/a)
            res=dict(projector_error=error,visibility_basis_rotation_relative_error=visibility_error,
                     group_count=len(sel['groups']),groups=sel['groups'],
                     pass_all=error<tol and visibility_error<tol and len(sel['groups'])==1)
        elif name in ('A04','A05'):
            sp=manufactured_slow();W=np.array([[1.,0.,-1.],[0.,0.,0.],[0.,0.,0.]])
            original=sp.select(W)
            if name=='A05':
                p=spec['permutation_zero_based'];R=np.asarray(spec['output_rotation'])
                transformed=Spectral([[sp.Q[i,j] for j in p] for i in p],[sp.pi[i] for i in p],80)
                selected=transformed.select(R@W[:,p])
                eig_error=max(float(abs(a-b)/a) for a,b in zip(sp.rates[1:],transformed.rates[1:]))
                visibility_error=max(abs(float(a['visibility'])-float(b['visibility'])) for a,b in zip(original['groups'],selected['groups']))
                res.update(permutation_relative_rate_error=eig_error,rotation_visibility_error=visibility_error)
            else:selected=original
            relative=float(abs(selected['rate']-mp.mpf('1e-6'))/mp.mpf('1e-6'))
            # Selection was constructed with W and projectors only; p0=pi is
            # separately inserted to show zero actual relaxation amplitude.
            amplitude=float(maxabs(mp.matrix(W.tolist())*(sp.projectors[1]*sp.pi)))
            res.update(rate_relative_error_against_slow_rate=relative,selected_rate=str(selected['rate']),
                       rates=[str(x) for x in sp.rates],groups=selected['groups'],
                       equilibrium_preparation_amplitude=amplitude,preparation_used_in_selection=False,
                       pass_all=relative<tol and res.get('permutation_relative_rate_error',0)<tol and res.get('rotation_visibility_error',0)<tol)
        elif name=='A06':
            one=Spectral([[0]],[1],80);u,_=one.propagator(1000)
            status=None
            try:Spectral([[-1,1,0],[1,-1,0],[0,0,0]],[mp.mpf('.25'),mp.mpf('.25'),mp.mpf('.5')],80)
            except core.Unresolved as exc:status=str(exc)
            res=dict(mathematical_one_state_operator=u.tolist(),
                     physical_F_single_state_support='single-state physical propagator is undefined',
                     nonunique_status=status,pass_all=bool(u[0,0]==1 and status=='N1_UNDEFINED_NONUNIQUE_EQUILIBRIUM'))
        elif name=='A07':
            rows=[]
            for B in spec['fields_T']:
                try:
                    graph=uniaxial(spec['K_J_m3'],spec['Ms_A_m'],spec['axis'],B)
                    s=spectral_N0(graph,(6*spec['V_m3']/math.pi)**(1/3)*1e9,spec['T_K']-273.15)
                    station=float(maxabs(s.Q*s.pi));rows.append(dict(field=B,status='COMPUTED_VALID',
                        barriers=graph['barrier_density'],verification=graph['verification'],stationary_residual=station,
                        directions=[p['m'] for p in graph['minima']]))
                except NotApplicable as exc:rows.append(dict(field=B,status='NOT_APPLICABLE_TWO_WELL',reason=str(exc)))
            h=spec['Ms_A_m']*spec['fields_T'][1][2]/(2*spec['K_J_m3'])
            rel=max(abs(float(e)-spec['K_J_m3']*(1+sign*h)**2)/(spec['K_J_m3']*(1+sign*h)**2)
                    for e,sign in zip(rows[1]['barriers'],(1,-1)))
            oblique=np.linalg.norm(np.asarray(rows[2]['directions'][0])+rows[2]['directions'][1])
            res=dict(cases=rows,analytic_axial_barrier_relative_error=rel,oblique_antipodal_distance=float(oblique),
                     pass_all=rel<tol and oblique>1e-10 and rows[-1]['status']=='NOT_APPLICABLE_TWO_WELL' and all(v.get('stationary_residual',0)<tol for v in rows))
        else:
            sp=binary(1e-6,.002);u10,_=sp.propagator(10);u20,_=sp.propagator(20)
            pi=np.array(spec['U3_pi']);Q3=spec['U3_lambda']*(pi[:,None]*np.ones((1,3))-np.eye(3))
            u3=expm(Q3*spec['U3_seconds']);P23=np.asarray(spec['P_2to3']);P32=np.asarray(spec['P_3to2']);W=np.asarray(spec['W2'])
            A=W@u20@P32@u3@P23@u10;errors=[]
            for p in spec['initial_states']:
                q=u10@p;q=P23@q;q=u3@q;q=P32@q;q=u20@q
                errors.append(float(np.max(abs(W@q-A@p))))
            stochastic=max(float(np.max(abs(x.sum(axis=0)-1))) for x in [u10,u20,u3,P23,P32])
            res=dict(sequential_composed_errors=errors,stochastic_error=stochastic,
                     rectangular_shapes=[list(P23.shape),list(P32.shape)],inversions=0,
                     pass_all=max(errors)<tol and stochastic<tol)
    return dict(id=name,status='PASS' if res['pass_all'] else 'FAIL',result=res,
                numeric_policy='scale-aware 1e-10; slow-mode relative error uses its own rate')


def spectral_control(spec,reference_dir,object_dir,resolver,count):
    path=Path(reference_dir)/spec['graph'];graph=resolver.resolve(json.loads(path.read_text()),object_dir)
    _,_,_,base_dps,_,_=physical_inputs(graph,graph['identity']['T_C'],spec['d_nm'])
    levels=[base_dps,min(base_dps+30,2048)];old_curves=[]
    for seconds in spec['operator_probe_seconds']:
        count('snapshot_old_reference_probes_total')
        U,info=old.operator(graph,graph['identity']['T_C'],spec['d_nm'],seconds)
        old_curves.append((U,info))
    results=[]
    for dps in levels:
        count('spectral_precision_evaluations_total')
        sp=spectral_F(graph,spec['d_nm'],graph['identity']['T_C'],dps)
        selection=sp.select(observer(graph,spec['R']));probes=[]
        for seconds,(ref,old_info) in zip(spec['operator_probe_seconds'],old_curves):
            count('snapshot_new_propagator_probes_total');U,info=sp.propagator(seconds)
            count('snapshot_new_propagator_probes_total');U1,info1=sp.propagator(seconds,'N1',selection)
            probes.append(dict(seconds=seconds,new_old_F_max_abs=float(np.max(abs(U-ref))),
                               F_mass=info['mass_residual'],F_minimum=info['minimum'],
                               N1_mass=info1['mass_residual'],N1_minimum=info1['minimum'],
                               old_dps=old_info['dps'],same_precision_old_reference=(old_info['dps']==dps)))
        results.append(dict(dps=dps,spectrum=sp.summary(full=True),selection={k:(str(v) if k=='rate' else v) for k,v in selection.items()},probes=probes))
    with mp.workdps(max(levels)):
        lo=mp.mpf(results[0]['selection']['rate']);hi=mp.mpf(results[1]['selection']['rate'])
        selected_rate_relative_difference=float(abs(lo-hi)/hi) if hi else 0.
    group_sizes=[[len(g['indices']) for g in x['selection']['groups']] for x in results]
    independent=levels[0]!=levels[1]
    max_error=max(p['new_old_F_max_abs'] for r in results for p in r['probes'])
    event_check=None
    if 'event' in spec:
        event=resolver.resolve(json.loads((Path(reference_dir)/spec['event']).read_text()),object_dir)
        P=np.asarray(event['map'],float)
        event_check=dict(shape=list(P.shape),column_mass_error=float(np.max(abs(P.sum(axis=0)-1))),
                         minimum=float(P.min()),status=event['status'],source_sha256=spec['event_sha256'])
    passed=(max_error<1e-12 and independent and selected_rate_relative_difference<1e-10 and group_sizes[0]==group_sizes[1]
            and (event_check is None or (event_check['column_mass_error']<1e-12 and event_check['minimum']>=0)))
    return dict(id=spec['id'],status='PASS' if passed else 'NUMERICAL_UNRESOLVED',graph=spec['graph'],
                graph_sha256=spec['graph_sha256'],d_nm=spec['d_nm'],precision_results=results,
                two_independent_precisions=independent,selected_rate_relative_difference=selected_rate_relative_difference,
                max_old_F_operator_difference=max_error,event_map=event_check,geometry_recomputed=False)
