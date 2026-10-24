"""Probability-coordinate integration of the unchanged E1 number measure.

The scalar reference uses analytic x moments and adaptive integration in s,
independently of the discrete probability-coordinate rule.
"""
import math
import numpy as np
from scipy.special import log_ndtr, ndtri_exp
from scipy.integrate import quad
import e1_population as inherited

def interval(a,b):
    """Return log P(a<Z<b), and inverse conditional CDF without tail loss."""
    if not a < b: raise ValueError('EMPTY_NORMAL_INTERVAL')
    reflected=a>=0
    lo,hi=(-b,-a) if reflected else (a,b)
    lhi=float(log_ndtr(hi));llo=float(log_ndtr(lo))
    delta=llo-lhi
    if not delta<0: raise ValueError('NORMAL_PROBABILITY_INTERVAL_NOT_REPRESENTABLE')
    logmass=lhi+math.log(-math.expm1(delta))
    if logmass<math.log(np.nextafter(0.,1.)): raise ValueError('POSITIVE_MASS_BELOW_FLOAT_RANGE')
    def quantile(u):
        u=np.asarray(u)
        logp=np.logaddexp(llo,np.log(u)+logmass)
        z=ndtri_exp(logp)
        if reflected:z=-z
        if not np.all(np.isfinite(z)):raise ValueError('INVERSE_CDF_NOT_REPRESENTABLE')
        return z
    return logmass,quantile

def _r1_nodes_not_used(spec,nd=33,ns=8,nc=6,alignment='isotropic',retain_nodes=False):
    t=spec['theta'];sx=t['sigma_ln_d']*math.sqrt(1-t['rho_latent']**2)
    qd,wd=np.polynomial.legendre.leggauss(nd);ud=(qd+1)/2;wd=wd/2
    qs,ws=np.polynomial.legendre.leggauss(ns);us=(qs+1)/2;ws=ws/2
    if alignment=='isotropic':
        cs,wc=inherited.triangle(nc);wc*=inherited.orientation_density(cs,t['kappa'])
    else:
        cs=np.array([[1.,0,0]]) if alignment=='hard_100' else np.ones((1,3))/math.sqrt(3);wc=np.ones(1)
    xs=[];ss=[];cc=[];ww=[];branch_records=[]
    for k,comp in enumerate(spec['components']):
        for sign,fraction in [(1,t['f_prolate']),(-1,1-t['f_prolate'])]:
            if fraction==0 or comp['weight']==0:continue
            center=sign*(t['shape_center_abs_s']+comp['delta_s']);sd=t['sigma_s']
            a,b=(0.,inherited.SMAX) if sign>0 else (-inherited.SMAX,0.)
            logprob,q=interval((a-center)/sd,(b-center)/sd)
            logzs=logprob-float(log_ndtr(sign*center/sd));zs=math.exp(logzs)
            shape=center+sd*q(us)
            branch_records.append(dict(component=k,sign=sign,Z_s=zs,n_s=ns))
            for sk,wsk in zip(shape,ws):
                mu=t['mu_ln_d']+comp['delta_ln_d']+t['rho_latent']*t['sigma_ln_d']/sd*(sk-center)
                # The inherited envelope is independent of c. Keeping c explicit
                # prevents applying this simplification to a future c support.
                for ck,wck in zip(cs,wc):
                    upper=inherited.sd_limit(float(sk),alignment)
                    lzx,qx=interval((math.log(5.)-mu)/sx,(math.log(upper)-mu)/sx)
                    x=mu+sx*qx(ud)
                    raw=comp['weight']*fraction*zs*math.exp(lzx)*wsk*wck*wd
                    xs.extend(x);ss.extend(np.full(nd,sk));cc.extend(np.tile(ck,(nd,1)));ww.extend(raw)
    x=np.array(xs);s=np.array(ss);c=np.array(cc);raw=np.array(ww);mass=float(raw.sum())
    if not 0<mass<=1+1e-10:raise ValueError('JOINT_MASS_OUTSIDE_PROBABILITY_DOMAIN')
    weights=raw/mass;d=np.exp(x);V=np.pi/6*1e-27*d**3;meanV=float(weights@V)
    result=dict(nd=nd,ns_per_nonzero_component_branch=ns,nc=len(cs),nodes=len(x),
        effective_s_nodes=ns*len(branch_records),component_branches=branch_records,
        retained_number_mass=mass,raw_volume_moment_m3=float(raw@V),mean_volume_m3=meanV,
        total_number=1e-18/meanV,weight_sum=float(weights.sum()),
        mean_ln_d=float(weights@x),mean_s=float(weights@s),
        jacobian='x=ln(d/nm); Gaussian x measure, no extra diameter Jacobian',
        one_joint_normalization=True,magnetic_observable_qualified=False)
    if retain_nodes:result.update(d_nm=d,s=s,c=c,number_weights=weights,raw_number_weights=raw,volume_m3=V)
    return result

def partition(spec):
    """Freeze SD interpolation panels, retaining the entire original support."""
    table=inherited.json.loads((inherited.BASE/'sources/SD_TABLE_EXTRACT.json').read_text())
    knots=np.log(np.asarray(table['x_Hard DOWN'][0][1:],float))
    t=spec['theta'];records=[]
    for k,comp in enumerate(spec['components']):
        for sign,f in [(1,t['f_prolate']),(-1,1-t['f_prolate'])]:
            if not f or not comp['weight']:continue
            center=sign*(t['shape_center_abs_s']+comp['delta_s']);sd=t['sigma_s']
            a,b=(0.,inherited.SMAX) if sign>0 else (-inherited.SMAX,0.)
            edges=[a]+sorted(set(float(x) for x in knots if a<x<b))+[b];merges=[]
            # Only a representability failure permits a merge, never a small
            # probability threshold. Every removed boundary is recorded.
            while True:
                failed=None
                for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
                    try:interval((lo-center)/sd,(hi-center)/sd)
                    except ValueError:failed=j;break
                if failed is None:break
                if len(edges)<=2:raise ValueError('FULL_BRANCH_PROBABILITY_UNREPRESENTABLE')
                boundary=failed+1 if failed<len(edges)-2 else failed
                merges.append(dict(removed_boundary=edges[boundary],reason='UNREPRESENTABLE_PANEL_MERGED_WITH_NEIGHBOR'))
                edges.pop(boundary)
            lm,_=interval((a-center)/sd,(b-center)/sd)
            panels=[]
            for lo,hi in zip(edges[:-1],edges[1:]):
                lp,_=interval((lo-center)/sd,(hi-center)/sd)
                panels.append(dict(s_interval=[lo,hi],log_normal_mass=lp,conditional_branch_fraction=math.exp(lp-lm)))
            total=math.fsum(p['conditional_branch_fraction'] for p in panels)
            if abs(total-1)>2e-13:raise ValueError('PANELS_DO_NOT_RECONSTRUCT_BRANCH')
            records.append(dict(component=k,sign=sign,edges=edges,panels=panels,merges=merges,
                Z_s=math.exp(lm-float(log_ndtr(sign*center/sd))),panel_probability_sum=total))
    return dict(population=spec['id'],branches=records,rule='probability_GL_per_SD_panel',support_unchanged=True)

def node_blocks(spec,part,nd=33,ns=8,nc=6,alignment='isotropic'):
    """Yield bounded raw number-measure blocks. No panel normalization."""
    t=spec['theta'];sd=t['sigma_s'];sx=t['sigma_ln_d']*math.sqrt(1-t['rho_latent']**2)
    qd,wd=np.polynomial.legendre.leggauss(nd);ud=(qd+1)/2;wd/=2
    qs,ws=np.polynomial.legendre.leggauss(ns);us=(qs+1)/2;ws/=2
    if alignment=='isotropic':
        cs,wc=inherited.triangle(nc);wc*=inherited.orientation_density(cs,t['kappa'])
    else:
        cs=np.array([[1.,0,0]]) if alignment=='hard_100' else np.ones((1,3))/math.sqrt(3);wc=np.ones(1)
    for branch in part['branches']:
        comp=spec['components'][branch['component']];sign=branch['sign']
        fraction=t['f_prolate'] if sign>0 else 1-t['f_prolate']
        center=sign*(t['shape_center_abs_s']+comp['delta_s'])
        for panel in branch['panels']:
            a,b=panel['s_interval'];lp,q=interval((a-center)/sd,(b-center)/sd)
            panel_factor=comp['weight']*fraction*branch['Z_s']*panel['conditional_branch_fraction']
            for sk,wsk in zip(center+sd*q(us),ws):
                mu=t['mu_ln_d']+comp['delta_ln_d']+t['rho_latent']*t['sigma_ln_d']/sd*(sk-center)
                lzx,qx=interval((math.log(5.)-mu)/sx,(math.log(inherited.sd_limit(float(sk),alignment))-mu)/sx)
                x=mu+sx*qx(ud);d=np.exp(x)
                raw=panel_factor*math.exp(lzx)*wsk*(wc[:,None]*wd[None,:])
                yield dict(x=np.tile(x,len(cs)),d_nm=np.tile(d,len(cs)),s=np.full(raw.size,sk),
                    c=np.repeat(cs,nd,axis=0),raw_number_weights=raw.ravel(),
                    volume_m3=np.tile(math.pi/6*1e-27*d**3,len(cs)))

def nodes(spec,nd=33,ns=8,nc=6,alignment='isotropic',retain_nodes=False,part=None):
    part=partition(spec) if part is None else part
    sums=[];vol=[];mx=[];ms=[];kept=[];count=0;max_block=0
    for block in node_blocks(spec,part,nd,ns,nc,alignment):
        w=block['raw_number_weights'];count+=len(w);max_block=max(max_block,len(w))
        sums.append(float(np.sum(w)));vol.append(float(w@block['volume_m3']))
        mx.append(float(w@block['x']));ms.append(float(w@block['s']))
        if retain_nodes:kept.append(block)
    mass=math.fsum(sums);rawV=math.fsum(vol)
    if not 0<mass<=1+1e-10:raise ValueError('JOINT_MASS_OUTSIDE_PROBABILITY_DOMAIN')
    npanels=sum(len(b['panels']) for b in part['branches'])
    result=dict(nd=nd,ns_per_active_subinterval_component_branch=ns,nc=nc,nodes=count,
        effective_s_nodes=ns*npanels,active_panels=npanels,maximum_block_nodes=max_block,
        retained_number_mass=mass,raw_volume_moment_m3=rawV,mean_volume_m3=rawV/mass,
        total_number=1e-18/(rawV/mass),weight_sum=math.fsum(v/mass for v in sums),
        mean_ln_d=math.fsum(mx)/mass,mean_s=math.fsum(ms)/mass,
        one_joint_normalization=True,analytic_constant_substituted_for_weight_sum=False,
        magnetic_observable_qualified=False)
    if retain_nodes:
        for key in kept[0]:result[key]=np.concatenate([b[key] for b in kept])
        result['number_weights']=result['raw_number_weights']/mass
    return result

def scalar_reference(spec):
    t=spec['theta']
    if t['kappa']!=0:raise ValueError('REFERENCE_ONLY_CURRENT_ISOTROPIC_ATLAS')
    sx=t['sigma_ln_d']*math.sqrt(1-t['rho_latent']**2);out=[]
    table=inherited.json.loads((inherited.BASE/'sources/SD_TABLE_EXTRACT.json').read_text())
    breaks=np.log(np.array(table['x_Hard DOWN'][0][1:],float))
    for power in [0,3]:
        total=0.;err=0.;records=[]
        for i,comp in enumerate(spec['components']):
            for sign,f in [(1,t['f_prolate']),(-1,1-t['f_prolate'])]:
                if f==0 or comp['weight']==0:continue
                center=sign*(t['shape_center_abs_s']+comp['delta_s']);sd=t['sigma_s']
                a,b=(0.,inherited.SMAX) if sign>0 else (-inherited.SMAX,0.)
                logsemi=float(log_ndtr(sign*center/sd))
                def fun(s):
                    mu=t['mu_ln_d']+comp['delta_ln_d']+t['rho_latent']*t['sigma_ln_d']/sd*(s-center)
                    aa=(math.log(5.)-mu-power*sx*sx)/sx
                    bb=(math.log(inherited.sd_limit(s))-mu-power*sx*sx)/sx
                    lm,_=interval(aa,bb)
                    lp=-.5*((s-center)/sd)**2-math.log(sd*math.sqrt(2*math.pi))-logsemi
                    return math.exp(lp+power*mu+.5*power*power*sx*sx+lm)
                known=sorted(set(float(v) for v in breaks if a<v<b))
                val,e,info=quad(fun,a,b,epsabs=1e-10 if power==0 else 1e-5,epsrel=1e-10,limit=200,points=known,full_output=1)[:3]
                factor=comp['weight']*f;total+=factor*val;err+=factor*e
                records.append(dict(component=i,sign=sign,integral=val,estimated_error=e,subdivisions=int(info['last']),evaluations=int(info['neval']),known_breaks=known))
        if power==3:total*=math.pi/6*1e-27;err*=math.pi/6*1e-27
        out.append(dict(power=power,value=total,estimated_error=err,branches=records))
    mass,vol=out
    return dict(mass=mass['value'],raw_volume_moment_m3=vol['value'],mean_volume_m3=vol['value']/mass['value'],
        estimated_mass_error=mass['estimated_error'],estimated_raw_volume_error_m3=vol['estimated_error'],
        estimated_mean_volume_relative_error=vol['estimated_error']/vol['value']+mass['estimated_error']/mass['value'],details=out,
        scalar_simplification='current kappa=0 and inherited SD envelope independent of c only')

def assemble(curves,weights,volumes,scale=inherited.S):
    """curves: unit moment Ms*R*u, not normalized grain curves; number weights."""
    w=np.asarray(weights,float);v=np.asarray(volumes,float)
    if abs(w.sum()-1)>1e-12 or np.min(w)<0:raise ValueError('INVALID_NUMBER_MEASURE')
    N=1e-18/float(w@v)
    return N*np.einsum('i,i,ijk->jk',w,v,np.asarray(curves))/scale
