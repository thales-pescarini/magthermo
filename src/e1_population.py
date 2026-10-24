"""E1 number density, finite quadrature and prospectively fixed sources.

This module does not claim that TEM proxies measure a 3D magnetic population.
"""
from pathlib import Path
import math
import json
import numpy as np
from scipy.special import ndtr, log_ndtr
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1]
BASE=Path(__import__('json').loads((ROOT/'config/BASE_DEPENDENCY.json').read_text())['root'])
SMAX=math.log(2.5)
DMIN=5.
TOTAL_VOLUME=1e-18
S=4.884610123479026e-13

def spherical_area(a,b,c):
    return 2*math.atan2(abs(float(np.linalg.det(np.array([a,b,c])))),1+a@b+b@c+c@a)

def triangle(n):
    order={6:2,15:4,28:6}[int(n)]
    A=np.array([1.,0,0]);B=np.array([1.,1,0])/math.sqrt(2);C=np.ones(3)/math.sqrt(3)
    positions={};weights={}
    for i in range(order+1):
        for j in range(order+1-i):
            v=((order-i-j)*A+i*B+j*C)/order;v/=np.linalg.norm(v)
            positions[i,j]=v;weights[i,j]=0.
    def add(keys):
        area=spherical_area(*(positions[k] for k in keys))
        for k in keys:weights[k]+=area/3
    for i in range(order):
        for j in range(order-i):
            add([(i,j),(i+1,j),(i,j+1)])
            if i+j<order-1:add([(i+1,j),(i+1,j+1),(i,j+1)])
    keys=list(positions)
    return np.array([positions[k] for k in keys]),np.array([weights[k]/(math.pi/12) for k in keys])

def shape_rule(n):
    assert n in [8,16,32]
    x,w=np.polynomial.legendre.leggauss(n//2)
    positive=[];weight=[]
    for a,b in [(0.,.3),(.3,SMAX)]:
        positive.extend((a+b)/2+(b-a)/2*x);weight.extend(w*(b-a)/2)
    return np.r_[-np.asarray(positive)[::-1],positive],np.r_[weight[::-1],weight]

def sd_limit(s,alignment='isotropic'):
    table=json.loads((BASE/'sources/SD_TABLE_EXTRACT.json').read_text())
    log_aspect=np.log(np.asarray(table['x_Hard DOWN'][0][1:],float))
    hard=np.asarray(table['x_Hard DOWN'][1][1:],float)
    easy=np.asarray(table['x_Easy DOWN'][1][1:],float)
    # Conservative design interpolation; not a bound for unmeasured c or high T.
    values=hard if alignment=='hard_100' else easy if alignment=='easy_111' else np.minimum(hard,easy)
    return float(np.interp(s,log_aspect,values)-5.)

def orientation_density(c,kappa):
    high,hw=triangle(28)
    h=np.sum(c**4,axis=1)
    highh=np.sum(high**4,axis=1)
    normalizer=float(hw@np.exp(kappa*(highh-1/3)))
    return np.exp(kappa*(h-1/3))/normalizer

def nodes(spec,nd=33,ns=8,nc=6,alignment='isotropic',retain_nodes=False):
    """Global joint truncation. No conditional re-normalization at each s,c.

    p_d(d|s)=Normal(ln d; mu(s),sigma)/d, dd=d d(ln d).
    The implementation multiplies by d explicitly; it cancels 1/d once.
    """
    theta=spec['theta'];mix=spec['components'];s,ws=shape_rule(ns)
    if alignment=='isotropic':c,wc=triangle(nc);wc=wc*orientation_density(c,theta['kappa'])
    else:c=np.array([[1.,0,0]]) if alignment=='hard_100' else np.array([[1.,1,1]])/math.sqrt(3);wc=np.ones(1)
    xd,wd=np.polynomial.legendre.leggauss(nd)
    logs=[];shapes=[];directions=[];weights=[]
    sigd=theta['sigma_ln_d'];sigs=theta['sigma_s'];rho=theta['rho_latent']
    conditional_sd=sigd*math.sqrt(1-rho*rho)
    for sk,wsk in zip(s,ws):
        sign=1 if sk>0 else -1
        branch_fraction=theta['f_prolate'] if sk>0 else 1-theta['f_prolate']
        upper=sd_limit(float(sk),alignment)
        a,b=math.log(DMIN),math.log(upper)
        x=(a+b)/2+(b-a)/2*xd;wx=wd*(b-a)/2;d=np.exp(x)
        density=np.zeros(nd)
        for component in mix:
            center=sign*(theta['shape_center_abs_s']+component['delta_s'])
            log_sign_mass=float(log_ndtr(sign*center/sigs))
            ps=math.exp(-.5*((sk-center)/sigs)**2-math.log(sigs*math.sqrt(2*math.pi))-log_sign_mass)
            mu=theta['mu_ln_d']+component['delta_ln_d']+rho*sigd/sigs*(sk-center)
            pd=np.exp(-.5*((x-mu)/conditional_sd)**2)/(conditional_sd*math.sqrt(2*math.pi)*d)
            density+=component['weight']*branch_fraction*ps*pd*d
        for ck,wck in zip(c,wc):
            logs.extend(x);shapes.extend(np.full(nd,sk));directions.extend(np.tile(ck,(nd,1)))
            weights.extend(wx*wsk*wck*density)
    x=np.asarray(logs);ss=np.asarray(shapes);cc=np.asarray(directions);raw=np.asarray(weights)
    mass=float(raw.sum())
    if mass<=0 or not math.isfinite(mass):raise ValueError('JOINT_MASS_UNRESOLVED')
    w=raw/mass;d=np.exp(x);V=np.pi/6*(d*1e-9)**3
    mean_x=float(w@x);mean_s=float(w@ss)
    covariance=float(w@((x-mean_x)*(ss-mean_s)))
    varx=float(w@((x-mean_x)**2));vars=float(w@((ss-mean_s)**2))
    meanV=float(w@V)
    result=dict(nd=nd,ns_per_branch=ns,ns_total=2*ns,nc=len(c),nodes=len(w),
        retained_number_mass_approx=mass,excluded_number_mass_approx=1-mass,
        quadrature_mass_resolved=bool(0<mass<=1+1e-8),mean_ln_d=mean_x,mean_s=mean_s,
        mean_volume_m3=meanV,total_number=TOTAL_VOLUME/meanV,
        postcut_prolate_number_fraction=float(w@(ss>0)),
        postcut_correlation_ln_d_s=covariance/math.sqrt(varx*vars),
        volume_identity_m3=float((TOTAL_VOLUME/meanV)*(w@V)),
        fraction_number_d_below_35=float(w@(d<35)),
        fraction_volume_d_below_35=float((w*V)@(d<35)/meanV),
        below_35_is_not_an_SP_classification=True,
        internal_orientation_extrapolation_continuum_number_fraction=1. if alignment=='isotropic' else 0.,
        internal_orientation_extrapolation_continuum_volume_fraction=1. if alignment=='isotropic' else 0.,
        high_temperature_SD_validation=False,
        uncertainty_of_quadrature_mass='not bounded rigorously; compare frozen finite levels')
    if retain_nodes:result.update(d_nm=d,s=ss,c=cc,number_weights=w,raw_number_weights=raw,volume_m3=V)
    return result

def prior_nodes(n,seed=6222026,spec=None):
    if n not in [256,1024]:raise ValueError('FINITE_NUISANCE_LEVELS_ONLY')
    u=qmc.Sobol(7,scramble=True,seed=seed).random_base2(int(math.log2(n)))
    bounds=[(math.log(10),math.log(100)),(math.log(math.log(1.02)),math.log(math.log(1.6))),
            (0.,SMAX),(math.log(.005),math.log(.5)),(0.,1.),(-12.,12.),(-.9999,.9999)]
    if spec is not None:
        offsets=[c['delta_s'] for c in spec['components']]
        bounds[2]=(max(0.,-min(offsets)),min(SMAX,SMAX-max(offsets)))
    return np.array([lo+(hi-lo)*u[:,k] for k,(lo,hi) in enumerate(bounds)]).T
