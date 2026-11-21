"""Declared number distributions, Haar quadrature, and observation aggregation."""
from pathlib import Path
import math
import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc
import yaml

BASE = Path(__file__).resolve().parents[1]

def legacy_haar_orientations(n):
    """Preserved coordinates for historical regressions and the 84 sentinels only."""
    levels = {12:(3,2,2), 48:(4,3,4), 192:(8,3,8)}
    na, nz, ng = levels[int(n)]
    zs, ws = np.polynomial.legendre.leggauss(nz)
    records = []
    def rz(a):
        c,s=np.cos(a),np.sin(a)
        return np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    for ia in range(na):
        for iz,(z,wz) in enumerate(zip(zs,ws)):
            beta = np.arccos(z)
            ry = np.array([[z,0,np.sin(beta)],[0,1.,0],[-np.sin(beta),0,z]])
            for ig in range(ng):
                R = rz(2*np.pi*ia/na) @ ry @ rz(2*np.pi*ig/ng)
                records.append({'id':f'H{len(records):03d}','number_fraction':float(wz/(2*na*ng)),
                    'R_fs':R.tolist(),'euler_haar':{'alpha_index':ia,'alpha_count':na,
                    'cos_beta_index':iz,'cos_beta_count':nz,'gamma_index':ig,'gamma_count':ng}})
    assert len(records)==n and abs(sum(o['number_fraction'] for o in records)-1)<1e-14
    return records

SECTOR_LEVELS = {'L1': (4, 2, 1), 'L2': (8, 4, 2),
                 'L3': (12, 6, 3), 'L4': (16, 8, 4)}

def haar_orientations(level):
    """One refining Euler family, with explicit polar and preparation sectors.

    Haar = d alpha/(2 pi) * sin(beta) d beta/2 * d gamma/(2 pi).
    Alpha is periodic trapezoidal. Beta has two half-intervals and gamma
    four quadrants, each using Gauss-Legendre. No moment/curve rescaling.
    Integer 12/48/192 explicitly denotes the preserved historical rule.
    """
    if isinstance(level, (int, np.integer)):
        return legacy_haar_orientations(int(level))
    na, nb, ng = SECTOR_LEVELS[level]
    xb, wb = np.polynomial.legendre.leggauss(nb)
    xg, wg = np.polynomial.legendre.leggauss(ng)
    records = []
    def rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    for ia in range(na):
        for half in range(2):
            for ib, (x, w) in enumerate(zip(xb, wb)):
                beta = (half + (x+1)/2)*np.pi/2
                cb, sb = np.cos(beta), np.sin(beta)
                ry = np.array([[cb,0,sb],[0,1.,0],[-sb,0,cb]])
                for sector in range(4):
                    for ig, (y, v) in enumerate(zip(xg, wg)):
                        gamma = (sector + (y+1)/2)*np.pi/2
                        weight = (1/na)*(w*np.pi/8*sb)*(v/8)
                        R = rz(2*np.pi*ia/na) @ ry @ rz(gamma)
                        records.append({'id':f'{level}_{len(records):04d}',
                            'number_fraction':float(weight), 'R_fs':R.tolist(),
                            'euler_haar':{'family':'sector_beta_gamma_GL_v1',
                                'level':level,'alpha_index':ia,'alpha_count':na,
                                'beta_half':half,'beta_index':ib,'beta_order':nb,
                                'gamma_sector':sector,'gamma_index':ig,'gamma_order':ng}})
    assert len(records) == na*(2*nb)*(4*ng) <= 4096
    return records

from functools import lru_cache

@lru_cache(maxsize=32)
def _hp_legendre(n, dps):
    import mpmath as mp
    with mp.workdps(dps):
        nodes, _ = mp.gauss_quadrature(n, 'legendre')
        return tuple(nodes)

def high_precision_row(R, spec=None):
    import mpmath as mp
    if spec is None:
        # The caller's matrix is authoritative. Never reconstruct another grid.
        return [mp.mpf(float(v)) for v in np.asarray(R)[2]]
    if spec.get('family') == 'sector_beta_gamma_GL_v1':
        xb = _hp_legendre(spec['beta_order'], mp.mp.dps)[spec['beta_index']]
        xg = _hp_legendre(spec['gamma_order'], mp.mp.dps)[spec['gamma_index']]
        beta = (spec['beta_half']+(xb+1)/2)*mp.pi/2
        gamma = (spec['gamma_sector']+(xg+1)/2)*mp.pi/2
        return [-mp.sin(beta)*mp.cos(gamma), mp.sin(beta)*mp.sin(gamma), mp.cos(beta)]
    zvalues = _hp_legendre(spec['cos_beta_count'], mp.mp.dps)
    z = zvalues[spec['cos_beta_index']]
    psi=2*mp.pi*spec['gamma_index']/spec['gamma_count']
    return [-mp.sqrt(1-z*z)*mp.cos(psi),mp.sqrt(1-z*z)*mp.sin(psi),z]

def population_quadrature(n, median_nm, geometric_sd, support_nm):
    if n<1 or median_nm<=0 or geometric_sd<=1 or not 0<support_nm[0]<support_nm[1]:
        raise ValueError('INVALID_DECLARED_POPULATION')
    x,w=np.polynomial.legendre.leggauss(int(n))
    a,b=np.log(support_nm);mu=np.log(median_nm);sig=np.log(geometric_sd)
    x=(a+b)/2+(b-a)/2*x;w=w*(b-a)/2
    mass=ndtr((b-mu)/sig)-ndtr((a-mu)/sig)
    if mass<=0: raise ValueError('TRUNCATED_POPULATION_MASS_UNRESOLVED')
    weights=w*np.exp(-.5*((x-mu)/sig)**2)/(sig*np.sqrt(2*np.pi)*mass)
    discrete_mass=float(weights.sum());weights=weights/discrete_mass
    d=np.exp(x);V=np.pi/6*(d*1e-9)**3
    third=np.exp(3*mu+4.5*sig**2)*(ndtr((b-mu-3*sig**2)/sig)-ndtr((a-mu-3*sig**2)/sig))/mass
    return {'diameter_nm':d,'number_weights':weights,'volume_m3':V,
        'mean_grain_volume_m3':float(weights@V),'analytic_mean_grain_volume_m3':float(np.pi/6*third*1e-27),
        'discrete_mass_before_normalization':discrete_mass,'retained_number_mass':float(mass),
        'median_nm':float(median_nm),'geometric_sd':float(geometric_sd),'support_nm':list(map(float,support_nm))}

def sum_number_mixture(raw_class_moments, class_mean_volumes, number_fractions, S, total_volume=1e-18):
    """Each class raw M refers to total_volume; undo class number scale first."""
    f=np.asarray(number_fractions,float);v=np.asarray(class_mean_volumes,float)
    if np.any(f<0) or abs(f.sum()-1)>1e-12 or np.any(v<=0):raise ValueError('INVALID_NUMBER_MIXTURE')
    mean=float(f@v)
    per_particle=[np.asarray(M)*vi/total_volume for M,vi in zip(raw_class_moments,v)]
    M=total_volume/mean*sum(fi*mi for fi,mi in zip(f,per_particle))
    return M/S, M, {'total_volume_m3':total_volume,'mean_grain_volume_m3':mean,'total_number':total_volume/mean}

def characterization_loglikelihood(c, eta, multiplier):
    # c is a measurement; h(eta) is the candidate's physical population.
    s=np.asarray([.1,.2,.05,.05])*float(multiplier)
    h=np.c_[eta['log_median_nm'],eta['log_log_size_sd'],eta['number_fractions'][:,:2]]
    return -.5*np.sum(((np.asarray(c)-h)/s)**2,axis=1)-np.sum(np.log(s*np.sqrt(2*np.pi)))
