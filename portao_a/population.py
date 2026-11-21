"""Frozen correlated MF density and positive joint hat weights on (s,z)."""
from pathlib import Path
from portao_a.common import read
from environment import size_table
import math
import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad

LOG5 = math.log(5.)
SMAX = math.log(2.5)
ROOT2PI = math.sqrt(2*math.pi)


class MF:
    def __init__(self):
        self.spec = read(Path(__file__).resolve().parents[1] / 'config/mf_population.json')
        t = self.spec['theta']
        self.mx, self.sx = t['mu_ln_d'], t['sigma_ln_d']
        self.ms, self.ss = t['shape_center_abs_s'], t['sigma_s']
        self.rho = t['rho_latent']
        assert t['f_prolate'] == 1. and t['kappa'] == 0.
        assert self.spec['components'] == [dict(weight=1.,delta_ln_d=0.,delta_s=0.)]
        self.branch_norm = float(ndtr(self.ms/self.ss))
        self.conditional_sd = self.sx*math.sqrt(1-self.rho**2)
        self.parent_third = (math.exp(3*self.mx+4.5*self.sx**2)*
            float(ndtr((self.ms+3*self.rho*self.sx*self.ss)/self.ss))/self.branch_norm)
        sd = size_table()
        self.sd_x = np.log(np.asarray(sd['x_Hard DOWN'][0][1:], dtype=np.float64))
        self.sd_y = np.minimum(np.asarray(sd['x_Hard DOWN'][1][1:],dtype=np.float64),
                               np.asarray(sd['x_Easy DOWN'][1][1:],dtype=np.float64))
        assert np.all(np.diff(self.sd_x)>0)

    def sd_limit(self, s):
        if not self.sd_x[0] <= s <= self.sd_x[-1]:
            raise ValueError('NO_SD_TABLE_EXTRAPOLATION')
        return float(np.interp(s,self.sd_x,self.sd_y)-5.)

    def joint(self, x, s):
        x = np.asarray(x,dtype=np.float64)
        if s < 0:
            return np.zeros_like(x)
        mean = self.mx+self.rho*self.sx/self.ss*(s-self.ms)
        return (np.exp(-.5*((s-self.ms)/self.ss)**2)/(self.ss*ROOT2PI*self.branch_norm)*
                np.exp(-.5*((x-mean)/self.conditional_sd)**2)/(self.conditional_sd*ROOT2PI))

    def tilted_interval(self, s, lower, upper, power):
        mean = self.mx+self.rho*self.sx/self.ss*(s-self.ms)
        sd = self.conditional_sd
        lo, hi = (lower-mean-power*sd*sd)/sd, (upper-mean-power*sd*sd)/sd
        probability = float(ndtr(-lo)-ndtr(-hi) if lo>0 else ndtr(hi)-ndtr(lo))
        if probability < 0:
            raise ArithmeticError('NEGATIVE_INTERVAL_PROBABILITY')
        amplitude = math.exp(-.5*((s-self.ms)/self.ss)**2-
                             math.log(self.ss*ROOT2PI)+power*mean+.5*power*power*sd*sd)
        return amplitude*probability/self.branch_norm

    def integrate(self, lower_s, upper_s, bounds, power=3):
        denominator = self.parent_third if power == 3 else 1.
        points = [float(v) for v in self.sd_x if lower_s<v<upper_s]
        kwargs = dict(points=points) if math.isfinite(upper_s) else {}
        def integrand(s):
            a,b = bounds(s)
            return self.tilted_interval(s,a,b,power)/denominator
        value,error = quad(integrand,lower_s,upper_s,epsabs=2e-12,
                           epsrel=2e-10,limit=300,**kwargs)
        return dict(value=float(value),quad_reported_error=float(error))

    def partition(self, power):
        definitions = {
            'below_5': (0.,np.inf,lambda s:(-np.inf,LOG5)),
            'shape_out_after_floor': (SMAX,np.inf,lambda s:(LOG5,np.inf)),
            'above_SD_after_shape_floor': (0.,SMAX,lambda s:(math.log(self.sd_limit(s)),np.inf)),
            'retained_SD': (0.,SMAX,lambda s:(LOG5,math.log(self.sd_limit(s))))}
        pieces = {name:self.integrate(a,b,bounds,power) for name,(a,b,bounds) in definitions.items()}
        return dict(portions=pieces,sum_residual=math.fsum(p['value'] for p in pieces.values())-1.,
                    absolute_certified_bound=None)

    def diameter_nodes(self, s, count):
        upper = self.sd_limit(s)
        if upper<=5.:
            raise ValueError('EMPTY_SD_DOMAIN')
        d = np.exp(np.linspace(LOG5,math.log(upper),count))
        d[0],d[-1] = 5.,upper
        return d


def hat_values(nodes, values):
    nodes = np.asarray(nodes,dtype=np.float64)
    values = np.asarray(values,dtype=np.float64)
    if np.any(np.diff(nodes)<=0):
        raise ValueError('STRICT_NODES_REQUIRED')
    flat = values.ravel()
    if np.any((flat<nodes[0]) | (flat>nodes[-1])):
        raise ValueError('HAT_EXTRAPOLATION_FORBIDDEN')
    i = np.clip(np.searchsorted(nodes,flat,side='right')-1,0,len(nodes)-2)
    t = (flat-nodes[i])/(nodes[i+1]-nodes[i])
    answer = np.zeros((len(flat),len(nodes)),dtype=np.float64)
    answer[np.arange(len(flat)),i] = 1-t
    answer[np.arange(len(flat)),i+1] = t
    return answer.reshape((*values.shape,len(nodes)))


def joint_weights(s_nodes, z_nodes, order=16, power=3):
    density = MF()
    s_nodes,z_nodes = np.asarray(s_nodes,dtype=np.float64),np.asarray(z_nodes,dtype=np.float64)
    assert s_nodes[0]==0. and abs(s_nodes[-1]-SMAX)<1e-15
    assert z_nodes[0]==0. and z_nodes[-1]==1.
    assert np.all(np.diff(s_nodes)>0) and np.all(np.diff(z_nodes)>0)
    t,w = np.polynomial.legendre.leggauss(order)
    t,w = (t+1)/2,w/2
    dz = np.diff(z_nodes)
    z = z_nodes[:-1,None]+dz[:,None]*t
    wz = dz[:,None]*w
    result = np.zeros((len(s_nodes),len(z_nodes)),dtype=np.float64)
    denominator = density.parent_third if power==3 else 1.
    for si,(a,b) in enumerate(zip(s_nodes[:-1],s_nodes[1:])):
        breaks = [a]+[v for v in density.sd_x if a<v<b]+[b]
        for left,right in zip(breaks[:-1],breaks[1:]):
            for u,wu in zip(t,w):
                s = float(left+(right-left)*u)
                shat = (s-a)/(b-a)
                span = math.log(density.sd_limit(s))-LOG5
                x = LOG5+span*z
                integrand = density.joint(x,s)*np.exp(power*x)*span/denominator
                ld = np.sum(integrand*wz*(1-t),axis=1)
                rd = np.sum(integrand*wz*t,axis=1)
                diameter_weight = np.zeros(len(z_nodes),dtype=np.float64)
                diameter_weight[:-1] += ld
                diameter_weight[1:] += rd
                scale = (right-left)*wu
                result[si] += scale*(1-shat)*diameter_weight
                result[si+1] += scale*shat*diameter_weight
    assert np.isfinite(result).all() and np.all(result>=0)
    return result
