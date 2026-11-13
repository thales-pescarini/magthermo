"""Global joint T/log10(seconds) inference on declared complete numerical banks."""
from portao_a.common import *
import math
import numpy as np
from scipy.special import logsumexp
from scipy import ndimage

EPS=np.finfo(np.float64).eps
SIGMA=.0025/math.sqrt(2.)


def edges(nodes):
    nodes=np.asarray(nodes,dtype=np.float64)
    if nodes.ndim!=1 or len(nodes)<2 or not np.isfinite(nodes).all() or np.any(np.diff(nodes)<=0):
        raise ValueError('STRICT_FINITE_GRID_REQUIRED')
    return np.r_[nodes[0],(nodes[:-1]+nodes[1:])/2,nodes[-1]]


def hpd(nodes,mass,probability):
    nodes,mass=np.asarray(nodes),np.asarray(mass)
    e=edges(nodes);width=np.diff(e)
    density=mass/width
    order=np.argsort(-density,kind='stable')
    last=int(np.searchsorted(np.cumsum(mass[order]),probability,side='left'))
    last=min(last,len(order)-1)
    # Include every equal-density cell at the cutoff, without favorable ties.
    selected=density>=density[order[last]]
    changes=np.diff(np.r_[False,selected,False].astype(np.int8))
    intervals=[[float(e[a]),float(e[b])] for a,b in
               zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))]
    return dict(probability_requested=probability,mass=float(mass[selected].sum()),
                intervals=intervals,total_width=float(width[selected].sum()),
                convention='highest marginal nodal density on Voronoi cells; equal cutoffs retained')


def marginal_cdf(nodes,mass,value):
    """Integral of the normalized piecewise-linear marginal nodal density."""
    nodes,mass=np.asarray(nodes),np.asarray(mass)
    if value<=nodes[0]:return 0.
    if value>=nodes[-1]:return 1.
    density=mass/np.diff(edges(nodes))
    widths=np.diff(nodes)
    areas=(density[:-1]+density[1:])*widths/2
    i=int(np.searchsorted(nodes,value,side='right')-1)
    delta=value-nodes[i]
    partial=density[i]*delta+(density[i+1]-density[i])*delta*delta/(2*widths[i])
    return float((areas[:i].sum()+partial)/areas.sum())


class GridEstimator:
    def __init__(self,T,ell,curves,sigma=SIGMA,*,protocol='PX12',scale='parental_capacity'):
        self.T=np.asarray(T,dtype=np.float64)
        self.ell=np.asarray(ell,dtype=np.float64)
        self.wT=np.diff(edges(self.T));self.well=np.diff(edges(self.ell))
        self.sigma=np.broadcast_to(np.asarray(sigma,dtype=np.float64),(60,)).copy()
        if not np.isfinite(self.sigma).all() or np.any(self.sigma<=0):
            raise ValueError('POSITIVE_DIAGONAL_NOISE_REQUIRED')
        self.curves=np.asarray(curves,dtype=np.float64)
        if self.curves.shape!=(len(self.T),len(self.ell),60) or not np.isfinite(self.curves).all():
            raise ValueError('COMPLETE_FINITE_GLOBAL_BANK_REQUIRED')
        self.flat=self.curves.reshape(-1,60)
        self.origin=self.flat.mean(axis=0)
        self.standard=(self.flat-self.origin)/self.sigma
        self.norm=np.einsum('ij,ij->i',self.standard,self.standard)
        self.cell_area=np.outer(self.wT,self.well)
        self.log_cell_area=np.log(self.cell_area.ravel())
        self.protocol,self.scale=protocol,scale

    def validate_observations(self,observations,protocol,scale):
        if protocol!=self.protocol or scale!=self.scale:
            raise ValueError('PROTOCOL_OR_NORMALIZATION_MISMATCH')
        y=np.asarray(observations,dtype=np.float64)
        if y.ndim==1:y=y[None,:]
        if y.ndim!=2 or y.shape[1]!=60 or not np.isfinite(y).all():
            raise ValueError('FINITE_PX12_XYZ_60_COMPONENT_OBSERVATIONS_REQUIRED')
        return y

    def fit(self,observations,*,protocol='PX12',scale='parental_capacity',return_chi2=False):
        y=self.validate_observations(observations,protocol,scale)
        if len(y)>256:
            raise ValueError('OBSERVATION_BLOCK_MAX_256')
        standardized=(y-self.origin)/self.sigma
        ynorm=np.einsum('ij,ij->i',standardized,standardized)
        products=standardized@self.standard.T
        chi=ynorm[:,None]+self.norm[None,:]-2*products
        # Matrix products are used for screening. Direct residual sums settle
        # the minimizer and all possible ties despite cancellation in screening.
        error_bound=256*EPS*(ynorm[:,None]+self.norm[None,:]+2*np.abs(products)+1.)
        approximate_min=chi.min(axis=1)
        indices=np.full(len(y),-1,dtype=np.int64)
        counts=np.zeros(len(y),dtype=np.int64)
        minimum=np.empty(len(y),dtype=np.float64)
        ties=[]
        for row in range(len(y)):
            near=np.flatnonzero(chi[row]<=approximate_min[row]+2*float(error_bound[row].max()))
            direct=np.empty(len(near),dtype=np.float64)
            for start in range(0,len(near),16384):
                selected=near[start:start+16384]
                residual=(y[row]-self.flat[selected])/self.sigma
                direct[start:start+len(selected)]=np.einsum('ij,ij->i',residual,residual)
            chi[row,near]=direct
            smallest=float(direct.min())
            tolerance=64*EPS*max(1.,abs(smallest))
            minimizers=near[direct-smallest<=tolerance]
            counts[row]=len(minimizers)
            minimum[row]=smallest
            if len(minimizers)==1:
                indices[row]=minimizers[0]
            ties.append(minimizers)
        result=dict(index=indices,tie_count=counts,minimum_chi2=minimum,minimizers=ties)
        if return_chi2:result['chi2']=chi
        return result

    def coordinates(self,indices):
        indices=np.asarray(indices,dtype=np.int64)
        answer=np.full((*indices.shape,2),np.nan,dtype=np.float64)
        valid=indices>=0
        answer[valid,0]=self.T[indices[valid]//len(self.ell)]
        answer[valid,1]=self.ell[indices[valid]%len(self.ell)]
        return answer

    def posterior(self,observation,*,protocol='PX12',scale='parental_capacity'):
        fit=self.fit(observation,protocol=protocol,scale=scale,return_chi2=True)
        if len(fit['index'])!=1:raise ValueError('ONE_OBSERVATION_FOR_SAVED_POSTERIOR')
        chi=fit['chi2'][0]
        log_mass=-.5*chi+self.log_cell_area
        log_z=float(logsumexp(log_mass))
        mass=np.exp(log_mass-log_z).reshape(len(self.T),len(self.ell))
        return mass,self.describe(mass,fit),fit

    def describe(self,mass,fit=None):
        mass=np.asarray(mass,dtype=np.float64)
        if mass.shape!=self.cell_area.shape or not np.isfinite(mass).all() or np.any(mass<0):
            raise ValueError('INVALID_POSTERIOR_MASS')
        if abs(float(mass.sum())-1.)>1e-10:
            raise ValueError('POSTERIOR_NOT_NORMALIZED')
        mt,ml=mass.sum(axis=1),mass.sum(axis=0)
        et,el=float(mt@self.T),float(ml@self.ell)
        vt=float(mt@((self.T-et)**2));vl=float(ml@((self.ell-el)**2))
        cov=float(np.sum(mass*(self.T[:,None]-et)*(self.ell[None,:]-el)))
        prior=self.cell_area/self.cell_area.sum()
        positive=mass>0
        kl=float(np.sum(mass[positive]*np.log(mass[positive]/prior[positive])))
        density=mass/self.cell_area
        peak=(density>0)&(density==ndimage.maximum_filter(density,size=3,mode='constant',cval=-np.inf))
        _,components=ndimage.label(peak,structure=np.ones((3,3),dtype=np.int8))
        out=dict(domain=[[float(self.T[0]),float(self.T[-1])],[float(self.ell[0]),float(self.ell[-1])]],
            posterior_sum=float(mass.sum()),mean=[et,el],variance=[vt,vl],covariance=cov,
            correlation=cov/math.sqrt(vt*vl) if vt>0 and vl>0 else None,
            HPD_T={str(p):hpd(self.T,mt,p) for p in (.68,.95)},
            HPD_ell={str(p):hpd(self.ell,ml,p) for p in (.68,.95)},
            edge_cell_mass=dict(T_low=float(mt[0]),T_high=float(mt[-1]),ell_low=float(ml[0]),ell_high=float(ml[-1])),
            posterior_to_prior_KL_nats=kl,grid_local_peak_components=int(components),
            peak_definition='connected positive-density plateaus of 3x3 local maxima; zero underflow background excluded; not a physical mode count',
            prior='uniform in T and ell=log10(seconds)',
            boundary_limited_diagnostic='edge cell mass reported without an invented information threshold')
        if fit is not None:
            out['ML_tie_count']=int(fit['tie_count'][0])
            ids=fit['minimizers'][0]
            out['ML_minimizer_indices']=ids.tolist()
            out['ML_minimizer_coordinates']=np.c_[self.T[ids//len(self.ell)],self.ell[ids%len(self.ell)]].tolist()
            out['minimum_chi2']=float(fit['minimum_chi2'][0])
        return out


def cell_transfer(coarse,fine):
    a,b=edges(coarse),edges(fine)
    if a[0]!=b[0] or a[-1]!=b[-1]:
        raise ValueError('COMMON_DOMAIN_REQUIRED')
    overlap=np.maximum(0.,np.minimum(a[1:,None],b[None,1:])-np.maximum(a[:-1,None],b[None,:-1]))
    transfer=overlap/np.diff(b)[None,:]
    np.testing.assert_allclose(transfer.sum(axis=0),1.,atol=1e-13,rtol=0)
    return transfer


def posterior_discretization(coarse_T,coarse_ell,coarse_mass,fine_T,fine_ell,fine_mass):
    reduced=cell_transfer(coarse_T,fine_T)@fine_mass@cell_transfer(coarse_ell,fine_ell).T
    tv=float(np.sum(np.abs(coarse_mass-reduced))/2)
    differences={}
    for label,a,b,cm,fm in [('T',coarse_T,fine_T,np.sum(coarse_mass,axis=1),np.sum(fine_mass,axis=1)),
                           ('ell',coarse_ell,fine_ell,np.sum(coarse_mass,axis=0),np.sum(fine_mass,axis=0))]:
        differences[label]={str(p):abs(hpd(a,cm,p)['total_width']-hpd(b,fm,p)['total_width']) for p in (.68,.95)}
    return dict(total_variation_common_partition=tv,HPD_width_differences=differences,
                passed=tv<=.01 and all(v<=1. for v in differences['T'].values()) and
                       all(v<=.1 for v in differences['ell'].values()),
                meaning='finite comparison on common Voronoi partition; no continuous error bound')
