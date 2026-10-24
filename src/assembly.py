"""Number measure, one volume factor, full-discrete denominator, partial sum."""
import math
import numpy as np
S=4.884610123479026e-13
SIGMA=.0025/math.sqrt(2)
VTOTAL=1e-18
def partial_kernel(K,number_weights,haar_weights,volumes,full_volume_denominator):
    k=np.asarray(K,float);w=np.asarray(number_weights,float);a=np.asarray(haar_weights,float);v=np.asarray(volumes,float)
    if len(k)!=len(w) or len(w)!=len(v) or len(a)!=len(w) or np.min(w)<0 or np.min(a)<0 or np.min(v)<=0 or not full_volume_denominator>0:raise ValueError('INVALID_PARTIAL_NUMBER_MEASURE')
    if not np.all(np.isfinite(k)):raise ValueError('NONFINITE_KERNEL')
    terms=(w*a*v)[:,None,None]*k
    numerator=np.array([[math.fsum(terms[:,j,c]) for c in range(3)] for j in range(k.shape[1])])
    return VTOTAL/S*numerator/full_volume_denominator
def absent_bound(Ms20,missing_volume_fraction,nread=12):
    return math.sqrt(nread)*Ms20*VTOTAL/(S*SIGMA)*missing_volume_fraction
def algebra_checks():
    w=np.array([.2,.3,.5]);v=np.array([1.,2.,4.])*1e-23;a=np.ones(3);den=float(w@v)
    K=np.tile(np.array([[1.,-2.,3.],[0.,.5,0.]]),(3,1,1));same=partial_kernel(K,w,a,v,den);expected=VTOTAL/S*K[0]
    opposite=partial_kernel(np.array([K[0],-K[0]]),[.5,.5],[1.,1.],[2e-23,2e-23],2e-23)
    split=partial_kernel(np.array([K[0],K[0],K[1],K[2]]),[.1,.1,.3,.5],[1.]*4,[v[0],v[0],v[1],v[2]],den)
    return dict(identical_kernel_error=float(np.max(abs(same-expected))),opposite_kernel_error=float(np.max(abs(opposite))),split_weight_error=float(np.max(abs(split-same))),
        pass_all=bool(np.max(abs(same-expected))<1e-20 and np.max(abs(opposite))<1e-20 and np.max(abs(split-same))<1e-20),
        units='K A/m; Vtotal/S m/A; weighted V numerator and full discrete V denominator cancel once',no_magnetic_histories=True)
