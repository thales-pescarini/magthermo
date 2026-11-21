"""Known stationary sphere and analytic master-equation limits, no bank truths."""
import unittest
import math
import numpy as np
from scipy.linalg import expm
import e1_core as core
import r1_landscape as landscape
from wp1c_spectral import Spectral,spectral_F


class PhysicsLimitTests(unittest.TestCase):
    def test_spherical_zero_field_exact_cubic_stationary_counts(self):
        pm=core.PointModel(0.,[1.,0.,0.],20.,[0.,0.,0.])
        result=landscape.census(pm,6)
        self.assertEqual(result['counts'],{0:8,1:12,2:6})
        self.assertEqual(result['euler'],2)
        self.assertTrue(result['all_morse'])
        for point in result['points']:
            squared=np.sort(np.asarray(point['m'])**2)
            expected=([1/3,1/3,1/3] if point['index']==0 else
                      [0.,.5,.5] if point['index']==1 else [0.,0.,1.])
            np.testing.assert_allclose(squared,expected,atol=1e-10,rtol=0)

    def test_two_state_dimensional_rate_and_time_limits(self):
        T=20.;d=10.;rate=.25
        volume=math.pi/6*(d*1e-9)**3
        barrier=core.KB*(T+273.15)*math.log(core.NU0/rate)/volume
        graph=dict(status='COMPUTED_VALID',identity={'Ms':1.},
            minima=[dict(point_id='left',m=[1.,0.,0.],E=0.),dict(point_id='right',m=[-1.,0.,0.],E=0.)],
            channels=[dict(saddle_id='saddle',endpoints=[0,1],saddle_E=barrier,barrier_density=[barrier,barrier])])
        sp=spectral_F(graph,d,T)
        selected=sp.select(np.array([[1.,-1.],[0.,0.],[0.,0.]]))
        for seconds in (0.,5.,1e15):
            decay=math.exp(-2*rate*seconds)
            expected=np.array([[1+decay,1-decay],[1-decay,1+decay]])/2
            for model in ('F','N1'):
                U,info=sp.propagator(seconds,model,selected)
                np.testing.assert_allclose(U,expected,atol=2e-14,rtol=0)
                np.testing.assert_allclose(U.sum(axis=0),1.,atol=2e-15,rtol=0)
                self.assertGreaterEqual(float(U.min()),-1e-15)

    def test_full_three_state_propagator_against_independent_expm(self):
        Q=np.array([[-1.,1.,0.],[1.,-2.,1.],[0.,1.,-1.]])
        sp=Spectral(Q,[1.,1.,1.],80)
        selected=sp.select(np.array([[1.,0.,-.5],[0.,1.,0.],[0.,0.,.5]]))
        full,info=sp.propagator(.7,'F',selected)
        one,_=sp.propagator(.7,'N1',selected)
        np.testing.assert_allclose(full,expm(Q*.7),atol=5e-15,rtol=0)
        self.assertTrue(info['all_modes'])
        self.assertGreater(float(np.max(np.abs(full-one))),1e-3)


if __name__=='__main__':unittest.main(verbosity=2)
