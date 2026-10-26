import unittest
import numpy as np
from interpolation_diagnostics import cubic, midpoints, duration_diagnostic, error_metrics, tensor_cubic

class InterpolationDiagnosticTests(unittest.TestCase):
    def test_local_tensor_axis_order_and_new_thermal_intervals(self):
        T=np.array([192.5,195.,197.5,200.]);ell=np.linspace(0,15,121)
        f=lambda t,e: np.stack([(t-195.)**3+e**2, e**3-(t-195.), np.zeros_like(e)],axis=-1)[:,None,:]
        y=np.array([f(t,ell) for t in T]);q=midpoints(ell)
        np.testing.assert_allclose(tensor_cubic(T,ell,y,196.25,q),f(196.25,q),atol=2e-11,rtol=2e-14)
        Tr=np.sort(np.r_[T,196.25]);yr=np.array([f(t,ell) for t in Tr])
        for t in [195.625,196.875]:
            self.assertNotIn(t,Tr)
            np.testing.assert_allclose(tensor_cubic(Tr,ell,yr,t,q),f(t,q),atol=2e-11,rtol=2e-14)
    def test_vector_cubic_reproduction_and_domain_endpoints(self):
        x=np.linspace(0,15,121);q=np.r_[0,midpoints(x),15]
        f=lambda t: np.stack([t**3-2*t, 3*t**2+1, t*0+7],axis=-1)[:,None,:]
        np.testing.assert_allclose(cubic(x,f(x),q),f(q),atol=2e-11,rtol=2e-14)
    def test_refined_intervals_use_new_controls_and_keep_reference_unknown(self):
        x=np.linspace(0,15,121);m=midpoints(x);q=midpoints(np.sort(np.r_[x,m]))
        f=lambda t: np.stack([np.exp(-10**(t-7)),t*0,t*0],axis=-1)[:,None,:]
        d=duration_diagnostic(x,f(x),f(m),f(q))
        self.assertEqual(d['new_interval_control_count'],240)
        self.assertFalse(set(q)&set(np.r_[x,m]))
        self.assertIsNone(d['forward_reference_uncertainty'])
        self.assertIsNone(d['base']['combined_error_bound'])
        self.assertFalse(d['qualified'])
        self.assertLess(d['refined']['maximum_absolute_L2'],d['base']['maximum_absolute_L2'])
    def test_no_extrapolation_duplicate_axis_or_nonfinite_shortcut(self):
        for x,y,q in [([0,1,1,2],np.zeros((4,1,3)),[.5]),
                       ([0,1,2,3],np.zeros((4,1,3)),[-.1]),
                       ([0,1,2,3],np.full((4,1,3),np.nan),[.5])]:
            with self.subTest(x=x,q=q),self.assertRaises(ValueError):cubic(x,y,q)
    def test_reading_norm_and_noise_scaling_are_explicit(self):
        d=np.zeros((2,12,3));d[1,3]=[3,4,0]
        s=error_metrics(d)
        self.assertEqual(s['worst_control_index'],1)
        self.assertEqual(s['maximum_absolute_L2'],5)
        a=s['at_four_noises']
        self.assertEqual(a['0.0025']['maximum_history_L2'],2*a['0.005']['maximum_history_L2'])
        self.assertEqual(a['0.0025']['per_reading_maximum_L2'][0],0)
        self.assertIsNone(s['absolute_forward_error_bound'])
