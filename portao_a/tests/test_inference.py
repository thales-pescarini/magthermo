"""Known synthetic posteriors, cancellation, ties, priors and common cells."""
import unittest
import numpy as np
from portao_a.inference import GridEstimator,posterior_discretization,edges,marginal_cdf
from portao_a.inference_modes import FIXED_SCALE,L1


class InferenceTests(unittest.TestCase):
    def make(self,T=None,ell=None):
        T=np.array([40.,170.,300.]) if T is None else np.asarray(T,dtype=float)
        ell=np.array([0.,7.5,15.]) if ell is None else np.asarray(ell,dtype=float)
        curves=np.zeros((len(T),len(ell),60))
        curves[:,:,0]=T[:,None]/300
        curves[:,:,1]=ell[None,:]/15
        return GridEstimator(T,ell,curves)

    def test_global_minimum_matches_direct_exhaustive(self):
        estimator=self.make(np.linspace(40,300,9),np.linspace(0,15,7))
        rng=np.random.Generator(np.random.PCG64(2026111302))
        observations=rng.normal(0,.001,(17,60))+estimator.flat[31]
        result=estimator.fit(observations)
        direct=np.sum(((observations[:,None,:]-estimator.flat[None,:,:])/estimator.sigma)**2,axis=2)
        np.testing.assert_array_equal(result['index'],np.argmin(direct,axis=1))
        np.testing.assert_allclose(result['minimum_chi2'],direct.min(axis=1),atol=1e-12,rtol=1e-14)

    def test_large_exact_tie_sets_not_uint8_overflow(self):
        T=np.linspace(40,300,21);ell=np.linspace(0,15,21)
        estimator=GridEstimator(T,ell,np.ones((21,21,60)))
        fit=estimator.fit(np.ones(60))
        self.assertEqual(fit['tie_count'][0],441)
        self.assertEqual(fit['index'][0],-1)
        self.assertEqual(len(fit['minimizers'][0]),441)

    def test_cancellation_does_not_invent_ml_ties(self):
        e=self.make()
        e.curves[1,1]=e.curves[1,0]+2e-10
        e=GridEstimator(e.T,e.ell,e.curves)
        result=e.fit(e.curves[1,1])
        self.assertEqual(result['tie_count'][0],1)
        self.assertEqual(result['index'][0],4)

    def test_L1_matches_direct_weighted_profile_with_one_gain_across_all_readouts(self):
        rng=np.random.default_rng(1034)
        T=np.linspace(40,300,5);ell=np.linspace(0,15,3)
        curves=rng.normal(size=(len(T),len(ell),60))
        curves[0,0]=0.
        sigma=np.linspace(.15,.39,60)
        estimator=GridEstimator(T,ell,curves,sigma)
        observations=2.4*estimator.flat[[2,7,13]]+rng.normal(0,.03,(3,60))
        fit=estimator.fit(observations,amplitude_mode=L1,return_chi2=True)
        templates=estimator.flat/sigma
        whitened=observations/sigma
        product=np.sum(whitened[:,None,:]*templates[None,:,:],axis=2)
        norm=np.sum(templates**2,axis=1)
        gain=np.maximum(0.,np.divide(product,norm,out=np.zeros_like(product),where=norm>0.))
        direct=np.sum(((observations[:,None,:]-gain[:,:,None]*estimator.flat)/sigma)**2,axis=2)
        index=direct.argmin(axis=1)
        np.testing.assert_array_equal(fit['index'],index)
        np.testing.assert_allclose(fit['chi2'],direct,rtol=2e-13,atol=2e-12)
        np.testing.assert_allclose(fit['minimum_chi2'],direct.min(axis=1),rtol=2e-13,atol=2e-12)
        np.testing.assert_allclose(fit['amplitude'],gain[np.arange(len(index)),index],rtol=2e-15)
        fixed=estimator.fit(observations,amplitude_mode=FIXED_SCALE,return_chi2=True)
        default=estimator.fit(observations,return_chi2=True)
        np.testing.assert_array_equal(fixed['chi2'],default['chi2'])
        self.assertNotIn('amplitude',fixed)

    def test_L1_resolves_cancellation_and_preserves_equal_positive_rays(self):
        rng=np.random.default_rng(993)
        truth=rng.normal(size=60)
        curves=np.zeros((2,2,60))
        curves[0,0]=truth
        curves[0,1]=truth+rng.normal(0,1e-10,60)
        curves[1,0]=-.5*truth
        estimator=GridEstimator([40.,300.],[0.,15.],curves)
        mass,description,fit=estimator.posterior(3.*truth,amplitude_mode=L1)
        np.testing.assert_array_equal(fit['index'],[0])
        np.testing.assert_allclose(fit['amplitude'],[3.],rtol=2e-15)
        self.assertAlmostEqual(float(mass.sum()),1.)
        self.assertEqual(description['amplitude_mode'],L1)
        curves[0,1]=.25*truth
        estimator=GridEstimator([40.,300.],[0.,15.],curves)
        tied=estimator.fit(3.*truth,amplitude_mode=L1)
        np.testing.assert_array_equal(tied['minimizers'][0],[0,1])
        self.assertTrue(np.isnan(tied['amplitude'][0]))

    def test_L1_negative_projection_and_zero_template_share_boundary_fit(self):
        curves=np.ones((2,2,60))
        curves[1,1]=0.
        estimator=GridEstimator([40.,300.],[0.,15.],curves,sigma=1.)
        fit=estimator.fit(-np.ones(60),amplitude_mode=L1,return_chi2=True)
        np.testing.assert_array_equal(fit['minimizers'][0],np.arange(4))
        np.testing.assert_array_equal(fit['chi2'],[[60.]*4])
        self.assertTrue(np.isnan(fit['amplitude'][0]))

    def test_uniform_log_time_prior_and_likelihood_constant(self):
        T=np.array([40.,100.,300.]);ell=np.array([0.,1.,15.])
        e=GridEstimator(T,ell,np.zeros((3,3,60)))
        mass,description,_=e.posterior(np.zeros(60))
        expected=np.outer(np.diff(edges(T)),np.diff(edges(ell)))/(260*15)
        np.testing.assert_allclose(mass,expected,atol=1e-15,rtol=0)
        shifted,_,_=e.posterior(np.ones(60)*.01)
        np.testing.assert_allclose(mass,shifted,atol=2e-14,rtol=0)
        self.assertAlmostEqual(marginal_cdf(ell,mass.sum(axis=0),7.5),.5,places=13)
        self.assertLess(abs(description['posterior_to_prior_KL_nats']),1e-14)

    def test_positive_variances_with_underflowed_product_keep_correlation(self):
        e = self.make()
        mass = np.zeros((3, 3))
        mass[0, 0] = 1.
        mass[2, 2] = 1e-200
        description = e.describe(mass)
        vt, vl = description['variance']
        self.assertGreater(vt, 0.)
        self.assertGreater(vl, 0.)
        self.assertEqual(vt * vl, 0.)
        self.assertAlmostEqual(description['correlation'], 1., places=14)
        mass[2, 2] = 0.
        self.assertIsNone(e.describe(mass)['correlation'])

    def test_two_separate_modes_retained(self):
        e=self.make(np.linspace(40,300,9),np.linspace(0,15,9))
        curves=np.ones_like(e.curves)*.1
        curves[2,2]=0;curves[6,6]=0
        e=GridEstimator(e.T,e.ell,curves)
        mass,description,fit=e.posterior(np.zeros(60))
        self.assertEqual(fit['tie_count'][0],2)
        self.assertAlmostEqual(float(mass[2,2]),.5,places=12)
        self.assertAlmostEqual(float(mass[6,6]),.5,places=12)
        self.assertEqual(description['grid_local_peak_components'],2)

    def test_common_partition_uniform_prior_zero_tv(self):
        a,b=np.linspace(40,300,5),np.linspace(0,15,7)
        x,y=np.linspace(40,300,9),np.linspace(0,15,13)
        cm=np.outer(np.diff(edges(a)),np.diff(edges(b)))/(260*15)
        fm=np.outer(np.diff(edges(x)),np.diff(edges(y)))/(260*15)
        result=posterior_discretization(a,b,cm,x,y,fm)
        self.assertLess(result['total_variation_common_partition'],1e-14)

    def test_reject_wrong_scale_protocol_shape_and_missing_bank(self):
        e=self.make()
        for values,kw in [(np.zeros(60),dict(protocol='PX24')),(np.zeros(60),dict(scale='renormalized_SD')),
                          (np.zeros(59),{}),(np.full(60,np.nan),{})]:
            with self.assertRaises(ValueError):e.fit(values,**kw)
        bad=np.array(e.curves,copy=True);bad[0,0,0]=np.nan
        with self.assertRaises(ValueError):GridEstimator(e.T,e.ell,bad)


if __name__=='__main__':unittest.main(verbosity=2)
