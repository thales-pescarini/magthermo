"""Normalization and missing mass controls with analytically constant responses."""
import unittest
import numpy as np
from magthermo.integration import PositiveAccumulator


class IntegrationTests(unittest.TestCase):
    def test_full_constant_response_keeps_parental_capacity(self):
        a=PositiveAccumulator((20,3))
        response=np.zeros((20,3));response[:,0]=1.
        for key,weight in [('one',.2),('two',.3),('three',.3087943852735576)]:
            a.add(key,weight,response)
        self.assertTrue(a.complete)
        np.testing.assert_allclose(a.total()[:,0],.8087943852735576,rtol=0,atol=2e-16)
        self.assertLess(float(a.total().max()),1.)

    def test_missing_is_unknown_and_does_not_renormalize(self):
        a=PositiveAccumulator((1,3))
        a.add('complete',.7,np.array([[1.,0.,0.]]))
        a.add('failed',.1,None)
        self.assertFalse(a.complete)
        self.assertTrue(np.isnan(a.total()).all())
        self.assertAlmostEqual(a.known[0,0],.7)
        self.assertAlmostEqual(a.missing_capacity,.1)

    def test_retry_cannot_double_count_rotation(self):
        a=PositiveAccumulator((1,))
        a.add('R0',.5,np.array([1.]))
        with self.assertRaises(ValueError):a.add('R0',.5,np.array([1.]))

    def test_positive_missing_tail_not_rounded_to_complete(self):
        a=PositiveAccumulator((1,))
        a.add_known('main',np.array([.8]),.8,0.)
        a.add_known('tail',np.array([0.]),0.,2.8e-21)
        self.assertGreater(a.missing_capacity,0.)
        self.assertFalse(a.complete)


if __name__=='__main__':unittest.main(verbosity=2)
