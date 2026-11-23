"""Near-endpoint critical points retain strictly interior, separated anchors."""
import unittest
import numpy as np
from magthermo.lateral_anchors import extend_offsets


class AnchorTests(unittest.TestCase):
    def test_saved_local100_near_endpoint_has_valid_new_anchors(self):
        f=.9996102711534778;tol=1e-6/5
        values=extend_offsets([.1,.02,.004],30.,np.array([5e-5,0.,0.]),35.,np.array([5e-5,0.,0.]),f,tol)
        self.assertEqual(values[:3],[.1,.02,.004])
        new=values[3:]
        self.assertGreater(len(new),0)
        for offset in new:
            self.assertTrue(0<f-offset/5<f+offset/5<1)
            self.assertGreater(offset/5,4*tol)

    def test_unrepresentable_margin_stays_unresolved(self):
        original=[.1,.02,.004]
        values=extend_offsets(original,30.,np.zeros(3),35.,np.zeros(3),1.-1e-10,1e-6/5)
        self.assertEqual(values,original)


if __name__=='__main__':unittest.main(verbosity=2)
