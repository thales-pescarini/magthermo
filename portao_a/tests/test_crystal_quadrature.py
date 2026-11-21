"""Spherical-area conservation in the crystal quadrature."""
import math
import unittest
from portao_a.quadrature import initial, refine, validate, AREA


class CrystalQuadratureTests(unittest.TestCase):
    def test_child_replacement_preserves_complete_area(self):
        c4=initial(1)
        c16=initial(2)
        c64=refine(c16,[v['cell_id'] for v in c16])
        self.assertEqual(len(c64),64)
        self.assertTrue(validate(c64))
        mixed=refine(c16,[c16[0]['cell_id'],c16[-1]['cell_id']])
        self.assertEqual(len(mixed),22)
        self.assertAlmostEqual(math.fsum(v['area_sr'] for v in mixed),AREA,places=13)
        self.assertAlmostEqual(math.fsum(v['weight'] for v in c4),1.,places=13)

