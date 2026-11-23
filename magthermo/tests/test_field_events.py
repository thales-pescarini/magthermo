"""Critical-point hints retain exact path/model scope and full side checks."""
import unittest
import numpy as np
from magthermo import field_events


class FieldHintTests(unittest.TestCase):
    def catalog(self):
        return [dict(T_C=20.,B_sample_T=np.array([4.2e-5,0.,0.]),
                     field_critical_verified=True,critical=dict(kind='fold_stationary_KKT',fraction=.36))]

    def test_reverse_straight_field_fraction(self):
        hint=field_events.select_hint(self.catalog(),20.,np.array([5e-5,0.,0.]),20.,np.array([3.75e-5,0.,0.]))
        self.assertIsNotNone(hint)
        self.assertAlmostEqual(hint['_field_segment_fraction'],.64,places=13)
        self.assertEqual(hint['critical']['fraction'],.36)

    def test_unverified_other_path_or_other_temperature_not_reused(self):
        c=self.catalog()
        c[0]['field_critical_verified']=False
        self.assertIsNone(field_events.select_hint(c,20.,np.array([5e-5,0,0]),20.,np.array([3.75e-5,0,0])))
        self.assertIsNone(field_events.select_hint(self.catalog(),21.,np.array([5e-5,0,0]),21.,np.array([3.75e-5,0,0])))
        self.assertIsNone(field_events.select_hint(self.catalog(),20.,np.array([5e-5,1e-12,0]),20.,np.array([3.75e-5,1e-12,0])))

    def test_installation_restores_inherited_functions(self):
        before_frame=field_events.sequence._frame
        before_event=field_events.event_module.compute_event_reference
        with field_events.installed():
            self.assertIsNot(field_events.sequence._frame,before_frame)
            self.assertIsNot(field_events.event_module.compute_event_reference,before_event)
        self.assertIs(field_events.sequence._frame,before_frame)
        self.assertIs(field_events.event_module.compute_event_reference,before_event)
        self.assertEqual(len(field_events.SOURCE_MAP),2)


if __name__=='__main__':unittest.main(verbosity=2)
