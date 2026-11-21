"""Existing PX12 lab temperatures need the R2 guard without widening fields."""
import unittest
from unittest.mock import patch
import numpy as np

from portao_a import laboratory_event_domain as domain
from portao_a import field_events, lateral_anchors
import e1_core as core


class ReachedFullCensus(Exception):
    pass


class GuardOnlyExecutor:
    def __init__(self):
        self.event_solutions = {}
        self.pending_critical_hint = dict(T_C=365.2403405328137,
            critical=dict(kind='test_location_only_not_a_numerical_certificate'))
        self.events = []
        self.checkpoint = None
        self.er2_transport_version = 'TEST_GUARD_ROUTING_ONLY'

    def pm(self, T, B):
        return dict(T_C=T, B=np.asarray(B))

    def census_levels(self, T, B):
        return 12, 24


class LaboratoryEventDomainTests(unittest.TestCase):
    def test_only_exact_zero_field_can_extend_the_lab_guard_above350(self):
        zero = np.zeros(3)
        self.assertTrue(domain.anchor_domain(365.1, 365.3, zero, zero))
        self.assertTrue(domain.anchor_domain(20., 550., zero, zero))
        for T in (19.999, 550.001, np.nan):
            self.assertFalse(domain.anchor_domain(T, 365., zero, zero))
        for field in (np.array([1e-20, 0., 0.]), np.array([np.nan, 0., 0.])):
            self.assertFalse(domain.anchor_domain(365., 366., field, zero))
        self.assertFalse(domain.anchor_domain(20., 300., np.array([6e-5, 0., 0.]), zero))

    def test_original_route_retains_its_declared350_guard(self):
        _, event = field_events.build(lateral_anchors)
        executor = GuardOnlyExecutor()
        with patch.object(field_events.event_module.ls, 'enumerate_land', side_effect=ReachedFullCensus) as census:
            with self.assertRaisesRegex(core.Unresolved, 'FINITE_LATERAL_ANCHORS_EXHAUSTED'):
                event(executor, 365., np.zeros(3), 370., np.zeros(3), {})
            census.assert_not_called()
        self.assertTrue(all(a['reason']=='ANCHOR_OUTSIDE_PHYSICAL_RAMP_DOMAIN'
            for a in executor.events[-1]['anchors']))

    def test_opt_in_reaches_full_census_inside_the_existing_lab_protocol(self):
        _, event = field_events.build(domain)
        executor = GuardOnlyExecutor()
        with patch.object(field_events.event_module.ls, 'enumerate_land', side_effect=ReachedFullCensus) as census:
            with self.assertRaises(ReachedFullCensus):
                event(executor, 365., np.zeros(3), 370., np.zeros(3), {})
            pm, levels = census.call_args.args
            self.assertGreater(pm['T_C'], 365.)
            self.assertLess(pm['T_C'], 365.2403405328137)
            np.testing.assert_array_equal(pm['B'], np.zeros(3))
            self.assertEqual(levels, (12, 24))



if __name__ == '__main__':
    unittest.main(verbosity=2)
