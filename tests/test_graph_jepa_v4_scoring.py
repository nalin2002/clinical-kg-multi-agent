import unittest

from graph_jepa.schema import PatientGraph
from graph_jepa_v4.config import Config
from graph_jepa_v4.score import (
    _apply_schema_guard,
    _candidate_threshold,
    _edge_schema_error,
)


class GraphJEPAv4ScoringTests(unittest.TestCase):
    def test_schema_guard_caps_wrong_direction_typed_edges(self):
        graph = PatientGraph(
            nodes=[
                {"id": "M", "type": "MEDICATION", "text": "warfarin"},
                {"id": "D", "type": "DIAGNOSIS", "text": "anticoagulant use"},
            ],
            edges=[
                {"source_id": "M", "target_id": "D", "type": "TREATED_BY"},
            ],
        )
        scores = [0.98]
        flags = ["ok"]

        _apply_schema_guard(graph, scores, flags, Config())

        self.assertEqual(scores, [0.0])
        self.assertEqual(flags, ["inconsistent"])
        self.assertFalse(graph.edges[0]["jepa_schema_valid"])
        self.assertIn("no schema rule", graph.edges[0]["jepa_schema_error"])

    def test_schema_guard_leaves_unconstrained_cooccurrence_edges(self):
        graph = PatientGraph(
            nodes=[
                {"id": "D", "type": "DIAGNOSIS", "text": "sepsis"},
                {"id": "M", "type": "MEDICATION", "text": "vancomycin"},
            ],
            edges=[
                {"source_id": "D", "target_id": "M", "type": "CO_OCCURS_WITH"},
            ],
        )
        scores = [0.72]
        flags = ["ok"]

        _apply_schema_guard(graph, scores, flags, Config())

        self.assertEqual(scores, [0.72])
        self.assertEqual(flags, ["ok"])
        self.assertTrue(graph.edges[0]["jepa_schema_valid"])

    def test_mimic_candidate_thresholds_are_relation_specific(self):
        cfg = Config()

        self.assertEqual(_candidate_threshold("TREATED_BY", cfg), 0.975)
        self.assertEqual(_candidate_threshold("DIAGNOSED_BY", cfg), 0.985)
        self.assertEqual(_candidate_threshold("INVESTIGATED_BY", cfg), 0.99)
        self.assertEqual(_candidate_threshold("PERFORMED_FOR", cfg), 0.995)
        self.assertEqual(_candidate_threshold("COMPLICATED_BY", cfg), 0.995)
        self.assertEqual(_candidate_threshold("PART_OF_REGIMEN", cfg), 0.995)
        self.assertEqual(_candidate_threshold("HAS_DIAGNOSIS", cfg), 0.96)

    def test_legacy_checkpoint_thresholds_merge_with_mimic_defaults(self):
        cfg = Config.from_dict({
            "score": {
                "candidate_threshold_by_relation": {
                    "INDICATES": 0.91,
                    "CONFIRMS": 0.99,
                },
            },
        })

        self.assertEqual(_candidate_threshold("INDICATES", cfg), 0.95)
        self.assertEqual(_candidate_threshold("CONFIRMS", cfg), 0.99)
        self.assertEqual(_candidate_threshold("TREATED_BY", cfg), 0.975)

    def test_schema_error_identifies_missing_relation_rule(self):
        graph = PatientGraph(
            nodes=[
                {"id": "P", "type": "PATIENT", "text": "patient"},
                {"id": "D", "type": "DIAGNOSIS", "text": "pneumonia"},
            ],
            edges=[
                {"source_id": "P", "target_id": "D", "type": "CONFIRMS"},
            ],
        )

        error = _edge_schema_error(graph, graph.edges[0], Config())

        self.assertIn("no schema rule", error)


if __name__ == "__main__":
    unittest.main()
