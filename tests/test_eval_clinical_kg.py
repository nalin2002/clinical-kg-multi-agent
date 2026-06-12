import json
import tempfile
import unittest
from pathlib import Path

from eval.clinical_kg import evaluate_directories, evaluate_graph, evaluate_pair


def _graph(nodes, edges, subject_id="p1"):
    return {
        "subject_id": subject_id,
        "nodes": nodes,
        "edges": edges,
    }


class ClinicalKGEvalTests(unittest.TestCase):
    def test_graph_metrics_handle_raw_unscored_graph(self):
        graph = _graph(
            [
                {"id": "P", "type": "PATIENT", "normalized_name": "patient"},
                {"id": "D", "type": "DIAGNOSIS", "normalized_name": "sepsis"},
                {"id": "R", "type": "DRG", "normalized_name": "medical drg"},
            ],
            [
                {
                    "source": "P",
                    "target": "D",
                    "relation": "HAS_DIAGNOSIS",
                    "evidence": "diagnosis row",
                },
                {
                    "source": "P",
                    "target": "R",
                    "relation": "HAS_DRG",
                    "evidence": "",
                },
            ],
        )

        metrics = evaluate_graph(graph)

        self.assertEqual(metrics.edge_count, 2)
        self.assertEqual(metrics.score_coverage, 0.0)
        self.assertIsNone(metrics.jepa_quality)
        self.assertEqual(metrics.schema_valid_rate, 0.5)
        self.assertEqual(metrics.evidence_coverage, 0.5)
        self.assertGreater(metrics.structural_quality, 0.0)

    def test_pair_metrics_count_retention_removal_and_new_edge_quality(self):
        input_graph = _graph(
            [
                {"id": "P", "type": "PATIENT", "normalized_name": "patient"},
                {"id": "D", "type": "DIAGNOSIS", "normalized_name": "sepsis"},
                {"id": "R", "type": "DRG", "normalized_name": "medical drg"},
            ],
            [
                {
                    "source": "P",
                    "target": "D",
                    "relation": "HAS_DIAGNOSIS",
                    "evidence": "diagnosis row",
                },
                {
                    "source": "P",
                    "target": "R",
                    "relation": "HAS_DRG",
                    "evidence": "drg row",
                },
            ],
        )
        revised_graph = _graph(
            [
                {"id": "P", "type": "PATIENT", "normalized_name": "patient"},
                {"id": "D", "type": "DIAGNOSIS", "normalized_name": "sepsis"},
                {"id": "M", "type": "MEDICATION", "normalized_name": "ceftriaxone"},
            ],
            [
                {
                    "source_id": "P",
                    "target_id": "D",
                    "type": "HAS_DIAGNOSIS",
                    "evidence": "diagnosis row",
                    "jepa_schema_valid": True,
                    "jepa_score": 0.98,
                    "jepa_flag": "ok",
                },
                {
                    "source_id": "P",
                    "target_id": "M",
                    "type": "TAKES_MEDICATION",
                    "evidence": "prescription row",
                    "jepa_schema_valid": True,
                    "jepa_score": 0.97,
                    "jepa_flag": "ok",
                },
            ],
        )

        _, _, pair = evaluate_pair(input_graph, revised_graph, primary_score="structural")

        self.assertEqual(pair.retained_edges, 1)
        self.assertEqual(pair.removed_edges, 1)
        self.assertEqual(pair.added_edges, 1)
        self.assertEqual(pair.good_input_edge_retention, 1.0)
        self.assertEqual(pair.bad_input_edge_removal, 1.0)
        self.assertEqual(pair.new_edge_quality, 1.0)
        self.assertEqual(pair.core_fact_retention, 1.0)

    def test_evaluate_directories_matches_by_file_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            revised_dir = root / "revised"
            input_dir.mkdir()
            revised_dir.mkdir()
            graph = _graph(
                [
                    {"id": "P", "type": "PATIENT", "normalized_name": "patient"},
                    {"id": "D", "type": "DIAGNOSIS", "normalized_name": "sepsis"},
                ],
                [
                    {
                        "source": "P",
                        "target": "D",
                        "relation": "HAS_DIAGNOSIS",
                        "evidence": "diagnosis row",
                    }
                ],
                subject_id="100",
            )
            (input_dir / "100.json").write_text(json.dumps(graph))
            (revised_dir / "100.json").write_text(json.dumps(graph))

            result = evaluate_directories(input_dir, revised_dir)

            self.assertEqual(result["summary"]["patients_compared"], 1)
            self.assertEqual(result["patients"][0]["subject_id"], "100")


if __name__ == "__main__":
    unittest.main()
