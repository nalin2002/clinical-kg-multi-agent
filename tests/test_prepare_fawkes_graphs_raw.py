import unittest

import pandas as pd

from scripts.prepare_fawkes_graphs_raw import (
    build_edge,
    build_node,
    deduplicate_edges,
)


class PrepareFawkesGraphsRawTests(unittest.TestCase):
    def test_presenting_diagnosis_keeps_raw_node_type(self):
        node = build_node(
            pd.Series(
                {
                    "subject_id": 1,
                    "hadm_id": 2,
                    "node_id": "N_001",
                    "type": "DIAGNOSIS",
                    "name": "Adult failure to thrive",
                    "normalized_name": "adult failure to thrive",
                    "presenting": True,
                }
            )
        )

        self.assertEqual(node["type"], "DIAGNOSIS")
        self.assertEqual(node["mimic_type"], "DIAGNOSIS")
        self.assertTrue(node["presenting"])
        self.assertNotIn("schema_repair", node)

    def test_edge_keeps_raw_relation_direction_and_metadata(self):
        edge = build_edge(
            pd.Series(
                {
                    "subject_id": 1,
                    "hadm_id": 2,
                    "edge_idx": 7,
                    "source_id": "N_010",
                    "target_id": "N_001",
                    "source_type": "MEDICATION",
                    "target_type": "DIAGNOSIS",
                    "relation": "MANAGED_FOR",
                    "confidence": 0.82,
                }
            )
        )

        self.assertEqual(edge["source_id"], "N_010")
        self.assertEqual(edge["target_id"], "N_001")
        self.assertEqual(edge["type"], "MANAGED_FOR")
        self.assertEqual(edge["relation"], "MANAGED_FOR")
        self.assertEqual(edge["edge_idx"], 7)
        self.assertNotIn("schema_repair", edge)

    def test_deduplicates_edges_by_source_target_and_relation(self):
        edges = [
            {
                "source_id": "N_001",
                "target_id": "N_002",
                "type": "MANAGED_FOR",
                "confidence": 0.82,
            },
            {
                "source_id": "N_001",
                "target_id": "N_002",
                "type": "MANAGED_FOR",
                "confidence": 0.91,
            },
            {
                "source_id": "N_001",
                "target_id": "N_002",
                "type": "INDICATES",
                "confidence": 0.75,
            },
        ]

        deduplicated = deduplicate_edges(edges)

        self.assertEqual(deduplicated, [edges[0], edges[2]])


if __name__ == "__main__":
    unittest.main()
