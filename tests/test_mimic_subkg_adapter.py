import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from graph_jepa.data import MimicSubKGGraphBuilder, adapt_mimic_subkg
from graph_jepa.schema import EDGE_TYPE_TO_IDX, NODE_TYPE_TO_IDX
from graph_jepa_v3.data import to_graph_data
from graph_jepa_v3.score import _load_graph_for_scoring


class TinyEncoder:
    dim = 4

    def encode(self, keys):
        return np.ones((len(keys), self.dim), dtype=np.float32)


class MimicSubKGAdapterTests(unittest.TestCase):
    def test_adapts_mimic_subkg_without_dropping_raw_types(self):
        raw = {
            "subject_id": "10000000",
            "nodes": [
                {"id": "P", "type": "PATIENT", "name": "Patient 10000000"},
                {"id": "D", "type": "DIAGNOSIS", "name": "Pneumonia"},
                {"id": "M", "type": "MEDICATION", "name": "ceftriaxone"},
                {"id": "L", "type": "LAB_TEST", "name": "white blood cell count"},
                {"id": "B", "type": "MICROBIOLOGY", "name": "E. coli culture"},
                {"id": "R", "type": "PROCEDURE", "name": "chest x-ray"},
                {"id": "S", "type": "SERVICE", "name": "medicine"},
            ],
            "edges": [
                {"source": "P", "target": "D", "relation": "HAS_DIAGNOSIS"},
                {"source": "D", "target": "M", "relation": "TREATED_BY"},
                {"source": "D", "target": "L", "relation": "DIAGNOSED_BY"},
                {"source": "B", "target": "D", "relation": "CONFIRMS"},
                {"source": "R", "target": "D", "relation": "PERFORMED_FOR"},
                {"source": "S", "target": "D", "relation": "MANAGED_FOR"},
            ],
        }

        graph = adapt_mimic_subkg(raw, source_path="example.json")

        self.assertEqual(
            {node["id"] for node in graph.nodes},
            {"P", "D", "M", "L", "B", "R", "S"},
        )
        self.assertEqual(
            {node["id"]: node["type"] for node in graph.nodes},
            {
                "P": "PATIENT",
                "D": "DIAGNOSIS",
                "M": "MEDICATION",
                "L": "LAB_TEST",
                "B": "MICROBIOLOGY",
                "R": "PROCEDURE",
                "S": "SERVICE",
            },
        )
        self.assertEqual(graph.nodes[1]["text"], "Pneumonia")
        self.assertEqual(
            {(edge["source_id"], edge["type"], edge["target_id"]) for edge in graph.edges},
            {
                ("P", "HAS_DIAGNOSIS", "D"),
                ("D", "TREATED_BY", "M"),
                ("D", "DIAGNOSED_BY", "L"),
                ("B", "CONFIRMS", "D"),
                ("R", "PERFORMED_FOR", "D"),
                ("S", "MANAGED_FOR", "D"),
            },
        )
        self.assertEqual(graph.extra["_method"], "mimic_subkg_adapter")
        self.assertEqual(graph.extra["_mimic_adapter"]["dropped_nodes"], 0)
        self.assertEqual(graph.extra["_mimic_adapter"]["dropped_edges"], 0)

        data = to_graph_data(graph, TinyEncoder())
        self.assertEqual(data.x.shape, (7, 4))
        self.assertEqual(data.edge_index.shape, (2, 6))
        self.assertTrue(set(data.node_type.tolist()) <= set(NODE_TYPE_TO_IDX.values()))
        self.assertTrue(set(data.edge_type.tolist()) <= set(EDGE_TYPE_TO_IDX.values()))

    def test_builder_loads_file_path(self):
        raw = {
            "nodes": [
                {"id": "D", "type": "DIAGNOSIS", "name": "hypertension"},
                {"id": "M", "type": "MEDICATION", "name": "lisinopril"},
            ],
            "edges": [
                {"source": "D", "target": "M", "relation": "TREATED_BY"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one.json"
            with open(path, "w") as f:
                json.dump(raw, f)

            graphs = MimicSubKGGraphBuilder(path).build()

        self.assertEqual(len(graphs), 1)
        self.assertEqual(graphs[0].edges[0]["type"], "TREATED_BY")

    def test_v3_score_loader_adapts_mimic_subkg_file(self):
        raw = {
            "subject_id": "10000000",
            "nodes": [
                {"id": "P", "type": "PATIENT", "name": "Patient 10000000"},
                {"id": "D", "type": "DIAGNOSIS", "name": "heart failure"},
                {"id": "M", "type": "MEDICATION", "name": "furosemide"},
            ],
            "edges": [
                {"source": "D", "target": "M", "relation": "TREATED_BY"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mimic.json"
            with open(path, "w") as f:
                json.dump(raw, f)

            graph = _load_graph_for_scoring(path)

        self.assertEqual({node["id"] for node in graph.nodes}, {"P", "D", "M"})
        self.assertEqual(graph.edges[0]["type"], "TREATED_BY")
        data = to_graph_data(graph, TinyEncoder())
        self.assertTrue(set(data.node_type.tolist()) <= set(NODE_TYPE_TO_IDX.values()))


if __name__ == "__main__":
    unittest.main()
