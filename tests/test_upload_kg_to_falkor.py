import json
import tempfile
import unittest
from pathlib import Path

from scripts.upload_kg_to_falkor import (
    JsonlGraphAdapter,
    UploadOptions,
    build_upload_rows,
    coerce_property_value,
    flatten_properties,
    sanitize_identifier,
)


class UploadKGToFalkorTests(unittest.TestCase):
    def test_jsonl_adapter_uses_subject_and_hadm_for_graph_id(self):
        row = {
            "subject_id": 100,
            "hadm_id": 200,
            "nodes": [{"id": "N_000", "type": "PATIENT", "name": "patient_100"}],
            "edges": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graphs.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            record = next(JsonlGraphAdapter(path).iter_records())

        self.assertEqual(record.graph_id, "100:200")
        self.assertEqual(record.source_name, "graphs.jsonl:1")
        self.assertEqual(record.nodes[0]["id"], "N_000")

    def test_build_upload_rows_flattens_labels_and_adds_stable_ids(self):
        row = {
            "subject_id": 100,
            "hadm_id": 200,
            "note": "line one\nline two",
            "note_embedding": [0.1, 0.2],
            "nodes": [
                {"id": "N_000", "type": "PATIENT", "name": "patient_100"},
                {"id": "N_001", "type": "DIAGNOSIS", "name": "Sepsis"},
            ],
            "edges": [
                {
                    "source": "N_000",
                    "target": "N_001",
                    "relation": "HAS_DIAGNOSIS",
                    "labels": {"model": 1.0, "rxcui": None},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graphs.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            record = next(JsonlGraphAdapter(path).iter_records())

        rows = build_upload_rows([record], UploadOptions())

        self.assertEqual(rows.admissions[0]["graph_id"], "100:200")
        self.assertNotIn("note", rows.admissions[0]["properties"])
        self.assertNotIn("note_embedding", rows.admissions[0]["properties"])
        patient = rows.nodes_by_label["PATIENT"][0]
        self.assertEqual(patient["kg_node_id"], "100:200:N_000")
        self.assertEqual(patient["properties"]["text"], "patient_100")
        edge = rows.edges_by_relation["HAS_DIAGNOSIS"][0]
        self.assertEqual(edge["source_id"], "100:200:N_000")
        self.assertEqual(edge["target_id"], "100:200:N_001")
        self.assertEqual(edge["properties"]["labels_model"], 1.0)
        self.assertEqual(rows.stats.nodes, 2)
        self.assertEqual(rows.stats.edges, 1)
        self.assertEqual(rows.stats.graph_links, 2)

    def test_build_upload_rows_can_include_text_and_embedding(self):
        row = {
            "subject_id": 100,
            "hadm_id": 200,
            "note": "line one\nline two",
            "note_embedding": [0.1, 0.2],
            "nodes": [],
            "edges": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graphs.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            record = next(JsonlGraphAdapter(path).iter_records())

        rows = build_upload_rows(
            [record],
            UploadOptions(include_text_fields=True, include_note_embedding=True),
        )

        props = rows.admissions[0]["properties"]
        self.assertEqual(props["note"], "line one\\nline two")
        self.assertEqual(props["note_embedding"], [0.1, 0.2])

    def test_missing_edge_endpoint_is_skipped(self):
        row = {
            "subject_id": 100,
            "hadm_id": 200,
            "nodes": [{"id": "N_000", "type": "PATIENT"}],
            "edges": [{"source": "N_000", "target": "N_404", "relation": "BROKEN"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graphs.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            record = next(JsonlGraphAdapter(path).iter_records())

        rows = build_upload_rows([record], UploadOptions())

        self.assertEqual(rows.stats.edges, 0)
        self.assertEqual(rows.stats.skipped_edges, 1)
        self.assertEqual(rows.edges_by_relation, {})

    def test_sanitizers_and_property_coercion(self):
        self.assertEqual(sanitize_identifier("has diagnosis"), "HAS_DIAGNOSIS")
        self.assertEqual(sanitize_identifier("123 relation", "REL"), "_123_RELATION")
        self.assertEqual(coerce_property_value(float("nan")), None)
        self.assertEqual(coerce_property_value(["a", None, 1]), ["a", 1])
        self.assertEqual(
            flatten_properties({"labels": {"model-score": 0.9}}),
            {"labels_model_score": 0.9},
        )


if __name__ == "__main__":
    unittest.main()
