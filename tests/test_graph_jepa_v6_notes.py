import unittest

import numpy as np
import torch

from graph_jepa.schema import PatientGraph
from graph_jepa_v6.data import to_graph_data


class TinyEncoder:
    dim = 4

    def encode(self, keys):
        return np.ones((len(keys), self.dim), dtype=np.float32)


class RecordingEncoder(TinyEncoder):
    def __init__(self):
        self.keys = None

    def encode(self, keys):
        self.keys = list(keys)
        return super().encode(keys)


class GraphJEPAv6NoteTests(unittest.TestCase):
    def test_name_is_used_as_text_when_text_field_is_missing(self):
        graph = PatientGraph(
            nodes=[
                {"id": "D", "type": "DIAGNOSIS", "name": "Pneumonia"},
                {"id": "M", "type": "MEDICATION", "normalized_name": "ceftriaxone"},
            ],
            edges=[
                {
                    "source": "D",
                    "target": "M",
                    "relation": "TREATED_BY",
                },
            ],
            extra={"note_embedding": [0.0]},
        )
        encoder = RecordingEncoder()

        to_graph_data(graph, encoder, note_embedding_dim=1)

        self.assertEqual(
            encoder.keys,
            [("DIAGNOSIS", "Pneumonia"), ("MEDICATION", "ceftriaxone")],
        )

    def test_provenance_grounding_localizes_note_embedding(self):
        graph = PatientGraph(
            nodes=[
                {"id": "P", "type": "PATIENT", "text": "patient"},
                {"id": "D", "type": "DIAGNOSIS", "text": "pneumonia"},
                {"id": "M", "type": "MEDICATION", "text": "ceftriaxone"},
                {"id": "S", "type": "SERVICE", "text": "medicine"},
            ],
            edges=[
                {
                    "source_id": "P",
                    "target_id": "D",
                    "type": "HAS_DIAGNOSIS",
                    "labels": {"prov_in_note": 0.0},
                },
                {
                    "source_id": "D",
                    "target_id": "M",
                    "type": "TREATED_BY",
                    "labels": {"prov_in_note": 1.0},
                },
            ],
            extra={"note_embedding": [0.25, -0.5, 1.0]},
        )

        data = to_graph_data(graph, TinyEncoder(), note_embedding_dim=3)

        self.assertEqual(tuple(data.x.shape), (4, 7))
        expected_note = torch.tensor([0.25, -0.5, 1.0])
        torch.testing.assert_close(data.x[1, 4:], expected_note)
        torch.testing.assert_close(data.x[2, 4:], expected_note)
        torch.testing.assert_close(data.x[0, 4:], torch.zeros(3))
        torch.testing.assert_close(data.x[3, 4:], torch.zeros(3))
        self.assertEqual(data.note_grounded_mask.tolist(), [False, True, True, False])
        self.assertEqual(int(data.n_note_grounded[0]), 2)
        self.assertTrue(bool(data.note_embedding_present[0]))

    def test_missing_note_embedding_keeps_wide_zero_features(self):
        graph = PatientGraph(
            nodes=[
                {"id": "D", "type": "DIAGNOSIS", "text": "pneumonia"},
                {"id": "M", "type": "MEDICATION", "text": "ceftriaxone"},
            ],
            edges=[
                {
                    "source_id": "D",
                    "target_id": "M",
                    "type": "TREATED_BY",
                    "labels": {"prov_in_note": 1.0},
                },
            ],
        )

        data = to_graph_data(graph, TinyEncoder(), note_embedding_dim=2)

        self.assertEqual(tuple(data.x.shape), (2, 6))
        torch.testing.assert_close(data.x[:, 4:], torch.zeros((2, 2)))
        self.assertEqual(data.note_grounded_mask.tolist(), [True, True])
        self.assertFalse(bool(data.note_embedding_present[0]))

    def test_source_target_relation_edge_aliases_are_supported(self):
        graph = PatientGraph(
            nodes=[
                {"id": "D", "type": "DIAGNOSIS", "text": "pneumonia"},
                {"id": "M", "type": "MEDICATION", "text": "ceftriaxone"},
            ],
            edges=[
                {
                    "source": "D",
                    "target": "M",
                    "relation": "TREATED_BY",
                    "labels": {"prov_in_note": 1.0},
                },
            ],
            extra={"note_embedding": [1.0, 2.0]},
        )

        data = to_graph_data(graph, TinyEncoder(), note_embedding_dim=2)

        self.assertEqual(data.edge_index.tolist(), [[0], [1]])
        self.assertEqual(data.note_grounded_mask.tolist(), [True, True])
        torch.testing.assert_close(data.x[:, 4:], torch.tensor([[1.0, 2.0], [1.0, 2.0]]))

    def test_bad_note_embedding_dim_fails_fast(self):
        graph = PatientGraph(
            nodes=[{"id": "D", "type": "DIAGNOSIS", "text": "pneumonia"}],
            edges=[],
            extra={"note_embedding": [1.0, 2.0]},
        )

        with self.assertRaises(ValueError):
            to_graph_data(graph, TinyEncoder(), note_embedding_dim=3)


if __name__ == "__main__":
    unittest.main()
