import unittest

from graph_jepa.schema import PatientGraph
from graph_jepa_v2.config import Config
from graph_jepa_v2.score import _append_candidate_edges, _iter_candidate_specs


class GraphJEPAv2CandidateTests(unittest.TestCase):
    def test_iter_candidate_specs_uses_schema_and_skips_existing_edges(self):
        graph = PatientGraph(
            nodes=[
                {"id": "S", "type": "SYMPTOM", "text": "chest pain"},
                {"id": "D", "type": "DIAGNOSIS", "text": "angina"},
                {"id": "L", "type": "LOCATION", "text": "chest"},
                {"id": "T", "type": "TREATMENT", "text": "nitroglycerin"},
            ],
            edges=[
                {"source_id": "S", "target_id": "D", "type": "INDICATES"},
            ],
        )

        specs = {
            (graph.nodes[s]["id"], graph.nodes[t]["id"], relation)
            for s, t, relation in _iter_candidate_specs(graph)
        }

        self.assertIn(("S", "L", "LOCATED_AT"), specs)
        self.assertIn(("T", "S", "TAKEN_FOR"), specs)
        self.assertIn(("T", "D", "TAKEN_FOR"), specs)
        self.assertNotIn(("S", "D", "INDICATES"), specs)
        self.assertNotIn(("S", "D", "LOCATED_AT"), specs)

    def test_append_candidate_edges_marks_suggestions_unverified(self):
        graph = PatientGraph(
            nodes=[
                {"id": "S", "type": "SYMPTOM", "text": "cough"},
                {"id": "D", "type": "DIAGNOSIS", "text": "pneumonia"},
            ],
            edges=[],
        )

        added = _append_candidate_edges(
            graph,
            [(0.91, 0, 1, "INDICATES")],
            Config(),
        )

        self.assertEqual(added, 1)
        self.assertEqual(len(graph.edges), 1)
        edge = graph.edges[0]
        self.assertEqual(edge["source_id"], "S")
        self.assertEqual(edge["target_id"], "D")
        self.assertEqual(edge["type"], "INDICATES")
        self.assertEqual(edge["jepa_score"], 0.91)
        self.assertEqual(edge["jepa_flag"], "ok")
        self.assertTrue(edge["jepa_suggested"])
        self.assertTrue(edge["jepa_unverified"])
        self.assertEqual(edge["evidence"], "")
        self.assertEqual(edge["turn_id"], "")


if __name__ == "__main__":
    unittest.main()
