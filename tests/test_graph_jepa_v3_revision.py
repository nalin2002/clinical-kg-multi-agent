from types import SimpleNamespace
import unittest

import torch

from graph_jepa.schema import EDGE_TYPE_TO_IDX, NODE_TYPE_TO_IDX, PatientGraph
from graph_jepa_v3.config import Config, ModelConfig
from graph_jepa_v3.model import GraphJEPAv3, _sample_revision_negatives
from graph_jepa_v3.score import (
    _append_candidate_edges,
    _candidate_threshold,
    _flag,
    _iter_candidate_specs,
    _revision_action,
    _update_disabled_relations,
)


class GraphJEPAv3RevisionTests(unittest.TestCase):
    def test_iter_candidate_specs_filters_cross_res_and_negated_sources(self):
        graph = PatientGraph(
            nodes=[
                {
                    "id": "S1",
                    "type": "SYMPTOM",
                    "text": "dyspnea on exertion",
                    "occurrences": [{"res_id": "A"}],
                },
                {
                    "id": "S2",
                    "type": "SYMPTOM",
                    "text": "absent fever",
                    "occurrences": [{"res_id": "A"}],
                },
                {
                    "id": "D1",
                    "type": "DIAGNOSIS",
                    "text": "congestive heart failure",
                    "occurrences": [{"res_id": "A"}],
                },
                {
                    "id": "D2",
                    "type": "DIAGNOSIS",
                    "text": "medial epicondylitis",
                    "occurrences": [{"res_id": "B"}],
                },
            ],
            edges=[],
        )

        specs = {
            (graph.nodes[s]["id"], graph.nodes[t]["id"], relation)
            for s, t, relation in _iter_candidate_specs(graph, Config())
        }

        self.assertIn(("S1", "D1", "INDICATES"), specs)
        self.assertNotIn(("S1", "D2", "INDICATES"), specs)
        self.assertNotIn(("S2", "D1", "INDICATES"), specs)

    def test_iter_candidate_specs_includes_canonical_mimic_relations(self):
        graph = PatientGraph(
            nodes=[
                {"id": "P", "type": "PATIENT", "text": "Patient 10000000"},
                {"id": "D", "type": "DIAGNOSIS", "text": "pneumonia"},
                {"id": "M", "type": "MEDICATION", "text": "ceftriaxone"},
                {"id": "L", "type": "LAB_TEST", "text": "white blood cell count"},
                {"id": "B", "type": "MICROBIOLOGY", "text": "blood culture"},
                {"id": "R", "type": "PROCEDURE", "text": "chest x-ray"},
                {"id": "S", "type": "SERVICE", "text": "medicine"},
            ],
            edges=[
                {"source_id": "P", "target_id": "D", "type": "HAS_DIAGNOSIS"},
            ],
        )

        specs = {
            (graph.nodes[s]["id"], graph.nodes[t]["id"], relation)
            for s, t, relation in _iter_candidate_specs(graph, Config())
        }

        self.assertNotIn(("P", "D", "HAS_DIAGNOSIS"), specs)
        self.assertIn(("P", "M", "TAKES_MEDICATION"), specs)
        self.assertIn(("P", "L", "HAD_LAB_TEST"), specs)
        self.assertIn(("P", "B", "HAD_MICROBIOLOGY"), specs)
        self.assertIn(("P", "R", "UNDERWENT_PROCEDURE"), specs)
        self.assertIn(("P", "S", "MANAGED_BY_SERVICE"), specs)
        self.assertIn(("D", "M", "TREATED_BY"), specs)
        self.assertIn(("D", "L", "DIAGNOSED_BY"), specs)
        self.assertIn(("D", "B", "INVESTIGATED_BY"), specs)
        self.assertIn(("D", "L", "MONITORED_BY"), specs)
        self.assertIn(("B", "D", "CONFIRMS"), specs)
        self.assertIn(("M", "B", "TARGETS_ORGANISM"), specs)
        self.assertIn(("R", "D", "PERFORMED_FOR"), specs)
        self.assertIn(("S", "D", "MANAGED_FOR"), specs)
        self.assertIn(("M", "S", "ADMINISTERED_DURING"), specs)
        self.assertNotIn(("M", "D", "TREATED_BY"), specs)
        self.assertNotIn(("D", "S", "MANAGED_FOR"), specs)

    def test_append_candidate_edges_marks_v3_revision_metadata(self):
        graph = PatientGraph(
            nodes=[
                {"id": "S", "type": "SYMPTOM", "text": "edema"},
                {"id": "D", "type": "DIAGNOSIS", "text": "heart failure"},
            ],
            edges=[],
        )

        added = _append_candidate_edges(
            graph,
            [(0.91, 0, 1, "INDICATES")],
            Config(),
        )

        self.assertEqual(added, 1)
        edge = graph.edges[0]
        self.assertEqual(edge["jepa_source"], "graph_jepa_v3_candidate_generation")
        self.assertEqual(edge["jepa_revision_action"], "add_candidate")
        self.assertTrue(edge["jepa_suggested"])
        self.assertTrue(edge["jepa_unverified"])

    def test_relation_specific_review_thresholds_are_separate_from_global_defaults(self):
        cfg = Config()

        self.assertEqual(_flag(0.12, cfg, "LOCATED_AT"), "weak")
        self.assertEqual(_revision_action(0.12, cfg, relation="LOCATED_AT"), "review")
        self.assertEqual(_flag(0.12, cfg, "INDICATES"), "inconsistent")

    def test_candidate_thresholds_are_relation_specific_and_can_disable_relations(self):
        cfg = Config()

        self.assertEqual(_candidate_threshold("INDICATES", cfg), 0.95)
        self.assertEqual(_candidate_threshold("CONFIRMS", cfg), 0.96)
        self.assertEqual(_candidate_threshold("RULES_OUT", cfg), 0.99)
        self.assertEqual(_candidate_threshold("LOCATED_AT", cfg), 0.95)

        cfg.score.disabled_candidate_relations = _update_disabled_relations(
            cfg.score.disabled_candidate_relations,
            enable=None,
            disable=["RULES_OUT"],
        )
        self.assertIsNone(_candidate_threshold("RULES_OUT", cfg))

    def test_revision_loss_scores_hidden_edges_and_schema_valid_negatives(self):
        cfg = ModelConfig(
            in_dim=4,
            hidden_dim=8,
            latent_dim=8,
            num_gnn_layers=1,
            gnn_backend="torch",
            patch_heads=2,
        )
        model = GraphJEPAv3(cfg)
        data = SimpleNamespace(
            x=torch.randn(4, 4),
            edge_index=torch.tensor([[0, 0], [1, 3]], dtype=torch.long),
            edge_type=torch.tensor(
                [
                    EDGE_TYPE_TO_IDX["INDICATES"],
                    EDGE_TYPE_TO_IDX["LOCATED_AT"],
                ],
                dtype=torch.long,
            ),
            node_type=torch.tensor(
                [
                    NODE_TYPE_TO_IDX["SYMPTOM"],
                    NODE_TYPE_TO_IDX["DIAGNOSIS"],
                    NODE_TYPE_TO_IDX["DIAGNOSIS"],
                    NODE_TYPE_TO_IDX["LOCATION"],
                ],
                dtype=torch.long,
            ),
            num_nodes=4,
        )

        loss, log = model.revision_loss(data, mask_ratio=1.0, neg_per_pos=2)

        self.assertTrue(loss.requires_grad)
        self.assertGreater(log["revision_pos"], 0)
        self.assertGreater(log["revision_neg"], 0)
        self.assertEqual(log["revision_hidden"], 2)

    def test_revision_negative_sampler_adds_relation_source_and_target_corruptions(self):
        data = SimpleNamespace(
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_type=torch.tensor([EDGE_TYPE_TO_IDX["CONFIRMS"]], dtype=torch.long),
            node_type=torch.tensor(
                [
                    NODE_TYPE_TO_IDX["PROCEDURE"],
                    NODE_TYPE_TO_IDX["DIAGNOSIS"],
                    NODE_TYPE_TO_IDX["DIAGNOSIS"],
                    NODE_TYPE_TO_IDX["PROCEDURE"],
                ],
                dtype=torch.long,
            ),
            num_nodes=4,
        )

        neg_src, neg_dst, neg_rel = _sample_revision_negatives(data, neg_per_pos=3)
        triples = set(zip(neg_src.tolist(), neg_dst.tolist(), neg_rel.tolist()))

        self.assertNotIn((0, 1, EDGE_TYPE_TO_IDX["CONFIRMS"]), triples)
        self.assertTrue(
            any(
                s == 0 and t == 1 and r != EDGE_TYPE_TO_IDX["CONFIRMS"]
                for s, t, r in triples
            )
        )
        self.assertIn((0, 2, EDGE_TYPE_TO_IDX["CONFIRMS"]), triples)
        self.assertTrue(
            any(
                s != 0 and t == 1 and r == EDGE_TYPE_TO_IDX["CONFIRMS"]
                for s, t, r in triples
            )
        )


if __name__ == "__main__":
    unittest.main()
