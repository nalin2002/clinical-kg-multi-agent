from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from graph_jepa.schema import EDGE_TYPE_TO_IDX, NODE_TYPE_TO_IDX, PatientGraph
from graph_jepa_v4.config import ModelConfig
from graph_jepa_v5.config import Config
from graph_jepa_v5.data import to_graph_data
from graph_jepa_v5.finetune import _apply_finetune_args, build_arg_parser
from graph_jepa_v5.model import (
    GraphJEPAv5,
    _confidence_supervision_masks,
    _weighted_relation_balanced_bce,
    confidence_sanitized_graph_data,
    pretrain_sanitized_graph_data,
)


class TinyEncoder:
    dim = 4

    def encode(self, keys):
        return np.ones((len(keys), self.dim), dtype=np.float32)


def _confidence_data():
    return SimpleNamespace(
        x=torch.randn(8, 4),
        edge_index=torch.tensor(
            [
                [0, 2, 4, 6],
                [1, 3, 5, 7],
            ],
            dtype=torch.long,
        ),
        edge_type=torch.tensor(
            [EDGE_TYPE_TO_IDX["TREATED_BY"]] * 4,
            dtype=torch.long,
        ),
        node_type=torch.tensor(
            [
                NODE_TYPE_TO_IDX["DIAGNOSIS"],
                NODE_TYPE_TO_IDX["MEDICATION"],
            ]
            * 4,
            dtype=torch.long,
        ),
        edge_llm_confidence=torch.tensor(
            [0.9, 0.1, 0.5, float("nan")],
            dtype=torch.float,
        ),
        edge_is_llm=torch.tensor([True, True, True, False]),
        edge_clinical_artifact=torch.zeros(4, dtype=torch.bool),
        num_nodes=8,
    )


def _model():
    return GraphJEPAv5(
        ModelConfig(
            in_dim=4,
            hidden_dim=8,
            latent_dim=8,
            num_gnn_layers=1,
            gnn_backend="torch",
            patch_heads=2,
        )
    )


class GraphJEPAv5ConfidenceTests(unittest.TestCase):
    def test_data_conversion_preserves_aligned_confidence(self):
        graph = PatientGraph(
            nodes=[
                {"id": "D", "type": "DIAGNOSIS", "text": "pneumonia"},
                {"id": "M1", "type": "MEDICATION", "text": "ceftriaxone"},
                {"id": "M2", "type": "MEDICATION", "text": "azithromycin"},
                {"id": "M3", "type": "MEDICATION", "text": "Bag"},
            ],
            edges=[
                {
                    "source_id": "D",
                    "target_id": "M1",
                    "type": "TREATED_BY",
                    "confidence": 0.2,
                    "evidence": "llm",
                },
                {
                    "source_id": "D",
                    "target_id": "M2",
                    "type": "TREATED_BY",
                    "confidence": 0.2,
                    "evidence": "structured",
                },
                {
                    "source_id": "D",
                    "target_id": "M3",
                    "type": "TREATED_BY",
                    "confidence": 1.0,
                    "evidence": "structured",
                },
            ],
        )

        data = to_graph_data(graph, TinyEncoder())

        self.assertEqual(data.edge_llm_confidence.size(0), data.edge_index.size(1))
        self.assertAlmostEqual(float(data.edge_llm_confidence[0]), 0.2)
        self.assertEqual(data.edge_is_llm.tolist(), [True, False, False])
        self.assertEqual(data.edge_clinical_artifact.tolist(), [False, False, True])

    def test_confidence_metadata_stays_aligned_in_batches(self):
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
                    "confidence": 0.9,
                    "evidence": "llm",
                },
            ],
        )
        converted = [to_graph_data(graph, TinyEncoder()) for _ in range(2)]

        batch = next(iter(DataLoader(converted, batch_size=2)))

        self.assertEqual(batch.edge_index.size(1), 2)
        torch.testing.assert_close(
            batch.edge_llm_confidence,
            torch.tensor([0.9, 0.9]),
        )
        self.assertEqual(batch.edge_is_llm.tolist(), [True, True])
        self.assertEqual(batch.edge_clinical_artifact.tolist(), [False, False])

    def test_confidence_masks_and_message_sanitization(self):
        data = _confidence_data()

        positive, low, ignored = _confidence_supervision_masks(
            data,
            enabled=True,
            negative_threshold=0.3,
            positive_threshold=0.8,
        )
        clean = confidence_sanitized_graph_data(
            data,
            enabled=True,
            negative_threshold=0.3,
            positive_threshold=0.8,
        )

        self.assertEqual(positive.tolist(), [True, False, False, True])
        self.assertEqual(low.tolist(), [False, True, False, False])
        self.assertEqual(ignored.tolist(), [False, False, True, False])
        self.assertEqual(clean.edge_index.size(1), 2)
        self.assertEqual(clean.edge_llm_confidence.size(0), 2)
        self.assertNotIn(2, clean.edge_index[0].tolist())
        self.assertNotIn(4, clean.edge_index[0].tolist())

    def test_pretraining_drops_only_llm_weak_negatives(self):
        data = _confidence_data()

        clean = pretrain_sanitized_graph_data(
            data,
            negative_threshold=0.3,
        )

        self.assertEqual(clean.edge_index.size(1), 3)
        self.assertEqual(clean.edge_index[0].tolist(), [0, 4, 6])
        self.assertEqual(clean.edge_llm_confidence.size(0), 3)

    def test_relation_specific_thresholds_change_tri_state(self):
        data = _confidence_data()
        data.edge_type = torch.tensor(
            [
                EDGE_TYPE_TO_IDX["TREATED_BY"],
                EDGE_TYPE_TO_IDX["TREATED_BY"],
                EDGE_TYPE_TO_IDX["CONFIRMS"],
                EDGE_TYPE_TO_IDX["TREATED_BY"],
            ],
            dtype=torch.long,
        )
        data.edge_llm_confidence = torch.tensor([0.85, 0.5, 0.85, 1.0])
        data.edge_is_llm = torch.tensor([True, True, True, False])

        positive, weak, ignored = _confidence_supervision_masks(
            data,
            enabled=True,
            negative_threshold=0.3,
            positive_threshold=0.8,
            negative_threshold_by_relation={
                "TREATED_BY": 0.5,
                "CONFIRMS": 0.5,
            },
            positive_threshold_by_relation={
                "TREATED_BY": 0.85,
                "CONFIRMS": 0.95,
            },
        )

        self.assertEqual(positive.tolist(), [True, False, False, True])
        self.assertEqual(weak.tolist(), [False, True, False, False])
        self.assertEqual(ignored.tolist(), [False, False, True, False])

    def test_artifacts_override_structured_positive_status(self):
        data = _confidence_data()
        data.edge_is_llm = torch.zeros(4, dtype=torch.bool)
        data.edge_clinical_artifact = torch.tensor([False, True, False, False])

        positive, weak, ignored = _confidence_supervision_masks(
            data,
            enabled=True,
            negative_threshold=0.3,
            positive_threshold=0.8,
            clinical_artifact_filters=True,
        )

        self.assertEqual(positive.tolist(), [True, False, True, True])
        self.assertEqual(weak.tolist(), [False, True, False, False])
        self.assertFalse(bool(ignored.any()))

    def test_revision_uses_low_confidence_as_negative_and_ignores_middle(self):
        loss, log = _model().revision_loss(
            _confidence_data(),
            mask_ratio=1.0,
            neg_per_pos=0,
            llm_confidence_negatives=True,
            llm_negative_threshold=0.3,
            llm_positive_threshold=0.8,
            llm_negative_weight=1.0,
        )

        self.assertTrue(loss.requires_grad)
        self.assertEqual(log["revision_pos"], 2)
        self.assertEqual(log["revision_llm_neg"], 1)
        self.assertEqual(log["revision_llm_ignored"], 1)
        self.assertEqual(log["revision_hidden"], 2)

    def test_revision_preserves_legacy_behavior_when_disabled(self):
        _loss, log = _model().revision_loss(
            _confidence_data(),
            mask_ratio=1.0,
            neg_per_pos=0,
            llm_confidence_negatives=False,
        )

        self.assertEqual(log["revision_pos"], 4)
        self.assertEqual(log["revision_llm_neg"], 0)
        self.assertEqual(log["revision_llm_ignored"], 0)

    def test_ranking_only_hides_high_confidence_or_unscored_positives(self):
        loss, log = _model().candidate_ranking_loss(
            _confidence_data(),
            mask_ratio=1.0,
            neg_per_pos=1,
            max_pos=10,
            temperature=0.2,
            llm_confidence_negatives=True,
            llm_negative_threshold=0.3,
            llm_positive_threshold=0.8,
        )

        self.assertTrue(loss.requires_grad)
        self.assertEqual(log["ranking_hidden"], 2)
        self.assertEqual(log["ranking_pos"], 2)
        self.assertEqual(log["ranking_llm_excluded"], 2)

    def test_llm_negative_weight_scales_negative_loss(self):
        logits = torch.tensor([0.0, 4.0])
        labels = torch.tensor([1.0, 0.0])
        relations = torch.tensor([0, 0])

        base = _weighted_relation_balanced_bce(
            logits,
            labels,
            relations,
            torch.tensor([1.0, 1.0]),
        )
        weighted = _weighted_relation_balanced_bce(
            logits,
            labels,
            relations,
            torch.tensor([1.0, 2.0]),
        )

        self.assertGreater(float(weighted), float(base))

    def test_old_checkpoints_receive_disabled_defaults(self):
        cfg = Config.from_dict({"train": {"epochs": 3}})

        self.assertFalse(cfg.train.llm_confidence_negatives)
        self.assertEqual(cfg.train.llm_negative_threshold, 0.3)
        self.assertEqual(cfg.train.llm_positive_threshold, 0.8)
        self.assertEqual(cfg.train.llm_negative_weight, 0.2)
        self.assertEqual(
            cfg.train.llm_positive_threshold_by_relation["CONFIRMS"],
            0.95,
        )
        self.assertEqual(
            cfg.train.llm_negative_threshold_by_relation["MANAGED_FOR"],
            0.5,
        )
        self.assertEqual(
            cfg.train.llm_positive_threshold_by_relation["MANAGED_FOR"],
            0.8,
        )
        self.assertFalse(cfg.train.clinical_artifact_filters)

    def test_finetune_cli_overrides_relation_thresholds(self):
        args = build_arg_parser().parse_args(
            [
                "--llm-tri-state-supervision",
                "--clinical-artifact-filters",
                "--llm-negative-threshold-relation",
                "TREATED_BY=0.4",
                "--llm-positive-threshold-relation",
                "CONFIRMS=0.97",
            ]
        )
        cfg = Config()

        _apply_finetune_args(args, cfg)

        self.assertTrue(cfg.train.llm_confidence_negatives)
        self.assertTrue(cfg.train.clinical_artifact_filters)
        self.assertEqual(
            cfg.train.llm_negative_threshold_by_relation["TREATED_BY"],
            0.4,
        )
        self.assertEqual(
            cfg.train.llm_positive_threshold_by_relation["CONFIRMS"],
            0.97,
        )


if __name__ == "__main__":
    unittest.main()
