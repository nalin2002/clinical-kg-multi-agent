from types import SimpleNamespace
import unittest

import torch

from graph_jepa.schema import EDGE_TYPE_TO_IDX, NODE_TYPE_TO_IDX
from graph_jepa_v4.config import ModelConfig
from graph_jepa_v4.model import (
    GraphJEPAv4,
    _relation_balanced_bce,
    _schema_edge_masks,
    sanitized_graph_data,
)


class GraphJEPAv4TrainingRobustnessTests(unittest.TestCase):
    def test_sanitized_graph_removes_invalid_typed_edges_but_keeps_cooccurrence(self):
        data = SimpleNamespace(
            x=torch.randn(4, 4),
            edge_index=torch.tensor(
                [
                    [0, 1, 2],
                    [1, 0, 3],
                ],
                dtype=torch.long,
            ),
            edge_type=torch.tensor(
                [
                    EDGE_TYPE_TO_IDX["TREATED_BY"],
                    EDGE_TYPE_TO_IDX["TREATED_BY"],
                    EDGE_TYPE_TO_IDX["CO_OCCURS_WITH"],
                ],
                dtype=torch.long,
            ),
            node_type=torch.tensor(
                [
                    NODE_TYPE_TO_IDX["DIAGNOSIS"],
                    NODE_TYPE_TO_IDX["MEDICATION"],
                    NODE_TYPE_TO_IDX["DIAGNOSIS"],
                    NODE_TYPE_TO_IDX["MEDICATION"],
                ],
                dtype=torch.long,
            ),
            num_nodes=4,
        )

        valid, invalid, unconstrained = _schema_edge_masks(data, allow_unconstrained=True)
        clean = sanitized_graph_data(data)

        self.assertEqual(valid.tolist(), [True, False, True])
        self.assertEqual(invalid.tolist(), [False, True, False])
        self.assertEqual(unconstrained.tolist(), [False, False, True])
        self.assertEqual(clean.edge_index.size(1), 2)
        self.assertEqual(clean.edge_type.tolist(), [
            EDGE_TYPE_TO_IDX["TREATED_BY"],
            EDGE_TYPE_TO_IDX["CO_OCCURS_WITH"],
        ])

    def test_revision_loss_uses_valid_positives_and_invalid_observed_negatives(self):
        cfg = ModelConfig(
            in_dim=4,
            hidden_dim=8,
            latent_dim=8,
            num_gnn_layers=1,
            gnn_backend="torch",
            patch_heads=2,
        )
        model = GraphJEPAv4(cfg)
        data = SimpleNamespace(
            x=torch.randn(4, 4),
            edge_index=torch.tensor(
                [
                    [0, 1],
                    [1, 0],
                ],
                dtype=torch.long,
            ),
            edge_type=torch.tensor(
                [
                    EDGE_TYPE_TO_IDX["TREATED_BY"],
                    EDGE_TYPE_TO_IDX["TREATED_BY"],
                ],
                dtype=torch.long,
            ),
            node_type=torch.tensor(
                [
                    NODE_TYPE_TO_IDX["DIAGNOSIS"],
                    NODE_TYPE_TO_IDX["MEDICATION"],
                    NODE_TYPE_TO_IDX["DIAGNOSIS"],
                    NODE_TYPE_TO_IDX["MEDICATION"],
                ],
                dtype=torch.long,
            ),
            num_nodes=4,
        )

        loss, log = model.revision_loss(data, mask_ratio=1.0, neg_per_pos=1)

        self.assertTrue(loss.requires_grad)
        self.assertEqual(log["revision_pos"], 1)
        self.assertGreaterEqual(log["revision_invalid_neg"], 1)
        self.assertGreaterEqual(log["revision_reverse_neg"], 1)
        self.assertGreater(log["revision_neg"], 0)

    def test_relation_balanced_loss_averages_relation_label_cells(self):
        logits = torch.tensor([4.0, 4.0, -4.0, 0.0])
        labels = torch.tensor([1.0, 1.0, 0.0, 1.0])
        rel = torch.tensor([0, 0, 0, 1])

        loss = _relation_balanced_bce(logits, labels, rel)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
