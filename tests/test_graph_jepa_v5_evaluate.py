import unittest

import torch
from torch_geometric.data import Data

from graph_jepa.schema import EDGE_TYPE_TO_IDX, NODE_TYPE_TO_IDX
from graph_jepa_v5.config import Config
from graph_jepa_v5.evaluate import leave_one_out_recovery


class _FeatureScoreHead:
    def __call__(self, _z_src, z_tgt, _relation):
        return z_tgt[:, 0]


class _FeatureScoreModel:
    def __init__(self):
        self.edge_head = _FeatureScoreHead()

    def eval(self):
        return self

    def context_node_encoder(self, x, _edge_index, _edge_type):
        return x


def _cfg():
    cfg = Config()
    cfg.model.num_relations = len(EDGE_TYPE_TO_IDX)
    return cfg


class GraphJEPAv5EvaluateTests(unittest.TestCase):
    def test_leave_one_out_filters_other_true_tails(self):
        data = Data(
            x=torch.tensor([[0.0], [0.9], [0.1], [1.0], [0.5]]),
            edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
            edge_type=torch.tensor(
                [EDGE_TYPE_TO_IDX["TREATED_BY"], EDGE_TYPE_TO_IDX["TREATED_BY"]],
                dtype=torch.long,
            ),
        )
        data.node_type = torch.tensor(
            [
                NODE_TYPE_TO_IDX["DIAGNOSIS"],
                NODE_TYPE_TO_IDX["MEDICATION"],
                NODE_TYPE_TO_IDX["MEDICATION"],
                NODE_TYPE_TO_IDX["SERVICE"],
                NODE_TYPE_TO_IDX["MEDICATION"],
            ],
            dtype=torch.long,
        )
        data.num_nodes = 5

        metrics = leave_one_out_recovery(
            _FeatureScoreModel(),
            [data],
            _cfg(),
            torch.device("cpu"),
        )

        self.assertEqual(metrics["n"], 2)
        self.assertAlmostEqual(metrics["mrr"], 0.75)
        self.assertAlmostEqual(metrics["hits1"], 0.5)
        self.assertAlmostEqual(metrics["hits3"], 1.0)
        self.assertEqual(metrics["per_rel"][0]["rel"], "TREATED_BY")
        self.assertEqual(metrics["per_rel"][0]["C"], 2.0)

    def test_candidate_mode_schema_can_rank_against_other_allowed_types(self):
        data = Data(
            x=torch.tensor([[0.0], [0.2], [0.1], [0.8]]),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_type=torch.tensor([EDGE_TYPE_TO_IDX["CONFIRMS"]], dtype=torch.long),
        )
        data.node_type = torch.tensor(
            [
                NODE_TYPE_TO_IDX["PROCEDURE"],
                NODE_TYPE_TO_IDX["DIAGNOSIS"],
                NODE_TYPE_TO_IDX["DIAGNOSIS"],
                NODE_TYPE_TO_IDX["MICROBIOLOGY"],
            ],
            dtype=torch.long,
        )
        data.num_nodes = 4

        schema_metrics = leave_one_out_recovery(
            _FeatureScoreModel(),
            [data],
            _cfg(),
            torch.device("cpu"),
            candidate_mode="schema",
        )
        same_type_metrics = leave_one_out_recovery(
            _FeatureScoreModel(),
            [data],
            _cfg(),
            torch.device("cpu"),
            candidate_mode="same-type",
        )

        self.assertEqual(schema_metrics["n"], 1)
        self.assertEqual(same_type_metrics["n"], 1)
        self.assertAlmostEqual(schema_metrics["mrr"], 0.5)
        self.assertAlmostEqual(same_type_metrics["mrr"], 1.0)
        self.assertEqual(schema_metrics["per_rel"][0]["C"], 3.0)
        self.assertEqual(same_type_metrics["per_rel"][0]["C"], 2.0)

    def test_llm_weak_negatives_are_not_evaluated_as_positives(self):
        data = Data(
            x=torch.tensor([[0.0], [0.5], [0.9], [0.1]]),
            edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
            edge_type=torch.tensor(
                [EDGE_TYPE_TO_IDX["TREATED_BY"], EDGE_TYPE_TO_IDX["TREATED_BY"]],
                dtype=torch.long,
            ),
        )
        data.node_type = torch.tensor(
            [
                NODE_TYPE_TO_IDX["DIAGNOSIS"],
                NODE_TYPE_TO_IDX["MEDICATION"],
                NODE_TYPE_TO_IDX["MEDICATION"],
                NODE_TYPE_TO_IDX["MEDICATION"],
            ],
            dtype=torch.long,
        )
        data.edge_llm_confidence = torch.tensor([0.9, 0.1], dtype=torch.float)
        data.edge_is_llm = torch.tensor([True, True])
        data.edge_clinical_artifact = torch.tensor([False, False])
        data.num_nodes = 4
        cfg = _cfg()
        cfg.train.llm_confidence_negatives = True
        cfg.train.llm_negative_threshold = 0.3
        cfg.train.llm_positive_threshold = 0.8

        metrics = leave_one_out_recovery(
            _FeatureScoreModel(),
            [data],
            cfg,
            torch.device("cpu"),
        )

        self.assertEqual(metrics["n"], 1)
        self.assertEqual(metrics["per_rel"][0]["rel"], "TREATED_BY")


if __name__ == "__main__":
    unittest.main()
