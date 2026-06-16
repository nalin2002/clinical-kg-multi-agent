import unittest

import torch
from torch_geometric.data import Data

from graph_jepa.schema import EDGE_TYPE_TO_IDX, NODE_TYPE_TO_IDX
from graph_jepa_v5.config import Config
from graph_jepa_v5.evaluate_llm import collect_recovery_queries, parse_ranking


def _cfg():
    cfg = Config()
    cfg.model.num_relations = len(EDGE_TYPE_TO_IDX)
    return cfg


class GraphJEPAv5EvaluateLlmTests(unittest.TestCase):
    def test_parse_ranking_accepts_json_and_appends_missing_candidates(self):
        order, parse_ok, parse_complete = parse_ranking(
            '{"ranking":[3, 1]}',
            candidate_count=4,
        )

        self.assertTrue(parse_ok)
        self.assertFalse(parse_complete)
        self.assertEqual(order, [2, 0, 1, 3])

    def test_parse_ranking_falls_back_to_numbers(self):
        order, parse_ok, parse_complete = parse_ranking(
            "Best order: 2 > 1 > 3",
            candidate_count=3,
        )

        self.assertTrue(parse_ok)
        self.assertTrue(parse_complete)
        self.assertEqual(order, [1, 0, 2])

    def test_collect_queries_samples_per_relation_and_caps_candidates(self):
        data = Data(
            x=torch.randn(8, 4),
            edge_index=torch.tensor(
                [
                    [0, 0, 4],
                    [1, 2, 5],
                ],
                dtype=torch.long,
            ),
            edge_type=torch.tensor(
                [
                    EDGE_TYPE_TO_IDX["TREATED_BY"],
                    EDGE_TYPE_TO_IDX["TREATED_BY"],
                    EDGE_TYPE_TO_IDX["CONFIRMS"],
                ],
                dtype=torch.long,
            ),
        )
        data.node_type = torch.tensor(
            [
                NODE_TYPE_TO_IDX["DIAGNOSIS"],
                NODE_TYPE_TO_IDX["MEDICATION"],
                NODE_TYPE_TO_IDX["MEDICATION"],
                NODE_TYPE_TO_IDX["MEDICATION"],
                NODE_TYPE_TO_IDX["PROCEDURE"],
                NODE_TYPE_TO_IDX["DIAGNOSIS"],
                NODE_TYPE_TO_IDX["DIAGNOSIS"],
                NODE_TYPE_TO_IDX["MICROBIOLOGY"],
            ],
            dtype=torch.long,
        )
        data.num_nodes = 8

        queries = collect_recovery_queries(
            [data],
            _cfg(),
            samples_per_relation=1,
            candidate_mode="schema",
            max_candidates=2,
            seed=7,
        )

        self.assertEqual(len(queries), 2)
        self.assertEqual({query.relation for query in queries}, {
            EDGE_TYPE_TO_IDX["TREATED_BY"],
            EDGE_TYPE_TO_IDX["CONFIRMS"],
        })
        self.assertTrue(all(len(query.candidates) == 2 for query in queries))
        self.assertTrue(all(query.target in query.candidates for query in queries))

    def test_duplicate_exact_triples_are_skipped_by_default(self):
        data = Data(
            x=torch.randn(3, 4),
            edge_index=torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
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
            ],
            dtype=torch.long,
        )
        data.num_nodes = 3

        skipped = collect_recovery_queries(
            [data],
            _cfg(),
            samples_per_relation=10,
            candidate_mode="schema",
            max_candidates=None,
            seed=0,
        )
        allowed = collect_recovery_queries(
            [data],
            _cfg(),
            samples_per_relation=10,
            candidate_mode="schema",
            max_candidates=None,
            seed=0,
            allow_duplicate_triples=True,
        )

        self.assertEqual(skipped, [])
        self.assertEqual(len(allowed), 2)


if __name__ == "__main__":
    unittest.main()
