import unittest

import torch
from torch_geometric.data import Data

from graph_jepa.schema import EDGE_TYPE_TO_IDX, PatientGraph
from graph_jepa_v5.evaluate_llm import RecoveryQuery
from graph_jepa_v5.evaluate_loo_jepa_llm import (
    LOO_NODE_TYPES,
    LOO_RELATION_CANONICAL,
    _loo_relation_id,
    filter_loo_compatible_queries,
    to_loo_graph_data,
)


class GraphJEPAv5EvaluateLooJepaLlmTests(unittest.TestCase):
    def test_raw_patient_graph_converts_to_loo_tensor_contract(self):
        graph = PatientGraph(
            nodes=[
                {
                    "id": "N0",
                    "type": "PATIENT",
                    "normalized_name": "patient_1",
                },
                {
                    "id": "N1",
                    "type": "DIAGNOSIS",
                    "normalized_name": "pneumonia",
                },
                {
                    "id": "N2",
                    "type": "MEDICATION",
                    "normalized_name": "cefepime",
                },
            ],
            edges=[
                {
                    "source_id": "N0",
                    "target_id": "N1",
                    "type": "HAS_DIAGNOSIS",
                    "model": 1.0,
                },
                {
                    "source_id": "N1",
                    "target_id": "N2",
                    "type": "TREATED_BY",
                    "omop_lca_dist": 5.0,
                },
            ],
            extra={"_patient": {"gender": "F", "anchor_age": 50}},
        )

        data = to_loo_graph_data(graph, entity_vocab=128, use_scores=True)

        self.assertEqual(data.num_nodes, 3)
        self.assertEqual(
            data.node_type.tolist(),
            [
                LOO_NODE_TYPES["PATIENT"],
                LOO_NODE_TYPES["DIAGNOSIS"],
                LOO_NODE_TYPES["MEDICATION"],
            ],
        )
        self.assertEqual(
            data.edge_type.tolist(),
            [
                LOO_RELATION_CANONICAL["HAS_DIAGNOSIS"],
                LOO_RELATION_CANONICAL["TREATED_BY"],
            ],
        )
        self.assertTrue(
            torch.allclose(
                data.numfeat[0],
                torch.tensor([0.5, 0, 1, 0, 0, 0], dtype=torch.float),
            )
        )
        self.assertEqual(data.edge_feat.shape, (2, 14))

    def test_loo_relation_id_accepts_schema_aliases(self):
        self.assertEqual(
            _loo_relation_id("DIAGNORED_BY"),
            LOO_RELATION_CANONICAL["DIAGNOSED_BY"],
        )
        self.assertEqual(
            _loo_relation_id(EDGE_TYPE_TO_IDX["TARGET_ORGANISM"]),
            LOO_RELATION_CANONICAL["TARGETS_ORGANISM"],
        )

    def test_filter_loo_compatible_queries_requires_supported_exact_edge(self):
        data = Data()
        data.num_nodes = 3
        data.edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        data.edge_type = torch.tensor(
            [LOO_RELATION_CANONICAL["HAS_DIAGNOSIS"]],
            dtype=torch.long,
        )
        good = RecoveryQuery(
            graph_index=0,
            edge_index=0,
            source=0,
            target=1,
            relation=EDGE_TYPE_TO_IDX["HAS_DIAGNOSIS"],
            candidates=(1, 2),
        )
        missing = RecoveryQuery(
            graph_index=0,
            edge_index=1,
            source=0,
            target=2,
            relation=EDGE_TYPE_TO_IDX["HAS_DIAGNOSIS"],
            candidates=(1, 2),
        )
        unsupported = RecoveryQuery(
            graph_index=0,
            edge_index=2,
            source=0,
            target=1,
            relation=EDGE_TYPE_TO_IDX["LOCATED_AT"],
            candidates=(1, 2),
        )

        kept, skipped = filter_loo_compatible_queries(
            [good, missing, unsupported],
            [data],
        )

        self.assertEqual(kept, [good])
        self.assertEqual(skipped["missing_exact_edge:HAS_DIAGNOSIS"], 1)
        self.assertEqual(skipped["unsupported_relation:LOCATED_AT"], 1)


if __name__ == "__main__":
    unittest.main()
