import unittest

from Clinical_KG_OS_LLM.falkor_graphrag_qa_pipeline import (
    FalkorEdge,
    FalkorGraph,
    FalkorNode,
    GraphRef,
    candidate_bundle_ids,
    parse_transcript_text,
    retrieve_context,
)


class FakeEmbeddingModel:
    def get_text_embedding(self, text):
        return [1.0, 0.0] if "cough" in text.lower() else [0.0, 1.0]


class FalkorGraphRAGPipelineTests(unittest.TestCase):
    def test_candidate_bundle_ids_prefers_requested_and_res_ids(self):
        ref = GraphRef(
            graph_id="fawkes_aci_bench_eval_260616.jsonl:1",
            res_id="RES0200",
            encounter_id="enc-1",
            source_name="RES0200_cooperative_multi_agent.json",
            requested_id="RES0200",
        )

        self.assertEqual(
            candidate_bundle_ids(ref),
            [
                "RES0200",
                "enc-1",
                "fawkes_aci_bench_eval_260616.jsonl:1",
            ],
        )

    def test_retrieve_context_uses_falkor_nodes_edges_and_turns(self):
        graph = FalkorGraph(
            ref=GraphRef(graph_id="g1", res_id="RES0200"),
            transcript=None,
            note=None,
            input_block=None,
            nodes=[
                FalkorNode("g1:N1", "N1", "dry cough", "SYMPTOM", "P-1"),
                FalkorNode("g1:N2", "N2", "copd exacerbation", "DIAGNOSIS", "D-2"),
            ],
            edges=[
                FalkorEdge(
                    kg_edge_id="g1:E1",
                    source_kg_node_id="g1:N1",
                    source_text="dry cough",
                    source_type="SYMPTOM",
                    relation="SUPPORTS",
                    target_kg_node_id="g1:N2",
                    target_text="copd exacerbation",
                    target_type="DIAGNOSIS",
                    turn_id="D-2",
                    evidence="doctor linked the symptom to the assessment",
                )
            ],
        )
        turn_dict = parse_transcript_text(
            "[P-1] P: I have a dry cough.\n[D-2] D: This supports COPD exacerbation."
        )
        embeddings = [[1.0, 0.0], [0.0, 1.0]]

        context = retrieve_context(
            question="What does the cough support?",
            graph=graph,
            embeddings=embeddings,
            embed_model=FakeEmbeddingModel(),
            turn_dict=turn_dict,
            top_k=1,
            edge_limit=5,
            max_source_chars=1000,
        )

        self.assertIn("[SYMPTOM] dry cough", context)
        self.assertIn("--[SUPPORTS]-->", context)
        self.assertIn("[P-1] Patient: I have a dry cough.", context)
        self.assertIn("[D-2] Doctor: This supports COPD exacerbation.", context)

    def test_retrieve_context_can_use_lexical_scoring_without_embeddings(self):
        graph = FalkorGraph(
            ref=GraphRef(graph_id="g1"),
            transcript=None,
            note="Patient has fever and cough. Assessment includes pneumonia.",
            input_block=None,
            nodes=[
                FalkorNode("g1:N1", "N1", "metformin", "TREATMENT"),
                FalkorNode("g1:N2", "N2", "pneumonia", "DIAGNOSIS"),
            ],
            edges=[],
        )

        context = retrieve_context(
            question="What diagnosis explains the fever and cough?",
            graph=graph,
            embeddings=None,
            embed_model=None,
            turn_dict={},
            top_k=1,
            edge_limit=5,
            max_source_chars=1000,
        )

        self.assertIn("[DIAGNOSIS] pneumonia", context)
        self.assertNotIn("[TREATMENT] metformin", context)


if __name__ == "__main__":
    unittest.main()
