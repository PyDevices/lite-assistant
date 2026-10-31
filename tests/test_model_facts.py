"""
The curated facts about chat models that their files leave out or get wrong.
"""
import unittest

from lite_assistant import model_facts


class TestModelFacts(unittest.TestCase):
    def test_gemma_4_thinks_and_calls_tools_whatever_its_file_says(self):
        self.assertEqual(model_facts.abilities("gemma-4-E2B-it", {"thinking": False, "tools": False}),
                         {"thinking": True, "tools": True})

    def test_qwen3_and_qwen3_5_are_told_apart(self):
        self.assertEqual(model_facts.sampler("Qwen3.5-4B_mixed_int4", {"top_k": None, "top_p": 0.0, "temperature": 0.0}),
                         {"top_k": 20, "top_p": 0.95, "temperature": 1.0})
        self.assertEqual(model_facts.sampler("Qwen3-0.6B", {}), {"top_k": 20, "top_p": 0.95, "temperature": 0.6})

    def test_the_file_wins_where_it_says(self):
        self.assertEqual(model_facts.sampler("Qwen3-0.6B", {"top_k": 40, "top_p": 0.9, "temperature": 0.5}),
                         {"top_k": 40, "top_p": 0.9, "temperature": 0.5})

    def test_unknown_models_fall_back_to_their_file_then_gemma(self):
        self.assertEqual(model_facts.abilities("SmolLM-135M", {"thinking": True}), {"thinking": True, "tools": False})
        self.assertEqual(model_facts.sampler("SmolLM-135M", {}), model_facts.FALLBACK_SAMPLER)

    def test_models_that_dont_think(self):
        for model_id in ("gemma-3-1b-it", "Qwen2.5-1.5B-Instruct", "Phi-4-mini-instruct"):
            self.assertFalse(model_facts.abilities(model_id, {"thinking": True})["thinking"], model_id)


if __name__ == "__main__":
    unittest.main()
