import unittest

import pandas as pd

from clarifysae_llama.data.prompting import build_direct_question_prompt
from clarifysae_llama.eval.metrics import aggregate_metrics
from clarifysae_llama.utils.parsing import assess_direct_question_output


class DirectQuestionProtocolTests(unittest.TestCase):
    def test_direct_prompt_contains_operational_clear_rule(self):
        prompt = build_direct_question_prompt(
            'a red mug and a blue mug',
            'put the red mug in the top drawer',
        )
        self.assertIn('output exactly NONE', prompt)
        self.assertIn('multiple methods could work', prompt)
        self.assertIn('materially different valid outcomes', prompt)
        self.assertIn('one question ending with a question mark', prompt)

    def test_direct_question_parser_exact_and_recoverable(self):
        none = assess_direct_question_output('NONE')
        self.assertTrue(none['output_exact_valid'])
        self.assertFalse(none['predicted_ambiguous'])
        self.assertEqual(none['question'], [])

        question = assess_direct_question_output('Which drawer should I use?')
        self.assertTrue(question['output_exact_valid'])
        self.assertTrue(question['predicted_ambiguous'])
        self.assertEqual(question['question'], ['Which drawer should I use?'])

        recovered = assess_direct_question_output('Answer: Which drawer should I use?')
        self.assertFalse(recovered['output_exact_valid'])
        self.assertTrue(recovered['output_schema_valid'])
        self.assertTrue(recovered['output_recoverable_parse'])
        self.assertEqual(recovered['question'], ['Which drawer should I use?'])

        invalid = assess_direct_question_output('The task is clear.')
        self.assertFalse(invalid['output_schema_valid'])
        self.assertIsNone(invalid['predicted_ambiguous'])

    def test_aggregate_reports_ambiguous_only_resolution(self):
        rows = pd.DataFrame([
            {
                'id': 'a1', 'source_id': '1', 'variant': 'ambiguous',
                'ambiguity_type': 'preferences', 'gold_ambiguous': True,
                'predicted_ambiguous': True, 'ambiguity_decision_correct': True,
                'model_questions': ['Which drawer?'], 'num_questions': 1,
                'asked_question': True, 'model_question_first_similarity': 0.9,
                'model_question_best_similarity': 0.9, 'resolved_proxy_first': True,
                'resolved_proxy_any': True, 'output_exact_valid': True,
                'output_schema_valid': True, 'output_recoverable_parse': False,
            },
            {
                'id': 'a2', 'source_id': '2', 'variant': 'ambiguous',
                'ambiguity_type': 'safety', 'gold_ambiguous': True,
                'predicted_ambiguous': False, 'ambiguity_decision_correct': False,
                'model_questions': [], 'num_questions': 0, 'asked_question': False,
                'model_question_first_similarity': 0.0,
                'model_question_best_similarity': 0.0, 'resolved_proxy_first': False,
                'resolved_proxy_any': False, 'output_exact_valid': True,
                'output_schema_valid': True, 'output_recoverable_parse': False,
            },
            {
                'id': 'c1', 'source_id': '1', 'variant': 'clear',
                'ambiguity_type': 'unambiguous_direct', 'gold_ambiguous': False,
                'predicted_ambiguous': False, 'ambiguity_decision_correct': True,
                'model_questions': [], 'num_questions': 0, 'asked_question': False,
                'model_question_first_similarity': 0.0,
                'model_question_best_similarity': 0.0, 'resolved_proxy_first': False,
                'resolved_proxy_any': False, 'output_exact_valid': True,
                'output_schema_valid': True, 'output_recoverable_parse': False,
            },
        ])
        overall, _ = aggregate_metrics(rows)
        result = overall.iloc[0]
        self.assertEqual(result['resolved_proxy_first_rate'], 1 / 3)
        self.assertEqual(result['resolved_proxy_first_rate_ambiguous'], 0.5)
        self.assertEqual(result['resolution_first_given_asked_ambiguous'], 1.0)
        self.assertEqual(result['asking_rate_ambiguous'], 0.5)
        self.assertEqual(result['overasking_rate_clear'], 0.0)



if __name__ == '__main__':
    unittest.main()
