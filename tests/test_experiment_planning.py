from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from clarifysae_llama.data.ambik_loader import load_ambik_clarification_dataset
from clarifysae_llama.experiments.build_manifest import build_manifest
from clarifysae_llama.experiments.prepare_ambik import prepare_ambik_full_dataset
from clarifysae_llama.experiments.split_ambik import create_ambik_split


class ExperimentPlanningTests(unittest.TestCase):
    def _dataset(self, path: Path, n: int = 1000) -> None:
        categories = ['preferences', 'common_sense_knowledge', 'safety']
        frame = pd.DataFrame({
            'id': [str(i) for i in range(n)],
            'environment_full': [f'env {i}' for i in range(n)],
            'ambiguity_type': [categories[i % len(categories)] for i in range(n)],
            'ambiguous_task': [f'ambiguous task {i}' for i in range(n)],
            'unambiguous_direct': [f'clear task {i}' for i in range(n)],
            'question': [f'question {i}?' for i in range(n)],
            'answer': [f'answer {i}' for i in range(n)],
            'plan_for_clear_task': [f'plan {i}' for i in range(n)],
        })
        frame.to_csv(path, index=False)


    def test_prepare_historical_100_plus_900(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calib = root / 'calib100.csv'
            test = root / 'test900.csv'
            output = root / 'full.csv'
            self._dataset(calib, n=100)
            test_frame = pd.DataFrame({
                'environment_full': [f'test env {i}' for i in range(900)],
                'ambiguity_type': [['preferences', 'common_sense_knowledge', 'safety'][i % 3] for i in range(900)],
                'ambiguous_task': [f'test ambiguous task {i}' for i in range(900)],
                'unambiguous_direct': [f'test clear task {i}' for i in range(900)],
                'question': [f'test question {i}?' for i in range(900)],
                'answer': [f'test answer {i}' for i in range(900)],
                'plan_for_clear_task': [f'test plan {i}' for i in range(900)],
            })
            test_frame.to_csv(test, index=False)
            full, metadata = prepare_ambik_full_dataset(
                calib100_path=calib,
                test900_path=test,
                output_path=output,
            )
            self.assertEqual(len(full), 1000)
            self.assertTrue(full['id'].is_unique)
            self.assertEqual(metadata['total_rows'], 1000)
            self.assertTrue(output.with_suffix('.metadata.json').exists())

    def test_split_and_paired_loader(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / 'ambik.csv'
            split = root / 'split.csv'
            self._dataset(dataset)
            manifest, _ = create_ambik_split(
                dataset_path=dataset,
                output_path=split,
                explore_size=400,
                confirm_size=600,
                smoke_size=20,
                pilot_size=100,
                seed=7,
            )
            self.assertEqual(len(manifest), 1000)
            self.assertEqual((manifest['split'] == 'explore400').sum(), 400)
            self.assertEqual((manifest['split'] == 'confirm600').sum(), 600)
            self.assertEqual(manifest['is_smoke20'].sum(), 20)
            self.assertEqual(manifest['is_pilot100'].sum(), 100)
            self.assertTrue(
                set(manifest.loc[manifest['is_smoke20'], 'example_id']).issubset(
                    set(manifest.loc[manifest['is_pilot100'], 'example_id'])
                )
            )

            paired = load_ambik_clarification_dataset(
                dataset,
                split_manifest=split,
                split_name='explore400',
                instruction_variant='paired',
            )
            self.assertEqual(len(paired), 800)
            self.assertEqual((paired['variant'] == 'ambiguous').sum(), 400)
            self.assertEqual((paired['variant'] == 'clear').sum(), 400)
            self.assertEqual(
                set(paired.loc[paired['variant'] == 'clear', 'ambiguity_type']),
                {'unambiguous_direct'},
            )

            smoke = load_ambik_clarification_dataset(
                dataset, split_manifest=split, split_name='smoke20', instruction_variant='ambiguous'
            )
            pilot = load_ambik_clarification_dataset(
                dataset, split_manifest=split, split_name='pilot100', instruction_variant='ambiguous'
            )
            self.assertEqual(len(smoke), 20)
            self.assertEqual(len(pilot), 100)

    def test_explore_manifest_has_expected_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog = pd.read_csv('manifests/feature_catalog.csv')
            scales = catalog[['model_key', 'layer', 'feature_id']].copy()
            scales['selected_scale'] = 1.0
            scales['scale_method'] = 'positive_q95'
            scale_path = root / 'scales.csv'
            scales.to_csv(scale_path, index=False)

            config = yaml.safe_load(Path('configs/experiments/ambik_explore_v1.yaml').read_text())
            config['steering']['scale']['artifact'] = str(scale_path)
            config_path = root / 'experiment.yaml'
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            manifest_path = root / 'manifest.csv'
            manifest = build_manifest(config_path, manifest_path)

            self.assertEqual(len(catalog), 92)
            self.assertEqual(len(manifest), 92 * 6 + 4)
            self.assertEqual(manifest['run_id'].nunique(), len(manifest))
            self.assertEqual(manifest['feature_id'].notna().sum(), 92 * 6)
            self.assertEqual(manifest['feature_id'].isna().sum(), 4)


if __name__ == '__main__':
    unittest.main()
