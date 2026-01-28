import unittest
import json
from pathlib import Path
import logging # Import logging

# Adjust import path now that this file is in tests/
# Only import load_all_pipelines as others are not directly exposed
from config.loader import load_all_pipelines
from schemas.pipeline import Pipeline

# Configure logging for tests (optional, can help debugging)
logging.basicConfig(level=logging.WARNING) # Set to WARNING or ERROR to avoid clutter
logger = logging.getLogger(__name__)

class TestConfigLoader(unittest.TestCase):

    # Removed tests for load_identities, load_guidances, load_guardrails
    # as these functions are not directly exposed by config.loader

    def test_load_all_pipelines(self):
        """Test loading all pipeline configurations."""
        # This might log warnings for invalid pipelines, which is expected
        all_pipelines = load_all_pipelines()
        self.assertIsInstance(all_pipelines, dict)
        self.assertGreater(len(all_pipelines), 0, "No pipelines were loaded.")
        # Check for a known valid pipeline ID (adjust if needed)
        # Use the robust_scenario_pipeline ID as it's central to recent work
        self.assertIn("robust_scenario_pipeline_v2", all_pipelines,
                      f"Expected pipeline 'robust_scenario_pipeline_v2' not found. Loaded: {list(all_pipelines.keys())}")
        pipeline = all_pipelines["robust_scenario_pipeline_v2"]
        self.assertIsInstance(pipeline, Pipeline)
        self.assertEqual(pipeline.id, "robust_scenario_pipeline_v2")
        logger.info(f"Loaded {len(all_pipelines)} pipelines.")

    def test_load_specific_pipeline_content(self):
        """Test loading and verifying content of a specific pipeline."""
        pipeline_id_to_test = "robust_scenario_pipeline_v2" # Use the correct ID
        all_pipelines = load_all_pipelines()

        self.assertIn(pipeline_id_to_test, all_pipelines,
                      f"Pipeline ID '{pipeline_id_to_test}' not found.")

        loaded_config = all_pipelines[pipeline_id_to_test]

        # Verify basic structure and content
        self.assertEqual(loaded_config.id, pipeline_id_to_test)
        self.assertIsInstance(loaded_config.stages, list)
        self.assertGreater(len(loaded_config.stages), 0)
        # Check a specific stage detail (e.g., the tool stage)
        tool_stage = next((s for s in loaded_config.stages if s.id == "calculate_tax_impact"), None)
        self.assertIsNotNone(tool_stage, "calculate_tax_impact stage not found")
        self.assertEqual(tool_stage.type, "tool")
        self.assertEqual(tool_stage.tool_id, "simple_calculator")

if __name__ == '__main__':
    unittest.main()
