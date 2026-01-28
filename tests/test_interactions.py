# test_interactions.py
import unittest
import asyncio
import json # Added for tool result simulation if needed later
from typing import Dict, Any, List
from unittest.mock import MagicMock

import sys
import os
# Add project root to path if running tests directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Assuming schemas and handler are importable
from schemas.stage import Stage, StageOutputSpec, StageRole
from schemas.interaction import Interaction, InteractionRole, Metadata
from schemas.identity import Identity
from schemas.ethical_guidance import EthicalGuidance
from core.stages.interaction_handler import handle_interaction_stage, AUTOGEN_AVAILABLE
from unittest.mock import patch, AsyncMock, MagicMock, call # Added patch, AsyncMock, call
from config import llm_config as actual_llm_config # Import the actual llm_config

# Mock LLMConfig globally for tests that might need it - REMOVED as we'll use actual_llm_config in assertions
# MOCK_LLM_CONFIG_DICT = {"config_list": [{"model": "mock-model"}]}
# MOCK_LLM_CONFIG_OBJ = MagicMock()
# MOCK_LLM_CONFIG_OBJ.model_dump.return_value = MOCK_LLM_CONFIG_DICT

# Mock config module before importing handler
# Note: This might need adjustment based on how config is actually structured/imported
# If llm_config is directly imported, patching might be needed where it's used.
# For simplicity here, we assume it can be passed or mocked via engine.
# Let's try patching it directly in the handler's namespace if needed.

# Removed class-level patch
class TestInteractionHandler(unittest.TestCase):

    def setUp(self):
        # Basic setup reusable across tests
        self.mock_engine = MagicMock()
        # Setup mock identity registry
        self.mock_identity_registry = MagicMock()
        self.mock_engine.identity_registry = self.mock_identity_registry

        # Default pipeline-level identity/guidance (can be overridden in tests)
        self.mock_pipeline_identity = Identity(id="default_pipe_id", description="Default Pipeline Identity")
        self.mock_pipeline_guidance = EthicalGuidance(id="default_pipe_guide", description="Default Guidance", prompt_template="Default Guidance Template") # Added prompt_template
        self.active_guardrails: List = []

    def _run_handler(self, stage_dict: Dict[str, Any], context: Dict[str, Any], mock_llm_config_patch=None) -> tuple[Dict[str, Any], List[Interaction], Any]:
        """Helper to run the async handler."""
        # Ensure required fields for Stage are present
        if 'outputs' not in stage_dict:
             stage_dict['outputs'] = {"spec": {}} # Add default empty spec if missing
        if 'id' not in stage_dict:
             stage_dict['id'] = 'test_stage' # Add default id if missing
        if 'type' not in stage_dict:
             stage_dict['type'] = 'interaction' # Add default type if missing

        stage = Stage(**stage_dict)
        return asyncio.run(handle_interaction_stage(
            stage=stage,
            pipeline_context=context,
            engine_instance=self.mock_engine,
            identity=self.mock_pipeline_identity, # Pass default pipeline identity
            guidance=self.mock_pipeline_guidance, # Pass default pipeline guidance
            active_guardrails=self.active_guardrails
        ))

    def test_multi_line_prompt_parsing(self): # Removed patch arg
        """Test parsing of multi-line prompts with Role: prefixes."""
        stage_dict = {
            "id": "test_multi",
            "type": "interaction",
            "prompt": "User: Hello\nAssistant: Hi there!\nUser: How are you?",
            "outputs": {"spec": {"discussion_transcript": "text"}}
        }
        context = {}
        updated_context, interactions, status = self._run_handler(stage_dict, context)

        self.assertIsNone(status)
        self.assertEqual(len(interactions), 3)
        self.assertEqual(interactions[0].role, "user")
        self.assertEqual(interactions[0].content, "Hello")
        self.assertEqual(interactions[1].role, "assistant")
        self.assertEqual(interactions[1].content, "Hi there!")
        self.assertEqual(interactions[2].role, "user")
        self.assertEqual(interactions[2].content, "How are you?")
        self.assertIn("test_multi", updated_context, "Stage ID not found in updated context")
        self.assertIn("discussion_transcript", updated_context.get("test_multi", {}), "Output label not found in stage context")
        self.assertEqual(updated_context["test_multi"]["discussion_transcript"], "User: Hello\nAssistant: Hi there!\nUser: How are you?")

    def test_single_line_prompt_no_prefix(self): # Removed patch arg
        """Test handling of single-line prompts without Role: prefix (treated as system)."""
        stage_dict = {
            "id": "test_single",
            "type": "interaction",
            "prompt": "This is a system instruction.",
            "outputs": {"spec": {}} # Use dict with empty spec for validation
        }
        context = {}
        updated_context, interactions, status = self._run_handler(stage_dict, context)

        self.assertIsNone(status)
        self.assertEqual(len(interactions), 1)
        self.assertEqual(interactions[0].role, "system")
        self.assertEqual(interactions[0].content, "This is a system instruction.")
        # Check that stage_id is not added to context if no outputs defined/generated
        self.assertNotIn("test_single", updated_context) # No outputs defined, so stage_id shouldn't be key

    def test_placeholder_resolution(self): # Removed patch arg
        """Test placeholder resolution within a multi-line prompt."""
        stage_dict = {
            "id": "test_resolve",
            "type": "interaction",
            "prompt": "User: What was the plan?\nAssistant: The plan was {plan_stage.plan_text}",
            "outputs": {"spec": {"discussion_transcript": "text"}}
        }
        context = {
            "plan_stage": {"plan_text": "Step 1, Step 2"}
        }
        updated_context, interactions, status = self._run_handler(stage_dict, context)

        self.assertIsNone(status)
        self.assertEqual(len(interactions), 2)
        self.assertEqual(interactions[0].content, "What was the plan?")
        self.assertEqual(interactions[1].role, "assistant")
        self.assertEqual(interactions[1].content, "The plan was Step 1, Step 2")
        self.assertIn("test_resolve", updated_context)
        self.assertIn("discussion_transcript", updated_context.get("test_resolve", {}))
        self.assertIn("Assistant: The plan was Step 1, Step 2", updated_context["test_resolve"]["discussion_transcript"])

    def test_unresolved_placeholder(self): # Removed patch arg
        """Test handling of unresolved placeholders (should log warning, leave as is)."""
        stage_dict = {
            "id": "test_unresolved",
            "type": "interaction",
            "prompt": "User: Info is {missing_stage.info}",
            "outputs": {"spec": {"discussion_transcript": "text"}}
        }
        context = {}
        # Expecting a warning log, but the handler should complete without error status
        updated_context, interactions, status = self._run_handler(stage_dict, context)

        self.assertIsNone(status) # Handler itself shouldn't fail
        self.assertEqual(len(interactions), 1)
        self.assertEqual(interactions[0].content, "Info is {missing_stage.info}") # Placeholder remains
        self.assertIn("test_unresolved", updated_context)
        self.assertIn("discussion_transcript", updated_context.get("test_unresolved", {}))
        self.assertEqual(updated_context["test_unresolved"]["discussion_transcript"], "User: Info is {missing_stage.info}")

    def test_output_generation(self): # Removed patch arg
        """Test that the discussion_transcript is correctly generated and stored."""
        stage_dict = {
            "id": "test_output",
            "type": "interaction",
            "prompt": "User: Test",
            "outputs": {"spec": {"discussion_transcript": "text"}}
        }
        context = {}
        updated_context, interactions, status = self._run_handler(stage_dict, context)

        self.assertIsNone(status)
        self.assertIn("test_output", updated_context)
        self.assertIn("discussion_transcript", updated_context["test_output"])
        self.assertEqual(updated_context["test_output"]["discussion_transcript"], "User: Test")

    def test_no_prompt(self): # Removed patch arg
        """Test behavior when prompt is missing or empty."""
        stage_dict = {
            "id": "test_no_prompt",
            "type": "interaction",
            "prompt": "", # Empty prompt
            "outputs": {"spec": {"discussion_transcript": "text"}}
        }
        context = {}
        updated_context, interactions, status = self._run_handler(stage_dict, context)
        self.assertIsNone(status)
        self.assertEqual(len(interactions), 0) # No interactions generated
        self.assertIn("test_no_prompt", updated_context)
        self.assertEqual(updated_context["test_no_prompt"]["discussion_transcript"], "") # Empty transcript

        stage_dict_none = {
            "id": "test_none_prompt",
            "type": "interaction",
            # "prompt": None, # Missing prompt key
            "outputs": {"spec": {"discussion_transcript": "text"}}
        }
        updated_context_none, interactions_none, status_none = self._run_handler(stage_dict_none, context)
        self.assertIsNone(status_none)
        self.assertEqual(len(interactions_none), 0)
        self.assertIn("test_none_prompt", updated_context_none)
        self.assertEqual(updated_context_none["test_none_prompt"]["discussion_transcript"], "")

    # --- GroupChat Specific Tests ---

    # Only run if autogen is actually available
    @unittest.skipUnless(AUTOGEN_AVAILABLE, "Autogen library not installed, skipping GroupChat tests")
    # Removed patch for llm_config here
    @patch('core.stages.interaction_handler.AssistantAgent')
    @patch('core.stages.interaction_handler.GroupChatManager')
    @patch('core.stages.interaction_handler.UserProxyAgent')
    @patch('core.stages.interaction_handler.GroupChat')
    # Removed model_dump patch
    def test_group_chat_participant_configs(self, MockGroupChat, MockUserProxyAgent, MockGroupChatManager, MockAssistantAgent): # Removed mock_model_dump arg
        """Test GroupChat uses participant_configs for identity."""

        # --- Mock Setup ---
        # Configure MockAssistantAgent to set the name attribute
        agent_instances = {}
        def assistant_agent_side_effect(*args, **kwargs):
            instance = MagicMock() # Create a plain mock instance
            name = kwargs.get('name')
            instance.name = name # Set the name attribute
            agent_instances[name] = instance
            # print(f"Mock AssistantAgent created: name={name}") # Debug print
            return instance
        MockAssistantAgent.side_effect = assistant_agent_side_effect

        # Define a side effect function that returns mocks behaving like dicts for .get('description')
        def mock_get_identity_side_effect(identity_id):
            if identity_id == "Jiminies":
                mock_dict_like = MagicMock()
                # Configure .get() specifically for 'description' key
                mock_dict_like.get.side_effect = lambda key, default=None: "Intelligent Birds" if key == "description" else default
                return mock_dict_like
            elif identity_id == "NIMHs":
                mock_dict_like = MagicMock()
                # Configure .get() specifically for 'description' key
                mock_dict_like.get.side_effect = lambda key, default=None: "Intelligent Rodents" if key == "description" else default
                return mock_dict_like
            else:
                return None

        # Use the side effect function for the mock
        self.mock_identity_registry.get_identity.side_effect = mock_get_identity_side_effect

        # Mock Autogen classes behavior
        mock_group_chat_instance = MockGroupChat.return_value
        mock_group_chat_instance.messages = [ # Simulate some chat history
             {"role": "assistant", "name": "jiminy_ethicist", "content": "Hello from Jiminy!"},
             {"role": "assistant", "name": "nimh_representative", "content": "Greetings from NIMH! TERMINATE"}
        ]
        mock_manager_instance = MockGroupChatManager.return_value
        mock_initiator_instance = MockUserProxyAgent.return_value
        mock_initiator_instance.name = "InitiatorProxy" # Set name for role mapping

        # Mock the async initiate_chat - needs to be an AsyncMock
        mock_initiator_instance.a_initiate_chat = AsyncMock(return_value=None)

        # --- Stage Definition ---
        stage_dict = {
            "id": "test_gc_config",
            "type": "interaction",
            "participants": ["jiminy_ethicist", "nimh_representative"],
            "participant_configs": {
                "jiminy_ethicist": {"identity_id": "Jiminies"},
                "nimh_representative": {"identity_id": "NIMHs"}
            },
            "prompt": "Discuss the grain situation.",
            "outputs": {"spec": {"negotiation_transcript": "text"}}
        }
        context = {}

        # --- Run Handler ---
        updated_context, interactions, status = self._run_handler(stage_dict, context)

        # --- Assertions ---
        self.assertIsNone(status) # Expect success

        # Check AssistantAgent calls - Expect the actual strings now due to the refined mock
        expected_jiminy_sys_msg = f"You are jiminy_ethicist. Follow this guidance: {self.mock_pipeline_guidance.prompt_template} Consider this identity context: Intelligent Birds After the discussion concludes or reaches a natural stopping point, respond with TERMINATE."
        expected_nimh_sys_msg = f"You are nimh_representative. Follow this guidance: {self.mock_pipeline_guidance.prompt_template} Consider this identity context: Intelligent Rodents After the discussion concludes or reaches a natural stopping point, respond with TERMINATE."

        expected_calls = [
            # Call for jiminy_ethicist
            call(name='jiminy_ethicist',
                 system_message=expected_jiminy_sys_msg, # Use pre-formatted string with literal description
                 llm_config=actual_llm_config, # Expect the actual config object
                 is_termination_msg=unittest.mock.ANY), # Check lambda existence, not exact instance
            # Call for nimh_representative
            call(name='nimh_representative',
                 system_message=expected_nimh_sys_msg, # Use pre-formatted string with literal description
                 llm_config=actual_llm_config, # Expect the actual config object
                 is_termination_msg=unittest.mock.ANY)
        ]
        MockAssistantAgent.assert_has_calls(expected_calls, any_order=False) # Order matters here

        # Check interactions generated from mock history
        self.assertEqual(len(interactions), 2)
        self.assertEqual(interactions[0].role, "assistant")
        self.assertEqual(interactions[0].content, "Hello from Jiminy!")
        self.assertEqual(interactions[1].role, "assistant")
        self.assertEqual(interactions[1].content, "Greetings from NIMH!") # TERMINATE should be stripped

        # Check output context
        self.assertIn("test_gc_config", updated_context)
        self.assertIn("negotiation_transcript", updated_context["test_gc_config"])
        # The output is the last agent message
        self.assertEqual(updated_context["test_gc_config"]["negotiation_transcript"], "Greetings from NIMH!")


    @unittest.skipUnless(AUTOGEN_AVAILABLE, "Autogen library not installed, skipping GroupChat tests")
    # Removed patch for llm_config here
    @patch('core.stages.interaction_handler.AssistantAgent')
    @patch('core.stages.interaction_handler.GroupChatManager')
    @patch('core.stages.interaction_handler.UserProxyAgent')
    @patch('core.stages.interaction_handler.GroupChat')
    # Removed model_dump patch
    def test_group_chat_fallback_identity(self, MockGroupChat, MockUserProxyAgent, MockGroupChatManager, MockAssistantAgent): # Removed mock_model_dump arg
        """Test GroupChat falls back to pipeline identity if specific one not found or configured."""

        # --- Mock Setup ---
        # Configure MockAssistantAgent side effect for this test too
        agent_instances = {}
        def assistant_agent_side_effect_fallback(*args, **kwargs):
            instance = MagicMock() # Create a plain mock instance
            name = kwargs.get('name')
            instance.name = name
            agent_instances[name] = instance
            return instance
        MockAssistantAgent.side_effect = assistant_agent_side_effect_fallback # Use different name to avoid conflict
        # Reset side_effect for this specific test to simulate not finding IDs
        self.mock_identity_registry.get_identity.side_effect = lambda identity_id: None

        # Mock Autogen classes
        mock_group_chat_instance = MockGroupChat.return_value
        mock_group_chat_instance.messages = [] # No history needed for this check
        mock_manager_instance = MockGroupChatManager.return_value
        mock_initiator_instance = MockUserProxyAgent.return_value
        mock_initiator_instance.name = "InitiatorProxy"
        mock_initiator_instance.a_initiate_chat = AsyncMock(return_value=None)

        # --- Stage Definition ---
        stage_dict = {
            "id": "test_gc_fallback",
            "type": "interaction",
            "participants": ["agent_one", "agent_two"],
            # No participant_configs provided
            "prompt": "Discuss.",
            "outputs": {"spec": {"transcript": "text"}}
        }
        context = {}

        # --- Run Handler ---
        updated_context, interactions, status = self._run_handler(stage_dict, context)

        # --- Assertions ---
        # Check AssistantAgent calls - should use pipeline identity description and guidance template
        expected_calls = [
            call(name='agent_one',
                 system_message=f"You are agent_one. Follow this guidance: {self.mock_pipeline_guidance.prompt_template} Consider this identity context: {self.mock_pipeline_identity.description} After the discussion concludes or reaches a natural stopping point, respond with TERMINATE.", # Use prompt_template
                 llm_config=actual_llm_config, # Expect the actual config object
                 is_termination_msg=unittest.mock.ANY),
            call(name='agent_two',
                 system_message=f"You are agent_two. Follow this guidance: {self.mock_pipeline_guidance.prompt_template} Consider this identity context: {self.mock_pipeline_identity.description} After the discussion concludes or reaches a natural stopping point, respond with TERMINATE.", # Use prompt_template
                 llm_config=actual_llm_config, # Expect the actual config object
                 is_termination_msg=unittest.mock.ANY)
        ]
        MockAssistantAgent.assert_has_calls(expected_calls, any_order=False)


if __name__ == '__main__':
    unittest.main()
