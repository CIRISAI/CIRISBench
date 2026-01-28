# test_tools.py
import unittest
import asyncio
import json
import re
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch, AsyncMock # Import AsyncMock

# Assuming schemas, handler, and tools are importable
# Adjust imports based on actual project structure if needed
from schemas.stage import Stage
from schemas.interaction import Interaction
from core.tools import simple_calculator, TOOL_REGISTRY
from core.stages.tool_handler import handle_tool_stage

# Mock agent classes minimally for type hints if needed, but we patch interactions
class AssistantAgent: pass
class UserProxyAgent: pass
AUTOGEN_AVAILABLE = True # Assume available for patching to work

# Mock ChatResult class for type hinting and structure
class MockChatResult:
    def __init__(self, chat_history: List[Dict[str, Any]]):
        self.chat_history = chat_history

class TestToolHandlerAndTools(unittest.TestCase):

    def setUp(self):
        self.mock_engine = MagicMock()
        # Mock identity and guidance with necessary attributes if handler uses them
        self.mock_identity = MagicMock()
        self.mock_identity.description = "Mock Identity"
        self.mock_guidance = MagicMock()
        self.mock_guidance.prompt_template = "Follow mock guidance."
        self.active_guardrails: List = []
        self.pipeline_context: Dict[str, Any] = {"previous_stage": {"output": "some_value"}}

    # --- Test Tool Functions Directly (Keep these) ---

    def test_simple_calculator_success(self):
        result = simple_calculator("5 * 2.5")
        self.assertEqual(result, {"result": "12.5"})

    def test_simple_calculator_invalid_format(self):
        result = simple_calculator("5 + 2") # Only multiplication supported
        self.assertIn("error", result)
        self.assertIn("multiplication", result["error"])

    def test_simple_calculator_invalid_input(self):
        result = simple_calculator("five * two")
        self.assertIn("error", result)
        # Check specific error message from the tool function
        self.assertIn("Invalid expression format", result["error"])

    # --- Test Tool Handler with Mocked Agent Interaction ---

    def _run_handler_with_mock_chat(self, stage_dict: Dict[str, Any], mock_chat_history: List[Dict[str, Any]]) -> tuple[Dict[str, Any], List[Interaction], Any]:
        """Helper to run the handler with mocked a_initiate_chat."""
        stage = Stage(**stage_dict)
        mock_result = MockChatResult(chat_history=mock_chat_history)

        # Patch the a_initiate_chat method within the handler's scope
        with patch('core.stages.tool_handler.UserProxyAgent.a_initiate_chat', new_callable=AsyncMock) as mock_initiate_chat:
            mock_initiate_chat.return_value = mock_result

            # Run the handler function (which is async) using asyncio.run
            result = asyncio.run(handle_tool_stage(
                stage=stage,
                pipeline_context=self.pipeline_context,
                engine_instance=self.mock_engine,
                identity=self.mock_identity,
                guidance=self.mock_guidance,
                active_guardrails=self.active_guardrails
            ))
        return result

    def test_tool_handler_simple_calculator(self):
        """Test handler with simple_calculator tool (mocked chat)."""
        stage_dict = {
            "id": "calc_stage", "type": "tool", "tool_id": "simple_calculator",
            "prompt": "Calculate 5200 * 0.22",
            "outputs": {"spec": {"calculation_result": "text"}}
        }
        # Mock history simulating successful tool call and assistant reply
        mock_history = [
            {"role": "user", "name": "ExecutorProxy_calc_stage", "content": stage_dict["prompt"]},
            {"role": "assistant", "name": "Assistant_calc_stage", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "simple_calculator", "arguments": '{"expression":"5200 * 0.22"}'}}]},
            # Use tool_responses format
            {"role": "tool", "tool_responses": [{"tool_call_id": "call_123", "role": "tool", "content": "{'result': '1144.0'}"}]}, # Content as string dict literal
            {"role": "assistant", "name": "Assistant_calc_stage", "content": "The result is 1144.0. TERMINATE"} # The final reply we want
        ]
        updated_context, interactions, status = self._run_handler_with_mock_chat(stage_dict, mock_history)

        self.assertIsNone(status, f"Handler failed with status: {status}")
        self.assertGreater(len(interactions), 0)
        final_interaction = interactions[-1]
        self.assertEqual(final_interaction.role, "assistant")
        # Handler should extract the assistant's final reply
        self.assertEqual(final_interaction.content, "The result is 1144.0. TERMINATE")
        self.assertIn("calc_stage", updated_context)
        self.assertEqual(updated_context["calc_stage"]["calculation_result"], "The result is 1144.0. TERMINATE")

    def test_tool_handler_missing_tool(self):
        """Test handler when tool_id is not in registry (no chat needed)."""
        """Test handler when tool_id is not in registry."""
        stage_dict = {
            "id": "missing_tool_stage",
            "type": "tool",
            "tool_id": "non_existent_tool",
            "prompt": "Use a tool that doesn't exist.",
            "outputs": {"spec": {"result": "text"}}
        }
        # No need to mock chat, error should happen before agent interaction
        stage = Stage(**stage_dict)
        updated_context, interactions, status = asyncio.run(handle_tool_stage(
            stage=stage,
            pipeline_context=self.pipeline_context,
            engine_instance=self.mock_engine,
            identity=self.mock_identity,
            guidance=self.mock_guidance,
            active_guardrails=self.active_guardrails
        ))

        self.assertIsNotNone(status, "Status should indicate an error for missing tool")
        self.assertIn("error: specified tool", status) # Check specific error message
        self.assertIn("'non_existent_tool'", status)
        self.assertIn("not registered", status)

    def test_tool_handler_tool_error(self):
        """Test handler when the tool function itself returns an error (mocked chat)."""
        stage_dict = {
            "id": "tool_error_stage", "type": "tool", "tool_id": "simple_calculator",
            "prompt": "Use the simple_calculator tool for 'invalid expression'.",
            "outputs": {"spec": {"calc_result": "text"}}
        }
        # Simulate the tool function returning an error dict
        expected_error_dict = simple_calculator("invalid expression")
        expected_error_msg = expected_error_dict.get("error", "UNKNOWN_ERROR")
        # Mock history where tool returns error, and assistant relays it
        mock_history = [
            {"role": "user", "name": "ExecutorProxy_tool_error_stage", "content": stage_dict["prompt"]},
            {"role": "assistant", "name": "Assistant_tool_error_stage", "tool_calls": [{"id": "call_err", "type": "function", "function": {"name": "simple_calculator", "arguments": '{"expression":"invalid expression"}'}}]},
            # Tool returns the error dict as a string literal
            {"role": "tool", "tool_responses": [{"tool_call_id": "call_err", "role": "tool", "content": str(expected_error_dict)}]},
            # Assistant relays the error (or a summary of it)
            {"role": "assistant", "name": "Assistant_tool_error_stage", "content": f"The tool failed with error: {expected_error_msg}. TERMINATE"}
        ]

        updated_context, interactions, status = self._run_handler_with_mock_chat(stage_dict, mock_history)

        # Status should be None because the handler successfully processed the chat,
        # even though the *tool* reported an error. The error is in the content.
        self.assertIsNone(status, f"Handler status should be None, but got: {status}")

        # The final interaction content should be the assistant's message relaying the error
        self.assertGreater(len(interactions), 0)
        final_interaction = interactions[-1]
        self.assertEqual(final_interaction.role, "assistant")
        expected_content = f"The tool failed with error: {expected_error_msg}. TERMINATE"
        self.assertEqual(final_interaction.content, expected_content)

        # Check context reflects the assistant's error message
        self.assertIn("tool_error_stage", updated_context)
        self.assertEqual(updated_context["tool_error_stage"]["calc_result"], expected_content)


if __name__ == '__main__':
    # Adjust path if needed when running directly
    # import sys
    # import os
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    unittest.main()
