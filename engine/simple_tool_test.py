# simple_tool_test_v4.py # Using AssistantAgent + UserProxyAgent
import asyncio
import os
import json
import re
from dotenv import load_dotenv
from typing import Annotated

# Load environment variables
load_dotenv()

try:
    # Import necessary agents
    from autogen import AssistantAgent, UserProxyAgent
    print("Imported Autogen agents.")
except ImportError:
    print("Autogen components not found. Please install pyautogen.")
    exit()

# Import shared config, semaphore, and specific tool schema
try:
    from config import llm_config as shared_llm_config, semaphore # Import shared config and semaphore
    from core.tools import simple_calculator, simple_calculator_schema
    # Import schemas needed for system message construction (or use mocks/None)
    from schemas.ethical_guidance import EthicalGuidance
    from schemas.identity import Identity
    print("Imported shared llm_config, tools, and schemas.")
except ImportError:
    print("Could not import config/tools/schemas. Ensure PYTHONPATH is set correctly or run from project root.")
    exit()

# --- Configuration ---
# Use the shared llm_config, but add the specific tool for this test
if not shared_llm_config:
     print("Error: Shared llm_config from config.py is not available.")
     exit()

# Create a copy to modify for this specific test agent
assistant_llm_config = shared_llm_config.model_dump(exclude_unset=True) # Use model_dump if it's Pydantic
# Ensure 'tools' key exists and add the specific tool schema
assistant_llm_config["tools"] = assistant_llm_config.get("tools", []) + [
    {"type": "function", "function": simple_calculator_schema}
]
# Remove api_key if present, as autogen usually gets it from config_list
# assistant_llm_config.pop("api_key", None) # Let autogen handle keys from config_list

# --- Agent Setup ---

# Replicate system message construction from tool_handler.py
tool_id = "simple_calculator" # Define the tool ID for the message
# Mock guidance/identity or use None as they are Optional in the handler
mock_guidance = None # Or: EthicalGuidance(id="test_guidance", prompt_template="Follow test guidance.")
mock_identity = None # Or: Identity(id="test_identity", description="Test identity.")

system_message = "You are a helpful AI assistant capable of using tools."
system_message += f" You have access to the '{tool_id}' tool. Use it when appropriate based on the user request."
if mock_guidance and mock_guidance.prompt_template:
    system_message += f" {mock_guidance.prompt_template}"
# Add identity context if needed (commented out in handler, so keep it commented)
# if mock_identity: system_message += f" Context: {mock_identity.description}"

print(f"Constructed System Message: {system_message}")

# Assistant Agent (decides to use the tool)
assistant = AssistantAgent(
    name="CalculatorAssistant",
    system_message=system_message, # Use the constructed system message
    llm_config=assistant_llm_config, # Use the modified config
)

# User Proxy Agent (executes the tool code)
# Use parameters from tool_handler.py
user_proxy = UserProxyAgent(
    name="UserProxy", # Keep name simple for test clarity
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5, # Use handler's value
    is_termination_msg=lambda x: isinstance(x.get("content"), str) and "TERMINATE" in x.get("content", ""), # Use handler's check
    code_execution_config=False,
    llm_config=False, # No LLM needed for proxy
    # Define the function map for the UserProxyAgent
    # This tells it which functions it is allowed to execute when called by the assistant
    function_map={
        "simple_calculator": simple_calculator
    }
)

# --- Initiate Chat ---
async def run_two_agent_chat():
    # Prompt for the UserProxyAgent to send to the AssistantAgent
    user_prompt = "please use your tools to solve this and reply with the solution or answer: Calculate 5200 * 0.22"
    print(f"\nUser Prompt to Assistant: {user_prompt}\n")

    # UserProxy initiates the chat with the Assistant, wrapped in semaphore
    async with semaphore:
        print("Semaphore acquired.")
        await user_proxy.a_initiate_chat(
            assistant,
            message=user_prompt,
        )
    print("Semaphore released.")


    print("\n--- Conversation History (User Proxy Perspective) ---")
    # Print history from the user proxy's perspective
    history = user_proxy.chat_messages.get(assistant, [])
    for msg in history:
         content = msg.get('content')
         tool_calls = msg.get('tool_calls')
         role = msg.get('role')
         name = msg.get('name', 'N/A') # Use N/A if name is not present

         print(f"Role: {role}, Name: {name}")
         if content:
             print(f"  Content: {str(content).strip()}")
         if tool_calls:
             # Autogen formats tool calls as a list of dicts
             print(f"  Tool Calls: {tool_calls}")
             # Also print the function return if it's the next message
             current_index = history.index(msg)
             if current_index + 1 < len(history):
                 next_msg = history[current_index+1]
                 if next_msg.get("role") == "tool" and next_msg.get("tool_call_id") == tool_calls[0].get("id"):
                     print(f"  Tool Result ({next_msg.get('name')}): {next_msg.get('content')}")


    print("----------------------------------------------------")

    # --- Add History Processing Logic (Simulating Updated tool_handler.py) ---
    print("\n--- Processing History (Simulating Updated tool_handler.py) ---")
    import ast # For literal_eval

    tool_result_content = None
    tool_return_found_index = -1
    assistant_final_reply_content = None # Renamed for clarity
    raw_response_content = "Error: Could not determine stage output." # Default
    status = "error: no usable output" # Default

    # Iterate forwards to find the tool result message first
    for i, msg in enumerate(history):
        # Priority 1: Check for 'tool_responses' field
        if msg.get("tool_responses") and isinstance(msg["tool_responses"], list) and len(msg["tool_responses"]) > 0:
            tool_response_item = msg["tool_responses"][0]
            tool_result_content = tool_response_item.get("content") # Keep for potential fallback
            tool_return_found_index = i
            print(f"Found tool result via 'tool_responses' field in msg {i}: {tool_result_content}")
            break
        # Priority 2: Fallback to checking role == "tool"
        elif msg.get("role") == "tool":
            tool_result_content = msg.get("content") # Keep for potential fallback
            tool_return_found_index = i
            print(f"Found tool result via role='tool' in msg {i}: {tool_result_content}")
            break

    # If tool result was found, look for the *next* message immediately after it
    if tool_return_found_index != -1 and (tool_return_found_index + 1) < len(history):
         next_msg_index = tool_return_found_index + 1
         next_msg = history[next_msg_index]
         assistant_final_reply_content = next_msg.get("content")
         print(f"Found message immediately after tool result at index {next_msg_index} (Role: {next_msg.get('role')}, Name: {next_msg.get('name')}): {assistant_final_reply_content}")

    # --- Determine Final Content and Status ---
    if assistant_final_reply_content is not None:
         # Prioritize the message content immediately following the tool result
         raw_response_content = str(assistant_final_reply_content) # Ensure string
         status = None # Success
         print(f"Using content from message after tool result as output.")

    elif tool_return_found_index != -1:
        # Fallback to processing the tool result itself if no message followed
        print(f"Warning: Tool result found at index {tool_return_found_index}, but no subsequent message found or it had no content. Processing tool result directly.")
        tool_result_dict = None
        if isinstance(tool_result_content, dict):
            tool_result_dict = tool_result_content
        elif isinstance(tool_result_content, str):
            try:
                tool_result_dict = json.loads(tool_result_content)
            except json.JSONDecodeError:
                try:
                    tool_result_dict = ast.literal_eval(tool_result_content)
                except (ValueError, SyntaxError, TypeError):
                    raw_response_content = tool_result_content
                    status = "warning: tool result is raw string"
                    print(f"Warning: Tool return content is not valid JSON or dict literal: {tool_result_content}")
                    tool_result_dict = None
            except Exception as parse_err:
                 raw_response_content = f"Error processing tool result string: {parse_err}"
                 status = "error: failed to process tool result string"
                 print(f"Error processing tool return content string: {parse_err}")
                 tool_result_dict = None

        if tool_result_dict is not None:
            if "error" in tool_result_dict:
                error_message = tool_result_dict['error']
                raw_response_content = f"Error from tool '{tool_id}': {error_message}"
                status = f"error: tool execution failed: {error_message}"
                print(f"Extracted tool error: {raw_response_content}")
            elif "result" in tool_result_dict:
                raw_response_content = str(tool_result_dict["result"])
                status = None # Success (but message after tool was preferred)
                print(f"Extracted tool result (as fallback): {raw_response_content}")
            else:
                raw_response_content = str(tool_result_dict)
                status = "error: unexpected tool result structure"
                print(f"Warning: Tool return dict has unexpected structure: {tool_result_dict}")
        # If parsing failed, raw_response_content/status already set

    else: # Tool result not found at all
         print(f"Error: Tool return message for '{tool_id}' not found in chat history.")
         status = "error: tool result missing"
         # Use last message if available? Or keep default error.
         if history:
             raw_response_content = history[-1].get("content", raw_response_content)
             print(f"Using last message content as fallback: {raw_response_content}")
         else:
             raw_response_content = "Error: No tool result and history is empty."


    print(f"\nFinal Processed Status: {status or 'success'}") # Show 'success' if status is None
    print(f"Final Processed Content: {raw_response_content}")
    print("----------------------------------------------------")


# Run the async function
asyncio.run(run_two_agent_chat())
