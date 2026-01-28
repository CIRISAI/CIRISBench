import re
import logging
from typing import Dict, Any, Union, List

logger = logging.getLogger(__name__)

# Define a type alias for clarity
Resolvable = Union[str, Dict[str, Any], List[Any]]

def resolve_placeholders(template: Resolvable, context: Dict[str, Any]) -> Resolvable:
    """
    Recursively resolves placeholders in strings, dictionaries, or lists using the pipeline context.
    Placeholders can be:
    - {stage_id.output_label}
    - {{ direct_context_key }}

    Args:
        template: The string, dictionary, or list containing placeholders.
        context: The pipeline context dictionary, structured as
                 {stage_id: {output_label: value, ...}, direct_key: value, ...}.

    Returns:
        The template with placeholders resolved. Unresolved placeholders
        are currently left as is, with a warning logged. Types are preserved.
    """
    if isinstance(template, dict):
        resolved_dict = {}
        for key, value in template.items():
            # Recursively resolve placeholders in values
            resolved_dict[key] = resolve_placeholders(value, context)
        return resolved_dict
    elif isinstance(template, list):
        # Recursively resolve placeholders in list items
        return [resolve_placeholders(item, context) for item in template]
    elif not isinstance(template, str):
        # If it's not a dict, list, or string, return it as is
        logger.debug(f"resolve_placeholders encountered non-resolvable type {type(template)}, returning as is.")
        return template

    # --- Handle String Resolution ---
    resolved_string = template # Start with the original string

    # 1. Resolve {stage_id.output_label} format
    stage_placeholders = re.finditer(r"\{([\w-]+)\.([\w-]+)\}", resolved_string)
    for match in stage_placeholders:
        full_placeholder = match.group(0)
        stage_id = match.group(1)
        output_label = match.group(2)

        # Look up the value in the context
        stage_output = context.get(stage_id)
        if stage_output is not None and isinstance(stage_output, dict):
            value = stage_output.get(output_label)
            if value is not None:
                # Replace the placeholder with the string representation of the value
                resolved_string = resolved_string.replace(full_placeholder, str(value))
                logger.debug(f"Resolved placeholder {full_placeholder} to value: {str(value)[:100]}...") # Log truncated value
            else:
                logger.warning(f"Could not resolve placeholder {full_placeholder}: Label '{output_label}' not found in output of stage '{stage_id}'.")
                # Option: Replace with empty string? resolved_string = resolved_string.replace(full_placeholder, "")
        else:
            logger.warning(f"Could not resolve placeholder {full_placeholder}: Stage ID '{stage_id}' not found in context or its output is not a dictionary.")
                # Option: Replace with empty string? resolved_string = resolved_string.replace(full_placeholder, "")

    # 2. Resolve {{ direct_context_key }} format
    # Corrected regex to match spaces inside braces and non-greedy key capture
    direct_placeholders = re.finditer(r"\{\{\s*([\w\s.-]+?)\s*\}\}", resolved_string) # Allow dots in keys
    placeholders_to_replace = {} # Store replacements to avoid issues with overlapping matches during iteration

    for match in direct_placeholders:
        full_placeholder = match.group(0)
        key = match.group(1).strip() # Get the key and strip whitespace

        # Look up the key directly in the context
        # Handle potential nested keys like 'stage_id.output_label' if used directly
        value = context
        try:
            for part in key.split('.'):
                if isinstance(value, dict):
                    value = value.get(part)
                elif isinstance(value, list) and part.isdigit():
                     value = value[int(part)] # Basic list index access
                else:
                    value = None # Key part not found or structure mismatch
                    break
        except (KeyError, IndexError, TypeError, ValueError):
             value = None # Handle potential errors during lookup

        if value is not None:
            # Store placeholder and its value for replacement after iteration
            placeholders_to_replace[full_placeholder] = str(value)
            logger.debug(f"Resolved direct placeholder {full_placeholder} for key '{key}'. Value: {str(value)[:100]}...")
        else:
            logger.warning(f"Could not resolve direct placeholder {full_placeholder}: Key '{key}' not found directly in context or structure mismatch.")
            # Option: Leave unresolved or replace with empty string? placeholders_to_replace[full_placeholder] = ""

    # Perform replacements after finding all matches
    for placeholder, value in placeholders_to_replace.items():
        resolved_string = resolved_string.replace(placeholder, value)

    return resolved_string

# Example Usage:
# if __name__ == "__main__":
#     ctx = {
#         "plan_phase": {"plan": "Step 1: Do X. Step 2: Do Y."},
#         "decision_phase": {"decision_text": "Final decision is Z."}
#     }
#     template1 = "Based on the plan {plan_phase.plan}, the decision is {decision_phase.decision_text}."
#     template2 = "Input was {user_input.query}." # This will fail to resolve
#
#     print(resolve_placeholders(template1, ctx))
#     print(resolve_placeholders(template2, ctx))
