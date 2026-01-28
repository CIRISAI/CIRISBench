import logging
# Assuming autogen is available and register_function works globally or is accessible
# If register_function needs to be applied to specific agents, this approach might need adjustment.
# from autogen import register_function

logger = logging.getLogger(__name__)

# --- Simple Calculator Tool ---

def simple_calculator(expression: str) -> dict:
    """
    Calculates the result of a simple mathematical expression.
    Currently supports only multiplication of two numbers.

    Args:
        expression: A string representing the mathematical expression (e.g., "5200 * 0.22").

    Returns:
        A dictionary containing the result or an error message.
    """
    logger.info(f"simple_calculator received expression: {expression}")
    try:
        # Basic parsing for "number * number" format
        if '*' in expression:
            parts = expression.split('*')
            if len(parts) == 2:
                num1 = float(parts[0].strip())
                num2 = float(parts[1].strip())
                result = num1 * num2
                logger.info(f"simple_calculator result: {result}")
                return {"result": str(result)} # Return result as string for consistency
            else:
                raise ValueError("Expression format not supported (expected 'number * number')")
        else:
             raise ValueError("Expression format not supported (only multiplication '*' is implemented)")

    except ValueError as ve:
        logger.error(f"Error in simple_calculator parsing expression '{expression}': {ve}")
        return {"error": f"Invalid expression format: {ve}"}
    except Exception as e:
        logger.error(f"Error calculating expression '{expression}': {e}", exc_info=True)
        return {"error": f"Calculation failed: {e}"}

# Define the schema for the simple_calculator tool
simple_calculator_schema = {
    "name": "simple_calculator",
    "description": "Calculates the result of a simple mathematical expression (currently only multiplication like 'X * Y').",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to calculate (e.g., '5200 * 0.22')."
            }
        },
        "required": ["expression"]
    }
}


# --- Tool Registry (for dynamic loading in handler) ---

# Registry mapping tool names to their function and schema
TOOL_REGISTRY = {
    "simple_calculator": {
        "function": simple_calculator,
        "schema": simple_calculator_schema
    }
}

# If register_function needs to be called explicitly later:
# We define the function here and it will be imported and registered elsewhere.
