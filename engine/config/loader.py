import os
import ujson
import yaml
import logging
from typing import Dict, List, Type, TypeVar, Optional
from pydantic import ValidationError

from config.settings import settings
from schemas.identity import Identity
from schemas.ethical_guidance import EthicalGuidance
from schemas.guardrail import Guardrail
from schemas.pipeline import Pipeline

logger = logging.getLogger(__name__)

T = TypeVar('T') # Generic type variable for Pydantic models

def _load_config_file(file_path: str) -> Optional[Dict | List]:
    """Loads a single JSON or YAML configuration file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                return ujson.load(f)
            elif file_path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(f)
            else:
                logger.warning(f"Unsupported file type skipped: {file_path}")
                return None
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading configuration file {file_path}: {e}", exc_info=True)
        return None

def _validate_and_parse(data: Dict | List, model: Type[T], file_path: str) -> List[T]:
    """Validates raw data against a Pydantic model."""
    validated_items = []
    if isinstance(data, list): # Expecting a list of items
        for i, item_data in enumerate(data):
            if not isinstance(item_data, dict):
                logger.warning(f"Skipping invalid item (not a dict) at index {i} in {file_path}")
                continue
            try:
                validated_item = model.model_validate(item_data)
                # Ensure 'id' field exists for dictionary mapping later
                if hasattr(validated_item, 'id') and getattr(validated_item, 'id'):
                    validated_items.append(validated_item)
                else:
                     logger.warning(f"Skipping item without 'id' at index {i} in {file_path}: {item_data}")
            except ValidationError as e:
                logger.warning(f"Validation failed for item at index {i} in {file_path}: {e}\nData: {item_data}")
            except Exception as e:
                 logger.error(f"Unexpected error parsing item at index {i} in {file_path}: {e}\nData: {item_data}", exc_info=True)

    elif isinstance(data, dict): # Expecting a single item or a dict of items
        # --- Start: Prioritize validating the entire dict as one item ---
        try:
            # Attempt to validate the entire dictionary as a single model instance
            validated_item = model.model_validate(data)
            # Ensure 'id' field exists for dictionary mapping later
            # For a single pipeline file, the 'id' should be defined within the file content.
            # If the model requires an 'id' and it's missing in the data, validation will fail here.
            if hasattr(validated_item, 'id') and getattr(validated_item, 'id'):
                 validated_items.append(validated_item)
                 # If successful, we assume the file contained a single item and stop.
                 return validated_items
            else:
                 # This case might occur if the model doesn't strictly require 'id',
                 # but our loading logic expects it. Or if the 'id' is empty/None.
                 logger.warning(f"Validated item from {file_path} lacks a valid 'id'. Data: {data}")
                 # Decide if we should still return it or treat it as an error/skip.
                 # For now, let's skip it if ID is crucial for the dictionary structure.
                 # If the file *was* intended to be a single item, this indicates a problem.
                 # Return empty list as validation didn't produce a usable item with an ID.
                 return []

        except ValidationError as e_single:
            # If validating the whole dict as a single item failed, log the specific error and skip this file.
            # Do NOT attempt to parse as a dict of items {key: item_data}, as that's the wrong assumption for single-item files.
            logger.warning(f"Validation failed for single item in {file_path}: {e_single}\nData: {data}")
            return [] # Return empty list, indicating no valid items were parsed from this file.
        except Exception as e: # Catch any other unexpected errors during single item validation
            logger.error(f"Unexpected error parsing single item dict in {file_path}: {e}\nData: {data}", exc_info=True)
            return [] # Return empty list on unexpected errors
    else:
        logger.warning(f"Unsupported data structure in {file_path}. Expected list or dict.")

    return validated_items


def load_configs_from_directory(directory: str, model: Type[T], file_pattern: str = "*") -> Dict[str, T]:
    """Loads all valid config files matching a pattern from a directory."""
    loaded_configs: Dict[str, T] = {}
    if not os.path.isdir(directory):
        logger.warning(f"Configuration directory not found: {directory}")
        return loaded_configs

    for filename in os.listdir(directory):
        if not (filename.endswith('.json') or filename.endswith('.yaml') or filename.endswith('.yml')):
            continue
        # TODO: Add file_pattern matching if needed

        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            raw_data = _load_config_file(file_path)
            if raw_data:
                validated_items = _validate_and_parse(raw_data, model, file_path)
                for item in validated_items:
                    if item.id in loaded_configs:
                        logger.warning(f"Duplicate ID '{item.id}' found. Overwriting previous definition from {filename}.")
                    loaded_configs[item.id] = item
    logger.info(f"Loaded {len(loaded_configs)} {model.__name__} configurations from {directory}.")
    return loaded_configs

# --- Specific Loaders ---

def load_all_identities(data_dir: str = settings.data_dir) -> Dict[str, Identity]:
    """Loads all Identity configurations."""
    identity_dir = os.path.join(data_dir, "identities")
    return load_configs_from_directory(identity_dir, Identity)

def load_all_guidances(data_dir: str = settings.data_dir) -> Dict[str, EthicalGuidance]:
    """Loads all EthicalGuidance configurations."""
    guidance_dir = os.path.join(data_dir, "guidances")
    return load_configs_from_directory(guidance_dir, EthicalGuidance)

def load_all_guardrails(data_dir: str = settings.data_dir) -> Dict[str, Guardrail]:
    """Loads all Guardrail configurations."""
    guardrail_dir = os.path.join(data_dir, "guardrails")
    return load_configs_from_directory(guardrail_dir, Guardrail)

def load_all_pipelines(data_dir: str = settings.data_dir) -> Dict[str, Pipeline]:
    """Loads all Pipeline configurations."""
    pipeline_dir = os.path.join(data_dir, "pipelines")
    return load_configs_from_directory(pipeline_dir, Pipeline)

# Example usage:
# if __name__ == "__main__":
#     from utils.logging_config import setup_logging
#     setup_logging()
#     identities = load_all_identities()
#     print(f"Loaded identities: {list(identities.keys())}")
#     pipelines = load_all_pipelines()
#     print(f"Loaded pipelines: {list(pipelines.keys())}")
