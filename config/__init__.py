import asyncio
import ujson
import logging
import os
from typing import Optional
from open_llm.config_llm import LLMSetter
# Attempt to import LLMConfig, handle gracefully if autogen is not installed
try:
    from autogen import LLMConfig
    AUTOGEN_AVAILABLE = True
except ImportError:
    LLMConfig = None # Define as None if import fails
    AUTOGEN_AVAILABLE = False

from .settings import settings

logger = logging.getLogger(__name__)

# --- Initialize Semaphore ---
semaphore = asyncio.Semaphore(settings.max_concurrent_llm_calls)
logger.info(f"Initialized asyncio.Semaphore with limit: {settings.max_concurrent_llm_calls}")

# Update the LLM initialization section in __init__.py

# --- Initialize LLMSetter ---
llm_setter = LLMSetter()

# --- Initialize LLMConfig ---
llm_config: Optional['LLMConfig'] = None  # Initialize as None
_config_list_source = "none"  # Track where the config came from

if AUTOGEN_AVAILABLE and LLMConfig:
    loaded_config_list = None
    # 1. Try loading from the specific environment variable first (as an override)
    config_list_env_value = os.getenv(settings.config_list_env_var)
    if config_list_env_value:
        logger.info(f"Attempting to load LLM config from environment variable: {settings.config_list_env_var}")
        try:
            # Use LLMConfig.from_json which handles file paths or JSON strings
            _llm_config_loader = LLMConfig.from_json(env=settings.config_list_env_var)
            loaded_config_list = _llm_config_loader.config_list
            if loaded_config_list:
                _config_list_source = f"env_var ({settings.config_list_env_var})"
                logger.info(f"Successfully loaded LLM config list from '{settings.config_list_env_var}'.")
            else:
                logger.warning(f"Env var '{settings.config_list_env_var}' set, but failed to load a valid config list from it.")
        except FileNotFoundError:
            logger.error(f"Env var '{settings.config_list_env_var}' points to a file path, but the file was not found.")
            # Continue to fallback
        except ujson.JSONDecodeError:
            logger.error(f"Failed to decode JSON from env var '{settings.config_list_env_var}'. Ensure it's valid JSON or a path to a valid JSON file.")
            # Continue to fallback
        except Exception as e:
            logger.error(f"Unexpected error loading LLM config from '{settings.config_list_env_var}': {e}", exc_info=True)
            # Continue to fallback

    # 2. If not loaded from env var, use the default config list
    if not loaded_config_list:
        logger.info("LLM config not loaded from environment variable, using default configuration.")
        default_list = settings.default_llm_config_list
        
        # Use the LLMSetter to configure the default list
        if default_list:
            try:
                # Configure each item in the default list
                for item in default_list:
                    llm_setter.config_llm(item)
                
                # Get the updated configuration
                loaded_config_list = llm_setter.get_config()
                if loaded_config_list:
                    _config_list_source = "default_settings"
                    logger.info("Using default LLM config list from settings.")
                else:
                    logger.error("Failed to create a valid config list from default settings.")
            except Exception as e:
                logger.error(f"Error configuring LLM from default list: {e}", exc_info=True)
        else:
            logger.warning("No default LLM config list defined in settings.")

    # 3. If a valid config list was loaded, create the LLMConfig object
    if loaded_config_list:
        try:
            # Pass only config_list to LLMConfig
            llm_config = LLMConfig(config_list=loaded_config_list)
            logger.info(f"Initialized global LLMConfig (Source: {_config_list_source}) with config_list.")
        except Exception as e:
            logger.error(f"Error creating LLMConfig object from loaded list (Source: {_config_list_source}): {e}", exc_info=True)
            llm_config = None  # Ensure it's None if instantiation fails
    else:
        logger.error("Failed to load any valid LLM configuration. LLM functionality will be disabled.")

else:
    if not AUTOGEN_AVAILABLE:
        logger.error("Autogen library not found. LLM functionality will be disabled.")
    # If LLMConfig is None for some other reason
    elif not LLMConfig:
        logger.error("LLMConfig class could not be resolved even though Autogen seems available. LLM functionality disabled.")

# --- Initialize Concurrency Monitor ---
# Import late to avoid circular dependency if monitor imports config items
from utils.concurrency_monitor import ConcurrencyMonitor
# Pass the limit from settings during instantiation
concurrency_monitor = ConcurrencyMonitor(
    semaphore=semaphore,
    limit=settings.max_concurrent_llm_calls,
    name="LLM_Semaphore"
)


# --- Expose key configurations for easy import ---
# Make settings, semaphore, llm_config, and constants easily importable from 'config'
AGENT_TIMEOUT = settings.agent_timeout
AG2_REASONING_SPECS = settings.ag2_reasoning_specs

__all__ = [
    "settings",
    "semaphore",
    "llm_config",
    "concurrency_monitor", # Export the monitor instance
    "AGENT_TIMEOUT",
    "AG2_REASONING_SPECS",
    "AUTOGEN_AVAILABLE" # Expose availability flag
]
