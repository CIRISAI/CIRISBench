from typing import Dict, List, Any, Optional
import os
import traceback

class LLMSetter:
    """Class for configuring LLM settings."""

    def __init__(self):
        # Check environment for LLM provider configuration
        llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        llm_model = os.getenv("LLM_MODEL", "gemma3:4b-it-q8_0")

        if llm_provider == "openrouter":
            # OpenRouter uses OpenAI-compatible API
            self.api_type = 'openai'
            self.model = llm_model
            self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self.api_key = os.getenv("OPENROUTER_API_KEY", "None")
            self.openai_llm = True
            self.ollama_llm = False
        elif llm_provider == "openai":
            self.api_type = 'openai'
            self.model = llm_model
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.api_key = os.getenv("OPENAI_API_KEY", "None")
            self.openai_llm = True
            self.ollama_llm = False
        elif llm_provider == "anthropic":
            # Anthropic uses a different API but some wrappers exist
            self.api_type = 'openai'  # Use OpenAI-compatible wrapper if available
            self.model = llm_model
            self.base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
            self.api_key = os.getenv("ANTHROPIC_API_KEY", "None")
            self.openai_llm = True
            self.ollama_llm = False
        elif llm_provider == "together":
            # Together.ai uses OpenAI-compatible API
            self.api_type = 'openai'
            self.model = llm_model
            self.base_url = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
            self.api_key = os.getenv("TOGETHER_API_KEY", "None")
            self.openai_llm = True
            self.ollama_llm = False
        else:
            # Default to Ollama
            self.api_type = 'ollama'
            self.model = llm_model
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434") + "/v1"
            self.api_key = os.getenv("OPENAI_API_KEY", "None")
            self.openai_llm = False
            self.ollama_llm = True

        self.default_llm_config_list = self._create_default_config()
    
    def _create_default_config(self) -> List[Dict[str, Any]]:
        """Create a default configuration list."""
        return [{
            "api_type": self.api_type,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": self.api_key
        }]

    def config_llm(self, llm_dict: Dict[str, Any]) -> None:
        """Configure LLM settings from a dictionary.
        
        Args:
            llm_dict: A dictionary containing LLM configuration parameters.
        """
        try:
            self.api_key = llm_dict.get('api_key', 'None')
            self.model = llm_dict.get('model', 'gemma3:4b-it-q8_0')
            self.base_url = llm_dict.get('base_url', 'http://127.0.0.1:11434/v1')
            self.api_type = llm_dict.get('api_type', 'ollama')
            
            # Update the flags based on api_type
            if self.api_type == 'openai':
                self.openai_llm = True
                self.ollama_llm = False
            elif self.api_type == 'ollama':
                self.ollama_llm = True
                self.openai_llm = False
            else:
                raise ValueError(f"Invalid API type: {self.api_type} - must be 'ollama' or 'openai'")
                
            # Update the default config list
            self.default_llm_config_list = self._create_default_config()
            return self.default_llm_config_list
            
        except Exception as e:
            traceback.print_exc()
            return None
            
    def get_config(self) -> List[Dict[str, Any]]:
        """Get the current LLM configuration.
        
        Returns:
            The current LLM configuration list.
        """
        return self._create_default_config()