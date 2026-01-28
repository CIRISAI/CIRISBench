from typing import Dict, List, Any, Optional
import os
import traceback

class LLMSetter:
    """Class for configuring LLM settings."""
    
    def __init__(self):
        self.api_type: str = 'ollama'
        self.model: str = 'gemma3:4b-it-q8_0'
        self.base_url: str = 'http://127.0.0.1:11434/v1'
        self.api_key: str = os.getenv("OPENAI_API_KEY", "None")
        self.openai_llm: bool = False
        self.ollama_llm: bool = True
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