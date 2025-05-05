"""
Custom LLM Implementation using LangChain
----------------------------------------
This script demonstrates how to create a custom LLM wrapper
using LangChain that connects to various LLM APIs.
"""

import os
from typing import Any, List, Mapping, Optional, Dict

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain.llms import OpenAI, Anthropic
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import HumanMessage


class CustomLLM:
    """Factory class to create appropriate LLM instances based on provider."""
    
    @staticmethod
    def create(provider: str, temperature: float = 0.7, **kwargs) -> LLM:
        """
        Create an LLM instance based on the specified provider.
        
        Args:
            provider: The LLM provider (openai, anthropic, etc.)
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            An LLM instance
        """
        provider = provider.lower()
        
        if provider == "openai":
            return OpenAILLM(temperature=temperature, **kwargs)
        elif provider == "anthropic":
            return AnthropicLLM(temperature=temperature, **kwargs)
        elif provider == "chat_openai":
            return ChatOpenAIWrapper(temperature=temperature, **kwargs)
        elif provider == "chat_anthropic":
            return ChatAnthropicWrapper(temperature=temperature, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


class OpenAILLM(LLM):
    """Custom LLM wrapper for OpenAI's completion models."""
    
    model_name: str = "text-davinci-003"  # Default model
    temperature: float = 0.7
    max_tokens: int = 256
    
    def __init__(self, temperature: float = 0.7, model_name: str = "text-davinci-003", 
                 max_tokens: int = 256, **kwargs):
        """Initialize the OpenAI LLM wrapper."""
        super().__init__()
        self.temperature = temperature
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = OpenAI(
            temperature=temperature,
            model_name=model_name,
            max_tokens=max_tokens,
            **kwargs
        )
        
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "openai"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Execute the LLM call."""
        return self.client(prompt, stop=stop)
        
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Return identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


class AnthropicLLM(LLM):
    """Custom LLM wrapper for Anthropic's Claude models."""
    
    model_name: str = "claude-3-opus-20240229"  # Default model
    temperature: float = 0.7
    max_tokens_to_sample: int = 256
    
    def __init__(self, temperature: float = 0.7, model_name: str = "claude-3-opus-20240229", 
                 max_tokens_to_sample: int = 256, **kwargs):
        """Initialize the Anthropic LLM wrapper."""
        super().__init__()
        self.temperature = temperature
        self.model_name = model_name
        self.max_tokens_to_sample = max_tokens_to_sample
        self.client = Anthropic(
            temperature=temperature,
            model=model_name,
            max_tokens_to_sample=max_tokens_to_sample,
            **kwargs
        )
        
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "anthropic"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Execute the LLM call."""
        return self.client(prompt, stop=stop)
        
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Return identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens_to_sample": self.max_tokens_to_sample
        }


class ChatOpenAIWrapper(LLM):
    """Wrapper for OpenAI's chat models."""
    
    model_name: str = "gpt-4-turbo"  # Default model
    temperature: float = 0.7
    max_tokens: int = 256
    
    def __init__(self, temperature: float = 0.7, model_name: str = "gpt-4-turbo", 
                 max_tokens: int = 256, **kwargs):
        """Initialize the Chat OpenAI wrapper."""
        super().__init__()
        self.temperature = temperature
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = ChatOpenAI(
            temperature=temperature,
            model_name=model_name,
            max_tokens=max_tokens,
            **kwargs
        )
        
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "chat_openai"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Execute the LLM call."""
        messages = [HumanMessage(content=prompt)]
        response = self.client.generate([messages])
        return response.generations[0][0].text
        
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Return identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


class ChatAnthropicWrapper(LLM):
    """Wrapper for Anthropic's chat models."""
    
    model_name: str = "claude-3-sonnet-20240229"  # Default model
    temperature: float = 0.7
    max_tokens: int = 1024
    
    def __init__(self, temperature: float = 0.7, model_name: str = "claude-3-sonnet-20240229", 
                 max_tokens: int = 1024, **kwargs):
        """Initialize the Chat Anthropic wrapper."""
        super().__init__()
        self.temperature = temperature
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = ChatAnthropic(
            temperature=temperature,
            model_name=model_name,
            max_tokens=max_tokens,
            **kwargs
        )
        
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "chat_anthropic"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Execute the LLM call."""
        messages = [HumanMessage(content=prompt)]
        response = self.client.generate([messages])
        return response.generations[0][0].text
        
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Return identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
