from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class BaseLLM(ABC):
    def __init__(self):
        self.is_setup = False

    @abstractmethod
    def setup(self):
        """Initialize any necessary resources for the LLM"""
        pass

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response for the given prompt"""
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup any resources"""
        pass 