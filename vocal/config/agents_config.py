import json
import random
import os
import aiohttp
import asyncio
from collections import OrderedDict
from typing import Dict, Optional
import time
from pydub import AudioSegment
import tempfile
from dotenv import load_dotenv
import threading
from dataclasses import dataclass

load_dotenv()

@dataclass
class AgentConfig:
    voice_samples: str
    system_prompt: str
    language: str
    cartesia_voice_id: str
    elevenlabs_voice_id: str
    
    def to_dict(self) -> dict:
        return {
            "voiceSamples": self.voice_samples,
            "systemPrompt": self.system_prompt,
            "language": self.language,
            "cartesiaVoiceId": self.cartesia_voice_id,
            "elevenlabsVoiceId": self.elevenlabs_voice_id
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AgentConfig':
        return cls(
            voice_samples=data.get("voiceSamples", ""),
            system_prompt=data.get("systemPrompt", ""),
            language=data.get("language", "en"),
            cartesia_voice_id=data.get("cartesiaVoiceId", ""),
            elevenlabs_voice_id=data.get("elevenlabsVoiceId", "")
        )

class AgentManager:
    def __init__(self):
        self.fast_mode = os.getenv("FAST_MODE", "False").lower() == "true"
        self.config_dir = "configs"
        self.configs: Dict[str, AgentConfig] = {}
        self.lock = threading.Lock()
        
        if not self.fast_mode:
            # Only create directory if we're in normal mode
            os.makedirs(self.config_dir, exist_ok=True)

    async def add_agent_config(self, config: dict) -> None:
        """Add or update agent configuration"""
        config_id = config.get("configId")
        if not config_id:
            raise ValueError("Config ID is required")

        agent_config = AgentConfig(
            voice_samples=config.get("voiceSampleUrl", ""),
            system_prompt=config.get("systemPrompt", ""),
            language=config.get("language", "en"),
            cartesia_voice_id=config.get("cartesiaVoiceId", ""),
            elevenlabs_voice_id=config.get("elevenlabsVoiceId", "")
        )
        
        with self.lock:
            self.configs[config_id] = agent_config
            
            if not self.fast_mode:
                # In normal mode, also write to file
                config_path = os.path.join(self.config_dir, f"{config_id}.json")
                with open(config_path, 'w') as f:
                    json.dump(agent_config.to_dict(), f)

    def get_agent_config(self, config_id: str) -> Optional[AgentConfig]:
        """Get agent configuration"""
        with self.lock:
            if self.fast_mode:
                # In fast mode, return directly from memory
                return self.configs.get(config_id)
            
            # In normal mode, try memory first, then file
            if config_id in self.configs:
                return self.configs[config_id]
            
            # Try loading from file
            config_path = os.path.join(self.config_dir, f"{config_id}.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                        agent_config = AgentConfig.from_dict(config_data)
                        # Cache in memory
                        self.configs[config_id] = agent_config
                        return agent_config
                except Exception as e:
                    print(f"Error loading config {config_id}: {e}")
                    return None
            
            return None

    def remove_agent_config(self, config_id: str) -> bool:
        """Remove agent configuration"""
        with self.lock:
            # Always remove from memory
            self.configs.pop(config_id, None)
            
            if not self.fast_mode:
                # In normal mode, also remove file
                config_path = os.path.join(self.config_dir, f"{config_id}.json")
                try:
                    if os.path.exists(config_path):
                        os.remove(config_path)
                except Exception as e:
                    print(f"Error removing config file {config_id}: {e}")
                    return False
            return True

    def clear_configs(self) -> None:
        """Clear all configurations"""
        with self.lock:
            self.configs.clear()
            
            if not self.fast_mode:
                # In normal mode, also clear files
                for filename in os.listdir(self.config_dir):
                    if filename.endswith('.json'):
                        try:
                            os.remove(os.path.join(self.config_dir, filename))
                        except Exception as e:
                            print(f"Error removing config file {filename}: {e}")


agent_manager = AgentManager()