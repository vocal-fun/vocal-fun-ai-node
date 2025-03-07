import json
import random
import os
import aiohttp
import asyncio
from collections import OrderedDict
from typing import Dict, Optional, Tuple
import time
from pydub import AudioSegment
import tempfile
from dotenv import load_dotenv
import threading

load_dotenv()

class AgentManager:
    def __init__(self):
        self.fast_mode = os.getenv("FAST_MODE", "False").lower() == "true"
        self.config_dir = "configs"
        self.configs: Dict[str, Tuple[str, str, str, str, str]] = {}  # config_id -> (voice_samples, system_prompt, language, cartesia_voice_id, elevenlabs_voice_id)
        self.lock = threading.Lock()
        
        if not self.fast_mode:
            # Only create directory if we're in normal mode
            os.makedirs(self.config_dir, exist_ok=True)

    async def add_agent_config(self, config: dict) -> None:
        """Add or update agent configuration"""
        config_id = config.get("configId")
        if not config_id:
            raise ValueError("Config ID is required")

        voice_samples = config.get("voiceSamples", "")
        system_prompt = config.get("systemPrompt", "")
        language = config.get("language", "en")
        cartesia_voice_id = config.get("cartesiaVoiceId", "")
        elevenlabs_voice_id = config.get("elevenlabsVoiceId", "")
        
        with self.lock:
            # Always update in-memory config
            self.configs[config_id] = (voice_samples, system_prompt, language, cartesia_voice_id, elevenlabs_voice_id)
            
            if not self.fast_mode:
                # In normal mode, also write to file
                config_path = os.path.join(self.config_dir, f"{config_id}.json")
                with open(config_path, 'w') as f:
                    json.dump({
                        "voiceSamples": voice_samples,
                        "systemPrompt": system_prompt,
                        "language": language,
                        "cartesiaVoiceId": cartesia_voice_id,
                        "elevenlabsVoiceId": elevenlabs_voice_id
                    }, f)

    def get_agent_config(self, config_id: str) -> Optional[Tuple[str, str, str, str, str]]:
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
                        config_tuple = (
                            config_data.get("voiceSamples", ""),
                            config_data.get("systemPrompt", ""),
                            config_data.get("language", "en"),
                            config_data.get("cartesiaVoiceId", ""),
                            config_data.get("elevenlabsVoiceId", "")
                        )
                        # Cache in memory
                        self.configs[config_id] = config_tuple
                        return config_tuple
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