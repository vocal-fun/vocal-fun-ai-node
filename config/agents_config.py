import json
import random
import os
import aiohttp
import asyncio
from collections import OrderedDict
from typing import Dict, Optional, Tuple
import time

class AgentConfigManager:
    def __init__(self):
        self.voice_samples_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "voice_samples")
        self.configs_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "active_configs.json")
        self.voice_sample_access_times: OrderedDict[str, float] = OrderedDict()
        self.max_samples = 100
        os.makedirs(self.voice_samples_dir, exist_ok=True)
        self._load_configs()

    def _load_configs(self):
        """Load configs from file"""
        try:
            if os.path.exists(self.configs_file):
                with open(self.configs_file, 'r') as f:
                    self.agent_configs = json.load(f)
            else:
                self.agent_configs = {}
        except Exception as e:
            print(f"Error loading configs: {e}")
            self.agent_configs = {}

    def _save_configs(self):
        """Save configs to file"""
        try:
            with open(self.configs_file, 'w') as f:
                json.dump(self.agent_configs, f)
        except Exception as e:
            print(f"Error saving configs: {e}")

    async def download_voice_sample(self, url: str, config_id: str) -> Optional[str]:
        """Download voice sample and return local file path"""
        if not url:
            return None
            
        file_path = os.path.join(self.voice_samples_dir, f"{config_id}.wav")
        
        # Check if file already exists
        if os.path.exists(file_path):
            print(f"Voice sample already exists for config_id: {config_id}")
            return file_path
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        with open(file_path, 'wb') as f:
                            f.write(await response.read())
                        print(f"Downloaded new voice sample for config_id: {config_id}")
                        return file_path
        except Exception as e:
            print(f"Error downloading voice sample: {e}")
            return None

    def cleanup_old_samples(self):
        """Remove oldest voice samples if we exceed max_samples"""
        while len(self.voice_sample_access_times) > self.max_samples:
            oldest_file = next(iter(self.voice_sample_access_times))
            file_path = os.path.join(self.voice_samples_dir, oldest_file)
            try:
                os.remove(file_path)
                del self.voice_sample_access_times[oldest_file]
            except:
                pass

    async def add_agent_config(self, config: dict) -> None:
        """Add or update agent configuration"""
        try:
            config_id = config["configId"]
            print(f"Adding config for config_id: {config_id}")
            
            # Download voice sample if URL provided
            if config.get("voiceSampleUrl"):
                voice_sample_path = await self.download_voice_sample(
                    config["voiceSampleUrl"], 
                    config_id
                )
                if voice_sample_path:
                    self.voice_sample_access_times[f"{config_id}.wav"] = time.time()
                    self.cleanup_old_samples()
                    config["local_voice_sample"] = voice_sample_path

            # Load latest configs before updating
            self._load_configs()
            
            # Update config
            self.agent_configs[config_id] = config
            
            # Save to file
            self._save_configs()
            
            print(f"Config saved. Current configs: {self.agent_configs}")
        except Exception as e:
            print(f"Error adding agent config: {e}")
            raise

    def get_agent_config(self, config_id: str) -> Tuple[list, str, str, str, str]:
        """Returns voice samples, system prompt, language, and voice IDs for the given config"""
        # Load latest configs before reading
        self._load_configs()
        
        print(f"Getting config for config_id: {config_id}")
        print(f"Available configs: {self.agent_configs}")
        
        if config_id not in self.agent_configs:
            print(f"Config ID {config_id} not found in configs")
            return [], None, "en", None, None
            
        config = self.agent_configs[config_id]
        
        # Update access time if we have a voice sample
        if "local_voice_sample" in config:
            filename = os.path.basename(config["local_voice_sample"])
            self.voice_sample_access_times.move_to_end(filename)
            voice_samples = [config["local_voice_sample"]]
        else:
            voice_samples = []

        return (
            voice_samples,
            config.get("systemPrompt"),
            config.get("language", "en"),
            config.get("cartesiaVoiceId"),
            config.get("elevenLabsVoiceId")
        )
    
    def get_agent_name(self, config_id: str) -> str:
        """Returns the agent name for the given config"""
        if config_id not in self.agent_configs:
            return ""
        return self.agent_configs[config_id]["agentName"]

# Only keep the singleton instance
agent_manager = AgentConfigManager()

