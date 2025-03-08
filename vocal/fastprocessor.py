import wave
import numpy as np
import json
import os
from datetime import datetime
from typing import Optional, List
import asyncio
from fastapi import WebSocket
from vocal.utils.speechdetector import AudioSpeechDetector
from vocal.config.agents_config import agent_manager
import time
import io

class FastProcessor:
    def __init__(self, session_id: str, config_id: str):
        self.session_id = session_id
        self.config_id = config_id
        self.audio_chunks: List[np.ndarray] = []
        self.is_speaking = False
        self.is_responding = False
        self.tts_lock = asyncio.Lock()
        
        # Lazy import services only when FastProcessor is actually used
        from vocal.stt.stt import stt_instance
        from vocal.chat.chat import chat_instance
        from vocal.tts.tts import tts_instance
        
        # Initialize services directly
        self.stt_service = stt_instance
        self.chat_service = chat_instance
        self.tts_service = tts_instance

        self.config = agent_manager.get_agent_config(config_id)
        self.voice_samples = self.config.voice_samples
        self.language = self.config.language
        self.system_prompt = self.config.system_prompt
        
        self.speech_detector = AudioSpeechDetector(
            sample_rate=16000,
            energy_threshold=0.15,
            min_speech_duration=0.4,
            max_silence_duration=0.5,
            max_recording_duration=10.0,
            debug=False
        )

    async def process_audio_chunk(self, binary_data: bytes, websocket: WebSocket) -> None:
        """Process incoming audio chunks and handle VAD internally"""
        if len(binary_data) > 0:
            audio_data = np.frombuffer(binary_data, dtype=np.int16)
            detection_result = self.speech_detector.add_audio_chunk(audio_data)
            
            if detection_result['action'] == 'process':
                self.audio_chunks = detection_result.get('audio_chunks', [])
                await self.process_speech(websocket)

    async def process_speech(self, websocket: WebSocket) -> None:
        """Handle speech processing and response generation"""
        print("Processing speech...")
        self.is_responding = True
        try:
            # Convert audio chunks list to a single numpy array
            if self.audio_chunks:
                combined_audio = np.concatenate(self.audio_chunks)
                self.audio_chunks = []
                
                # Pass the numpy array directly to STT service
                transcript = await self.stt_service.transcribe(combined_audio, self.language)

                if not transcript.strip() or "thank you" in transcript.lower():
                    self.is_responding = False
                    return
                    
                await self.process_text(transcript, websocket)
            else:
                self.is_responding = False

        except Exception as e:
            self.is_responding = False
            print(f"Error processing audio: {e}")
            await self.send_error(websocket, str(e))

    async def process_text(self, text: str, websocket: WebSocket) -> None:
        """Process text input and generate response"""
        print(f"Processing text: {text}")
        async with self.tts_lock:
            self.is_responding = True
            try:
                data = {    
                    "text": text,
                    "session_id": self.session_id,
                    "config_id": self.config_id
                }
                chat_response = await self.chat_service.generate_response(
                    data
                )
                await self.stream_tts(chat_response, websocket)
            except Exception as e:
                self.is_responding = False
                print(f"Error processing text: {e}")
                await self.send_error(websocket, str(e))

    async def stream_tts(self, text: str, websocket: WebSocket) -> None:
        """Stream TTS audio directly"""
        try:
            voice_id = self.tts_service.get_voice_id(self.config_id)
            async for chunk in await self.tts_service.generate_speech_stream(text, self.language, voice_id, self.voice_samples):
                data = {
                    "type": "audio_chunk",
                    "chunk": chunk.chunk,
                    "format": chunk.format,
                    "sample_rate": chunk.sample_rate,
                    "timestamp": time.time()
                }
                await websocket.send_json({
                    "type": "tts_stream",
                    "data": data,
                    "session_id": self.session_id
                })
                
            await websocket.send_json({
                    "type": "tts_stream_end",
                    "session_id": self.session_id
                })
        except Exception as e:
            print(f"Error in TTS streaming: {e}")
            await self.send_error(websocket, str(e))

    async def handle_client_message(self, message_type: str, websocket: WebSocket) -> None:
        """Handle client control messages"""
        if message_type == "speech_start":
            self.is_speaking = True
            self.audio_chunks = []
        elif message_type == "speech_end":
            self.is_speaking = False
            if self.audio_chunks:
                await self.process_speech(websocket)

    async def send_error(self, websocket: WebSocket, error: str) -> None:
        """Send error message to client"""
        try:
            await websocket.send_json({
                "type": "error",
                "error": error,
                "session_id": self.session_id
            })
        except RuntimeError:
            pass  # WebSocket is closed

    async def cleanup(self):
        """Clean up any resources"""
        pass  # No cleanup needed for direct calls 