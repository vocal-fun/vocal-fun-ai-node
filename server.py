from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
import json
import asyncio
import wave
import numpy as np
from typing import Dict, Optional
import os
from datetime import datetime
from speechdetector import AudioSpeechDetector

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs
STT_SERVICE_URL = "http://localhost:8001/transcribe"
CHAT_SERVICE_URL = "http://localhost:8002/chat/groq"
TTS_SERVICE_URL = "ws://localhost:8003/tts_stream"

class AudioSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.audio_chunks = []
        self.is_speaking = False
        self.is_responding = False
        self.agent_id = ""
        self.agent_name = ""
        self.websocket: Optional[WebSocket] = None
        self.tts_websocket: Optional[aiohttp.ClientWebSocketResponse] = None

    async def save_audio(self) -> str:
        if not self.audio_chunks:
            return ""
            
        os.makedirs("audio_files", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_files/{self.session_id}_{timestamp}.wav"
        
        audio_data = np.concatenate(self.audio_chunks)
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_data.tobytes())
        
        self.audio_chunks = []
        return filename

class ConnectionManager:
    def __init__(self):
        self.active_sessions: Dict[str, AudioSession] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = AudioSession(session_id)
        self.active_sessions[session_id].websocket = websocket
        return self.active_sessions[session_id]

    async def disconnect(self, session_id: str):
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            if session.tts_websocket:
                await session.tts_websocket.close()
            del self.active_sessions[session_id]

manager = ConnectionManager()

async def process_text_to_response(session: AudioSession, text: str) -> None:
    print(f"Processing text: {text}")
    session.is_responding = True
    try:
        async with aiohttp.ClientSession() as http_session:
            # Get chat response
            async with http_session.post(
                CHAT_SERVICE_URL,
                json={"text": text, "session_id": session.session_id, "personality": session.agent_name}
            ) as response:
                chat_result = await response.json()
                chat_response = chat_result['response']

            #discard existing audio if new text is received
            if session.tts_websocket:
                await session.tts_websocket.close()
                session.tts_websocket = None

            # Stream TTS response
            async with http_session.ws_connect(TTS_SERVICE_URL) as ws:
                session.tts_websocket = ws
                await ws.send_json({
                    "text": chat_response,
                    "personality": session.agent_name
                })

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        await session.websocket.send_json({
                            "type": "tts_stream",
                            "data": data
                        })
                        
                        if data.get("type") == "stream_end":
                            session.is_responding = False
                            await session.websocket.send_json({
                                "type": "tts_stream_end"
                            })
                            break
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

    except Exception as e:
        session.is_responding = False
        print(f"Error processing text: {e}")
        if session.websocket:
            await session.websocket.send_json({
                "type": "error",
                "error": str(e)
            })

async def process_audio_to_response(session: AudioSession) -> None:
    print("Processing audio...")
    session.is_responding = True
    try:
        audio_file = await session.save_audio()
        if not audio_file:
            return

        async with aiohttp.ClientSession() as http_session:
            # Get transcription
            files = {'audio_file': open(audio_file, 'rb')}
            async with http_session.post(STT_SERVICE_URL, data=files) as response:
                transcript_result = await response.json()
                transcript = transcript_result['text']

            #check if all whitespace or empty
            if not transcript.strip():
                session.is_responding = False
                return

            if len(transcript) < 10:
                session.is_responding = False
                return
            
            # Process the transcript through chat and TTS
            await process_text_to_response(session, transcript)

    except Exception as e:
        session.is_responding = False
        print(f"Error processing audio: {e}")
        if session.websocket:
            await session.websocket.send_json({
                "type": "error",
                "error": str(e)
            })
    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    session = await manager.connect(websocket, session_id)
    print(f"Client connected: {session_id}")

     # Wait for the first message (which should contain agentId and userId)
    data = await websocket.receive_text()
    message = json.loads(data)

    # Extract agentId and userId from the message
    agent_id = message.get("agentId")
    user_id = message.get("userId")
    agent_name = message.get("agentName")

    session.agent_id = agent_id
    session.agent_name = agent_name

    speech_detector = AudioSpeechDetector(
        sample_rate=16000,
        energy_threshold=0.1,  # Adjust based on your audio environment
        min_speech_duration=0.4,  # Minimum speech to process
        max_silence_duration=0.4,  # Pause length to trigger processing
        max_recording_duration=10.0,
        debug=False  # Enable detailed logging
    )

    print(f"Received agentId: {agent_id}, userId: {user_id}")
    
    try:
        while True:
            message = await websocket.receive()
            # print(f"Received message type: {message['type']}")
            
            if message["type"] == "websocket.disconnect":
                break
                
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    try:
                        if session.is_responding:
                            # print("Ignoring audio data while responding")
                            continue

                        binary_data = message["bytes"]
                        if len(binary_data) > 0:
                           # Convert binary data to numpy array
                            audio_data = np.frombuffer(binary_data, dtype=np.int16)
                            
                            # Detect speech
                            detection_result = speech_detector.add_audio_chunk(audio_data)
                            
                            if detection_result['action'] == 'process':
                                # If speech detected and processed
                                processed_chunks = detection_result.get('audio_chunks', [])
                                
                                # Temporarily store processed chunks in session
                                session.audio_chunks = processed_chunks
                                
                                print(f"Speech ended, processing response: {len(audio_data)} samples")
                                # Process the audio
                                await process_audio_to_response(session)
                            
                            # print(f"Received audio chunk: {len(audio_data)} samples")
                    except Exception as e:
                        print(f"Error processing audio data: {e}")
                
                elif "text" in message:
                    try:
                        data = json.loads(message["text"])
                        print(f"Received text message: {data}")
                        
                        if data["type"] == "speech_start":
                            session.is_speaking = True
                            session.audio_chunks = []
                            
                        elif data["type"] == "speech_end":
                            session.is_speaking = False
                            if session.audio_chunks:  # Only process if we have audio data
                                await process_audio_to_response(session)
                            else:
                                print("No audio chunks to process")

                        elif data["type"] == "transcript":
                            await process_text_to_response(session, data["text"])
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
        await manager.disconnect(session_id)
    except Exception as e:
        print(f"Error in websocket connection: {e}")
        await manager.disconnect(session_id)
    finally:
        print(f"Cleaning up session: {session_id}")
        await manager.disconnect(session_id)