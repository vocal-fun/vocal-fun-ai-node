from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Query

import json
from typing import Dict, Optional, Union
from vocal.config.agents_config import agent_manager
from dotenv import load_dotenv
import os

load_dotenv()

fast_mode = os.getenv("FAST_MODE", "False").lower() == "true"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.agent_id = ""
        self.config_id = ""
        self.websocket: Optional[WebSocket] = None
        self.processor = None

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
            if session.processor:
                await session.processor.cleanup()
            del self.active_sessions[session_id]

manager = ConnectionManager()

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    session = await manager.connect(websocket, session_id)
    print(f"Client connected: {session_id}")

    try:
        # Handle initial configuration
        data = await websocket.receive_text()
        config = json.loads(data)

        if not isinstance(config, dict):
            print("Error: Config is not a dictionary")
            return
            
        required_fields = ["configId", "agentId", "agentName"]
        if missing_fields := [field for field in required_fields if field not in config]:
            print(f"Error: Missing required fields in config: {missing_fields}")
            return

        await agent_manager.add_agent_config(config)
        
        session.agent_id = config["agentId"]
        session.config_id = config["configId"]
        
        # Create appropriate processor based on mode
        from vocal.processor import AudioProcessor
        from vocal.fastprocessor import FastProcessor
        ProcessorClass = FastProcessor if fast_mode else AudioProcessor
        session.processor = ProcessorClass(session_id, config["configId"])

        await websocket.send_json({
            "type": "call_ready",
            "session_id": session_id
        })

        print(f"Received agentId: {session.agent_id}, userId: {session.session_id}")
        
        # Main message handling loop
        while True:
            message = await websocket.receive()
            
            if message["type"] == "websocket.disconnect":
                break
                
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    try:
                        await session.processor.process_audio_chunk(message["bytes"], websocket)
                    except Exception as e:
                        print(f"Error processing audio data: {e}")
                
                elif "text" in message:
                    try:
                        data = json.loads(message["text"])
                        if data["type"] in ["speech_start", "speech_end"]:
                            await session.processor.handle_client_message(data["type"], websocket)
                        elif data["type"] == "transcript":
                            await session.processor.process_text(data["text"], websocket)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        print(f"Error in websocket connection: {e}")
    finally:
        print(f"Cleaning up session: {session_id}")
        await manager.disconnect(session_id)

if fast_mode:
    # initialize all services
    print("Initializing services")
    print("Initializing STT")
    import vocal.stt.stt
    print("Initializing Chat")
    import vocal.chat.chat
    print("Initializing TTS")
    import vocal.tts.tts

    stt_instance = vocal.stt.stt.stt_instance
    chat_instance = vocal.chat.chat.chat_instance
    tts_instance = vocal.tts.tts.tts_instance

    print("Services initialized")

    # start individual tts server (used for creating previews with text, which doesnt use socket)
    # if in fast mode, this will be used for tts
    # if not in fast mode, the tts server will be started as a separate process
    @app.get("/tts")
    async def generate_tts(
        text: str = Query(..., description="Text to convert to speech"),
        config_id: str = Query(..., description="Config ID to use")
    ):        
        try:
            config = agent_manager.get_agent_config(config_id)
            if not config:
                raise HTTPException(status_code=404, detail="Config not found")

            voice_samples = config.voice_samples
            language = config.language

            voice_id = tts_instance.get_voice_id(config_id)

            tts_chunk = await tts_instance.generate_speech(text, language, voice_id, voice_samples)

            return JSONResponse({
                "audio": tts_chunk.chunk,
                "format": tts_chunk.format,
                "sample_rate": tts_chunk.sample_rate
            })
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

