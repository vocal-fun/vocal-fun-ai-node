import asyncio
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# Fix relative imports
from websocket.connection_manager import ConnectionManager
from audio.speech_to_text import SpeechToText
from audio.text_to_speech import TextToSpeech
from ai.response_generator import ResponseGenerator
from config.settings import settings

app = FastAPI()
connection_manager = ConnectionManager()
stt = SpeechToText()
tts = TextToSpeech()
ai = ResponseGenerator()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = str(uuid.uuid4())
    await connection_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive audio data from client
            audio_data = await websocket.receive_bytes()
            
            # Convert speech to text
            text = await stt.process_audio(audio_data)
            if text:
                # Send transcribed text to client (optional)
                await connection_manager.send_text(client_id, f"Transcribed: {text}")
                
                # Generate AI response
                response = await ai.generate_response(text)
                if response:
                    # Send AI text response to client (optional)
                    await connection_manager.send_text(client_id, f"AI: {response}")
                    
                    # Generate speech from AI response
                    audio_response = await tts.generate_speech(response)
                    if audio_response:
                        # Send audio back to client
                        await connection_manager.send_audio(client_id, audio_response)
    
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
    except Exception as e:
        print(f"Error processing websocket message: {e}")
        connection_manager.disconnect(client_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )