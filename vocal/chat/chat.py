from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from typing import Optional
import os
from dotenv import load_dotenv
from .base_llm import BaseLLM
from .local_llm import LocalLLM
from .external.groq_llm import GroqLLM
from .conversation import ConversationManager, ConversationFormatter

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize conversation manager
conversation_manager = ConversationManager(max_history=1)
conversation_formatter = ConversationFormatter()

class Chat:
    def __init__(self):
        self.llm: Optional[BaseLLM] = None
        self.conversation_history = []
        self._setup_llm()

    def _setup_llm(self):
        use_external = os.getenv("USE_EXTERNAL_CHAT", "False").lower() == "true"
        provider = os.getenv("EXTERNAL_CHAT_PROVIDER", "").lower()

        if use_external:
            if provider == "groq":
                self.llm = GroqLLM()
            else:
                raise ValueError(f"Unsupported external chat provider: {provider}")
        else:
            self.llm = LocalLLM()

        self.setup()

    def setup(self):
        """Initialize the LLM"""
        if self.llm:
            self.llm.setup()

    async def generate_response(self, data: dict, **kwargs) -> str:
        """Generate response for the given prompt"""
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        session_id = data["session_id"]
        user_message = data["text"]
        config_id = data.get("config_id", "default")
        
        print(f"Client {session_id} INPUT {user_message}")
                
        history = conversation_manager.get_history(session_id)

        messages = conversation_formatter.format_messages(
            config_id,
            history,
            user_message
        )

        print(f"Client {session_id} MESSAGES {messages}")
        # Generate response
        response = await self.llm.generate(messages, **kwargs)

        # Clean up response if needed
        response = conversation_formatter.cleanup_response(response)
    
        conversation_manager.add_conversation(session_id, user_message, response)

        print(f"Client {session_id} RESPONSE {response}")
        return response

    async def cleanup(self):
        """Cleanup resources"""
        if self.llm:
            await self.llm.cleanup()

# Initialize chat instance
chat_instance = Chat()

@app.post("/chat")
async def generate_response(data: dict):
    try:
        # Generate response
        response = await chat_instance.generate_response(data)
        return {"response": response}
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    