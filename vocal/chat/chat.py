from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from typing import Optional
import os
import time
from dotenv import load_dotenv
from .base_llm import BaseLLM
from .conversation import ConversationManager, ConversationFormatter
from fastapi import APIRouter
from vocal.config.agents_config import agent_manager
import uuid

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
                from .external.groq_llm import GroqLLM
                self.llm = GroqLLM()
            else:
                raise ValueError(f"Unsupported external chat provider: {provider}")
        else:
            from .vllm import VLLM
            from .local_llm import LocalLLM
            self.llm = VLLM()

        self.setup()

    def setup(self):
        """Initialize the LLM"""
        if self.llm:
            self.llm.setup()

    async def generate_response(self, data: dict, **kwargs) -> str:
        """Generate response for the given prompt"""
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        session_id = data.get("session_id", str(uuid.uuid4()))
        user_message = data.get("text", "")
        config_id = data.get("config_id", "default")
    
        t0 = time.time()
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

        print("Chat response time: ", time.time() - t0)
        print(f"Client {session_id} RESPONSE {response}")
        return response

    async def cleanup(self):
        """Cleanup resources"""
        if self.llm:
            await self.llm.cleanup()

# Initialize chat instance
chat_instance = Chat()

chat_router = APIRouter()

@chat_router.post("/chat")
async def generate_response(data: dict):
    try:
        # if client sends config, add it to the agent manager
        config = data.get("config", {})
        if config:
             await agent_manager.add_agent_config(config)

        response = await chat_instance.generate_response(data)
        return {"response": response}
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    