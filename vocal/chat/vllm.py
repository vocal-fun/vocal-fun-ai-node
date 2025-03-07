from .base_llm import BaseLLM
import asyncio
import os
import time
import re
from typing import Optional
import torch
from vllm import LLM, SamplingParams, AsyncLLMEngine, AsyncEngineArgs
from transformers import AutoTokenizer
import uuid


class VLLM(BaseLLM):
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm = None
        self.engine = None
        self.is_setup = False

    def setup(self):
        if self.is_setup:
            return

        # Hugging Face login
        from huggingface_hub import login
        login(token=os.getenv('HUGGINGFACE_API_KEY'))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )

        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
               model=self.model_name,
                tensor_parallel_size=1,  # Adjust based on your setup
                gpu_memory_utilization=0.5,  # Optimize GPU usage
                dtype="float16",
                max_model_len=2048,
                max_num_seqs=20,
                quantization="fp8",
                enable_prefix_caching=True
            )
    )

        # Initialize vLLM engine
        # self.llm = LLM(
        #     model=self.model_name,
        #     tensor_parallel_size=1,  # Adjust based on your setup
        #     gpu_memory_utilization=0.9,  # Optimize GPU usage
        #     dtype="float16",
        #     max_model_len=2048,
        # )

        self.is_setup = True

    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_setup:
            raise RuntimeError("LLM not initialized")

        start_time = time.time()
        first_token_time = None
        token_count = 0
        response = ""

        # Format prompt using tokenizer
        formatted_chat = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

        # vLLM sampling parameters
        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_new_tokens", 90),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            top_k=kwargs.get("top_k", 50),
            repetition_penalty=kwargs.get("repetition_penalty", 1.2),
            stop=["User:", "USER:", "Human:", "HUMAN:"],  # Custom stopping criteria
        )

        request_id = str(uuid.uuid4())
        async for request_output in self.engine.generate(formatted_chat, sampling_params, request_id=request_id):
            for output in request_output.outputs:
                text = output.text

                # Log first token time
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                    print(f"Time to first token: {first_token_time:.3f}s")

                token_count += 1
                response = text  # Store final response            

        # Log performance
        total_time = time.time() - start_time
        tokens_per_second = token_count / total_time if total_time > 0 else 0
        print(f"\nTotal generation time: {total_time:.3f}s")
        print(f"Tokens per second: {tokens_per_second:.2f}")

        # Clean up response
        response = re.sub(r'^.*?:', '', response).strip()
        return response  # Return only final response

    async def cleanup(self):
        if self.llm:
            del self.llm
        if self.engine:
            del self.engine
        if self.tokenizer:
            del self.tokenizer
        self.is_setup = False
