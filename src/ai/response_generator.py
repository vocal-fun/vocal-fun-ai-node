from typing import Optional
# from src.ai.mistral import generate_response_mistral
# from src.ai.gptj import generate_response_gptj
from src.ai.bloom import generate_response_bloom


class ResponseGenerator:
    def __init__(self):
        pass

    async def generate_response(self, text: str) -> Optional[str]:
        response = await generate_response_bloom(text)
        return response