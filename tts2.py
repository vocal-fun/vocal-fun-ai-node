from auralis import TTS, TTSRequest

import os
import logging
from dotenv import load_dotenv

load_dotenv()

xttsPath = os.getenv('XTTS_MODEL_PATH')
xttsGptPath = os.getenv('XTTS_GPT_MODEL_PATH')
tts = TTS().from_pretrained(xttsPath, gpt_model=xttsGptPath)

# Generate speech
request = TTSRequest(
    text="We are gonna make america great again. We are gonna build a big beatiful wall and we are gonna make healthcare better. We are gonna make education better. We are gonna make the economy better. We are gonna make america great again.",
    speaker_files=['config/converted_agents/Donald Trump/Donald Trump.wav'],
    stream=True
)

output = tts.generate_speech(request)
output.save('hello.wav')