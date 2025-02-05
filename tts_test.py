from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from TTS.api import TTS
import base64
import os

# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# output_path = "response_speed.wav"

# tts.tts_to_file(
#         text=" Look who's gettin' all feisty! You think you can come at me like that, bro? I got a whole army of simps ready to defend me! You just mad 'cause you ain't gettin' no attention from the ladies, ain't nobody got time for that salty attitude, bruh!",
#         speaker_wav="voices/speed.wav",
#         language="en",
#         file_path=output_path,
#         split_sentences=False,
#         speed=2
#     )


tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24")
tts.voice_conversion_to_file(source_wav="response_speed.wav", target_wav="voices/speed.wav", file_path="response_speed_converted.wav")