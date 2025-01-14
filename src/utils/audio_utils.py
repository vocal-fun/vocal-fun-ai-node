# src/utils/audio_utils.py
import ffmpeg
import numpy as np
from pathlib import Path
import tempfile
from typing import Union, Tuple
import os
from src.config.settings import settings

class AudioConverter:
    @staticmethod
    async def webm_to_wav(webm_data: bytes) -> bytes:
        """Convert WebM audio data to WAV format"""
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file:
            webm_file.write(webm_data)
            webm_path = webm_file.name

        try:
            # Read input WebM file
            stream = ffmpeg.input(webm_path)
            
            # Convert to WAV with specified parameters
            stream = ffmpeg.output(
                stream,
                'pipe:',
                f='wav',
                acodec='pcm_s16le',
                ac=settings.CHANNELS,
                ar=settings.SAMPLE_RATE
            )
            
            # Run the conversion
            out, _ = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            return out

        finally:
            os.unlink(webm_path)

    @staticmethod
    async def wav_to_webm(wav_data: bytes) -> bytes:
        """Convert WAV audio data to WebM format"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
            wav_file.write(wav_data)
            wav_path = wav_file.name

        try:
            # Read input WAV file
            stream = ffmpeg.input(wav_path)
            
            # Convert to WebM with specified parameters
            stream = ffmpeg.output(
                stream,
                'pipe:',
                f='webm',
                acodec='libopus',
                ac=settings.CHANNELS,
                ar=settings.SAMPLE_RATE,
                audio_bitrate=f'{settings.OPUS_BITRATE}'  # Fixed syntax for bitrate
            )
            
            # Run the conversion
            out, _ = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            return out

        finally:
            os.unlink(wav_path)

    @staticmethod
    def get_audio_info(audio_data: bytes, format: str) -> dict:
        """Get audio metadata"""
        with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name

        try:
            probe = ffmpeg.probe(temp_path)
            audio_info = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
            return {
                'sample_rate': int(audio_info.get('sample_rate', 0)),
                'channels': int(audio_info.get('channels', 0)),
                'codec': audio_info.get('codec_name', ''),
                'duration': float(probe.get('format', {}).get('duration', 0))
            }
        finally:
            os.unlink(temp_path)