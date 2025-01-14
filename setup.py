# setup.py
from setuptools import setup, find_packages

setup(
    name="vocal-ai",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-multipart",
        "websockets",
        "numpy",
        "python-dotenv",
        "ffmpeg-python",
    ],
)