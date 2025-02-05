import os
import requests

# Define the files and their URLs
files = {
    "hubert_base.pt": "https://huggingface.co/Daswer123/RVC_Base/resolve/main/hubert_base.pt",
    "rmvpe.pt": "https://huggingface.co/Daswer123/RVC_Base/resolve/main/rmvpe.pt",
}

# Create the rvc folder if it doesn't exist
os.makedirs("rvc", exist_ok=True)

# Download the files
for filename, url in files.items():
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"rvc/{filename}", "wb") as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename} from {url}")
