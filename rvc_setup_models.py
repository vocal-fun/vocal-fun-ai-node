import os
import requests
from zipfile import ZipFile
import io

# Define the model download paths and model names
models = [
    {
        "url": "https://huggingface.co/Plasmati/AIDRIVEORNOTMODELS3/resolve/main/IShowSpeed.zip?download=true",
        "name": "IShowSpeed"
    }
    # You can add more models here in the same format
]

# Create the rvc/models folder if it doesn't exist
os.makedirs("rvc/models", exist_ok=True)

def download_and_extract(model_url, model_name):
    response = requests.get(model_url)
    
    if response.status_code == 200:
        print(f"Downloading {model_name}...")

        # Create a model-specific directory inside rvc/models
        model_dir = f"rvc/models/{model_name}"
        os.makedirs(model_dir, exist_ok=True)
        
        # If it's a zip file
        if model_url.endswith('.zip'):
            zip_file = ZipFile(io.BytesIO(response.content))
            zip_file.extractall(f"rvc/models/{model_name}")
            
            # Find the .pth and .index files
            for file_name in zip_file.namelist():
                if file_name.endswith('.pth'):
                    os.rename(f"rvc/models/{model_name}/{file_name}", f"rvc/models/{model_name}/{model_name}.pth")
                elif file_name.endswith('.index'):
                    os.rename(f"rvc/models/{model_name}/{file_name}", f"rvc/models/{model_name}/{model_name}.index")
        else:
            # For non-zip files, just download and rename
            with open(f"rvc/models/{model_name}/{model_name}.pth", "wb") as f:
                f.write(response.content)
        
        print(f"{model_name} downloaded and renamed successfully.")
    else:
        print(f"Failed to download {model_name} from {model_url}")

# Download and process each model
for model in models:
    download_and_extract(model["url"], model["name"])
