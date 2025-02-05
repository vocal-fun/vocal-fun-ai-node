from inferrvc import RVC
import os

os.environ['RVC_MODELDIR'] = 'rvc/models'

print("Loading RVC model...")
rvc_model = RVC('IShowSpeed/IShowSpeed.pth')
print("RVC model loaded")