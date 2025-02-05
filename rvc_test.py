from inferrvc import RVC
import os

os.environ['RVC_MODELDIR'] = 'rvc/models'
os.environ['RVC_INDEXDIR'] = 'rvc/models'

print("Loading RVC model...")
rvc_model = RVC('IShowSpeed/IShowSpeed.pth', index='IShowSpeed/IShowSpeed.index')
print("RVC model loaded")