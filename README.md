# pre-comfyui-stablsr
This is a development respository for debugging migration of StableSR to Comfyui 

There is a key bug the unet hook into Comfyui

Put the StableSR webui_786v_139.ckpt model into Comyfui/models/stablesr/

Download the ckpt from HuggingFace https://huggingface.co/Iceclear/StableSR/blob/main/webui_768v_139.ckpt

There is a setup json in /examples/ to load the workflow into Comfyui
