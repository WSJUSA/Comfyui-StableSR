# pre-comfyui-stablsr
This is a development respository for debugging migration of StableSR to Comfyui 

There is a key bug the unet hook into Comfyui. It manifests itself as a error: mat1 and mat2 must have the same dtype. Currently I do not know how to solve configuring the diffusion model to resolve this issue. I have posted this code in hopes of finding some help from a diffusion expert to resolve it.

Put the StableSR webui_786v_139.ckpt model into Comyfui/models/stablesr/

Download the ckpt from HuggingFace https://huggingface.co/Iceclear/StableSR/blob/main/webui_768v_139.ckpt

There is a setup json in /examples/ to load the workflow into Comfyui
