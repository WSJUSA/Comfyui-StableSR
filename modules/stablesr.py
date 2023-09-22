'''
# --------------------------------------------------------------------------------
#
#   StableSR for Comfyui 
#   Migrationed from sd-webui-stablesr for Automatic1111 WebUI
#
#   Introducing state-of-the super-resolution method: StableSR!
#   Techniques is originally proposed by Jianyi Wang et, al.
#
#   Project Page: https://iceclear.github.io/projects/stablesr/
#   Official Repo: https://github.com/IceClear/StableSR
#   Paper: https://arxiv.org/abs/2305.07015
#   
#   @original author: Jianyi Wang et, al.
#   @migration: LI YI, Will James
#   @organization: Nanyang Technological University - Singapore
#   @date: 2023-09-20
#   @license: 
#       S-Lab License 1.0 (see LICENSE file)
#       CC BY-NC-SA 4.0 (required by NVIDIA SPADE module)
# 
#   @disclaimer: 
#       All code in this extension is for research purpose only. 
#       The commercial use of the code & checkpoint is strictly prohibited.
#
# --------------------------------------------------------------------------------
#
#   IMPORTANT NOTICE FOR OUTCOME IMAGES:
#       - Please be aware that the CC BY-NC-SA 4.0 license in SPADE module
#         also prohibits the commercial use of outcome images.
#       - Jianyi Wang may change the SPADE module to a commercial-friendly one.
#         If you want to use the outcome images for commercial purposes, please
#         contact Jianyi Wang for more information.
#
#   Please give LI YI's repo and also Jianyi's repo a star if you like this project!
#
# --------------------------------------------------------------------------------
'''

import os
import torch
import numpy as np
import PIL.Image as Image

import folder_paths
import nodes
import comfy.utils
import comfy.model_management

# TODO might delete this in clean up
from comfy.model_patcher import ModelPatcher

from torch import Tensor
from ldm.modules.diffusionmodules.openaimodel import UNetModel

from spade import SPADELayers
from struct_cond import EncoderUNetModelWT, build_unetwt
from util import pil2tensor, tensor2pil 

FORWARD_CACHE_NAME = 'org_forward_stablesr'

class StableSR:
    '''
    Initializes a StableSR model.

    Args:
        path: The path to the StableSR checkpoint file.
        dtype: The data type of the model. If not specified, the default data type will be used.
        device: The device to run the model on. If not specified, the default device will be used.
    '''
        
    def __init__(self, path, dtype, device):
        print(f"[StbaleSR] in StableSR init - dtype: {dtype}, device: {device}")

        state_dict = comfy.utils.load_torch_file(path)

        self.struct_cond_model: EncoderUNetModelWT = build_unetwt()
        self.spade_layers: SPADELayers = SPADELayers()
        self.struct_cond_model.load_from_dict(state_dict)
        self.spade_layers.load_from_dict(state_dict)
        del state_dict

        self.struct_cond_model.apply(lambda x: x.to(dtype=dtype, device=device))
        self.spade_layers.apply(lambda x: x.to(dtype=dtype, device=device))
        self.latent_image: Tensor = None
        self.set_image_hooks = {}
        self.struct_cond: Tensor = None

    def set_latent_image(self, latent_image):
        self.latent_image = latent_image["samples"]
        for hook in self.set_image_hooks.values():
            hook(latent_image)

    '''
    # attempt to use Comfyui ModelPatcher.set_model_unet_function_wrapper()
    # hasn't been successful due to timestep complexity
    def sr_unet_forward(self, model_function, args_dict):
        try:
            # explode packed args
            input_x = args_dict.get("input")
            timestep_ = args_dict.get("timestep")
            c = args_dict.get("c")
            cond_or_uncond = args_dict.get("cond_or_uncond")

            # set latent image to device
            device = comfy.model_management.get_torch_device()
            latent_image = self.latent_image["samples"]
            latent_image = latent_image.to(device)

            timestep_ = timestep_.to(torch.float32)

            # Ensure the device of all modules layers is the same as the unet
            # This will fix the issue when user use --medvram or --lowvram
            self.spade_layers.to(device)
            self.struct_cond_model.to(device)

            #timestep_ = timestep_.to(device)
            self.struct_cond = None # mitigate vram peak
            self.struct_cond = self.struct_cond_model(latent_image, timestep_[:latent_image.shape[0]])

            # Call the model_function with the provided arguments
            result = model_function(input_x, timestep_, **c)

            # Return the result
            return result
        except Exception as e:
            print(f"[StbaleSR] Error in sr_unet_forward: {str(e)}")
            raise e
    
    def sr_hook(self, sd_model)
        # try set forward handler using ModelPatcher.set_model_unet_function_wrapper()
        #sd_model.set_model_unet_function_wrapper(self.sr_unet_forward)

    '''
            
    def hook(self, unet: UNetModel):
        # hook unet to set the struct_cond
        if not hasattr(unet, FORWARD_CACHE_NAME):
            setattr(unet, FORWARD_CACHE_NAME, unet.forward)

        print(f"[StbaleSR] in StableSR hook - unet dtype: {unet.dtype}")

        def unet_forward(x, timesteps=None, context=None, y=None,**kwargs):
            # debug print the dtypes going in
            print(f'[StableSR] in unet_forward()')
            print(f"[StbaleSR] in StableSR hook unet_forward - dtype timesteps: {timesteps.dtype}")
            print(f"[StbaleSR] in StableSR hook unet_forward - dtype latent_image: {self.latent_image.dtype}")

            self.latent_image = self.latent_image.to(x.device)

            # Ensure the device of all modules layers is the same as the unet
            # This will fix the issue when user use --medvram or --lowvram
            self.spade_layers.to(x.device)
            self.struct_cond_model.to(x.device)
            timesteps = timesteps.to(x.device)
            self.struct_cond = None # mitigate vram peak
            self.struct_cond = self.struct_cond_model(self.latent_image, timesteps[:self.latent_image.shape[0]])
            return getattr(unet, FORWARD_CACHE_NAME)(x, timesteps, context, y, **kwargs)

        unet.forward = unet_forward
        
        # set the spade_layers on unet
        self.spade_layers.hook(unet, lambda: self.struct_cond)

    '''
    # TODO migrate unhook
    def unhook(self, unet: UNetModel):
        # clean up cache
        self.latent_image = None
        self.struct_cond = None
        self.set_image_hooks = {}
        # unhook unet forward
        if hasattr(unet, FORWARD_CACHE_NAME):
            unet.forward = getattr(unet, FORWARD_CACHE_NAME)
            delattr(unet, FORWARD_CACHE_NAME)

        # unhook spade layers
        self.spade_layers.unhook()
    '''

class StableSRScript():
    params = None

    def __init__(self, upscale_factor, seed, steps, cfg, sampler_name, scheduler, denoise, pure_noise, basic_pipe, model, 
                 hook_opt=None) -> None:
        self.params = upscale_factor, seed, steps, cfg, sampler_name, scheduler, denoise, pure_noise, basic_pipe, model
        self.hook = hook_opt
        self.stablesr_model_path = None
        self.get_stablesr_model_path(model)
        self.stablesr_module: StableSR = None
        self.init_latent = None

    def get_stablesr_model_path(self, model):
        if self.stablesr_model_path is None:
            file_path = folder_paths.get_full_path("stablesr", model)
            if os.path.isfile(file_path):
                # save tha absolute path
                self.stablesr_model_path = file_path
            else:
                print(f'[StableSR] Invalid StableSR model reference')
        return self.stablesr_model_path

    def upscale_tensor_as_pil(self, image, scale_factor, save_temp_prefix=None):
        # Convert the PyTorch tensor to a PIL Image object.
        pil_image = tensor2pil(image)

        w = int(pil_image.width * scale_factor)
        h = int(pil_image.height * scale_factor)

        # if the target width is not dividable by 8, then round it up
        if w % 8 != 0:
            w = w + 8 - w % 8
        # if the target height is not dividable by 8, then round it up
        if h % 8 != 0:
            h = h + 8 - h % 8

        # Resize the PIL Image object using Lanczos interpolation.
        resized_image = pil_image.resize((w, h), Image.LANCZOS)

        resized_tensor = pil2tensor(resized_image)

        return resized_tensor

    def to_latent_image_with_vae(self, pixels, vae):
        x = pixels.shape[1]
        y = pixels.shape[2]
        if pixels.shape[1] != x or pixels.shape[2] != y:
            pixels = pixels[:, :x, :y, :]
        t = vae.encode(pixels[:, :, :, :3])
        return {"samples": t}

    # sampler wrapper
    def sample(self, image) -> Image:
        upscale_factor, seed, steps, cfg, sampler_name, scheduler, denoise, pure_noise, basic_pipe, model = self.params
        sd_model, clip, vae, positive, negative = basic_pipe

        # initial upscale on pixels to get target size and color map
        upscaled_image = self.upscale_tensor_as_pil(image, upscale_factor)

        # get the initial upscaled latent image
        self.init_latent = self.to_latent_image_with_vae(upscaled_image, vae)
        
        # get the device
        device = comfy.model_management.get_torch_device()

        # get dtype from sd model
        dtype = sd_model.model_dtype()

        # load StableSR
        if self.stablesr_module is None:
            self.stablesr_module = StableSR(self.stablesr_model_path, dtype, device)

        # set latent image on stablesr
        self.stablesr_module.set_latent_image(self.init_latent)

        # get the stablediffusion unet referrence from the nested BaseModel instance
        unet: UNetModel = sd_model.model.diffusion_model

        # hook unet forwards
        self.stablesr_module.hook(unet)

        # get an empty latent for ksampler, it will generate a random tensor from the seed
        empty_latent = {"samples": torch.zeros(self.init_latent["samples"].shape)}

        # run ksampler
        print('[StableSR] Target image size: {}x{}'.format(upscaled_image.shape[2], upscaled_image.shape[1]))
        refined_latent = \
        nodes.common_ksampler(sd_model, seed, steps, cfg, sampler_name, scheduler, positive, negative, empty_latent, denoise)[0]

        '''
        # TODO migrate variable noise
        if pure_noise:
            # NOTE: use txt2img instead of img2img sampling
            samples = sampler.sample(p, x, conditioning, unconditional_conditioning, image_conditioning=p.image_conditioning)
        else:
            if p.initial_noise_multiplier != 1.0:
                p.extra_generation_params["Noise multiplier"] =p.initial_noise_multiplier
                x *= p.initial_noise_multiplier
            samples = sampler.sample_img2img(p, p.init_latent, x, conditioning, unconditional_conditioning, image_conditioning=p.image_conditioning)

        # TODO migrate mask
        if p.mask is not None:
            print("[StableSR] trace - in sample_custom() - p.mask is applied")

            samples = samples * p.nmask + p.init_latent * p.mask
        del x
        devices.torch_gc()
        '''

        # decode latent
        refined_image = vae.decode(refined_latent['samples']) # final sr image - no color correction
        color_map_image = upscaled_image # pretty name
        return refined_image, color_map_image

        '''
        # TODO migrate unhook
        self.stablesr_model.unhook(unet)
        # in --medvram and --lowvram mode, we send the model back to the initial device
        self.stablesr_model.struct_cond_model.to(device=first_param.device)
        self.stablesr_model.spade_layers.to(device=first_param.device)
        '''