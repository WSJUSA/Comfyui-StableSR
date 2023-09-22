import os
import comfy
import numpy as np
import PIL.Image as Image
import torch

import folder_paths
model_path = folder_paths.models_dir
folder_paths.folder_names_and_paths["stablesr"] = ([os.path.join(model_path, "stablesr")], folder_paths.supported_pt_extensions)

import stablesr
from colorfix import adain_color_fix, wavelet_color_fix
from util import pil2tensor, tensor2pil

class ColorFix:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "color_map_image": ("IMAGE", ),
                    "color_fix": (["Wavelet", "AdaIN",],),
                    },
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fix_color"
    CATEGORY = "image"

    def fix_color(self, image, color_map_image, color_fix):
        print(f'[StableSR] fix_color')
        try:
            color_fix_func = wavelet_color_fix if color_fix == 'Wavelet' else adain_color_fix
            result_image = color_fix_func(tensor2pil(image), tensor2pil(color_map_image))
            refined_image = pil2tensor(result_image)
            return (refined_image, )
        except Exception as e:
            print(f'[StableSR] Error fix_color: {e}')

 
class StableSRUpscalerPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "upscale_factor": ("FLOAT", {"default": 1.5, "min": 1, "max": 10000, "step": 0.1}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "pure_noise": ("BOOLEAN", {"label_on": "enabled", "label_off": "disabled"}),
                    "basic_pipe": ("BASIC_PIPE",),
                    "stablesr_model": (folder_paths.get_filename_list("stablesr"), ),
                    },
                "optional": {
                        "pk_hook_opt": ("PK_HOOK", ),
                    }
                }

    RETURN_TYPES = ("IMAGE","IMAGE", )
    RETURN_NAMES = ("stablesr_image","color_map_image", )
    FUNCTION = "doit_pipe"
    CATEGORY = "image/upscaling"

    def doit_pipe(self, image, upscale_factor, seed, steps, cfg, sampler_name, scheduler, denoise, pure_noise, basic_pipe, stablesr_model, pk_hook_opt=None):
        upscaler = stablesr.StableSRScript(upscale_factor, seed, steps, cfg, sampler_name, scheduler, denoise, pure_noise, basic_pipe, stablesr_model, pk_hook_opt)
        upscale_image, color_map_image = upscaler.sample(image)
        return (upscale_image, color_map_image, )
    
NODE_CLASS_MAPPINGS = {
    "ColorFix": ColorFix,
    "StableSRUpscalerPipe": StableSRUpscalerPipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorFix": "ColorFix",
    "StableSRUpscalerPipe": "StableSRUpscaler (pipe)",
}