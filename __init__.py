"""
@author: WSJUSA
@title: StableSR
@nickname: StableSR
@description: This module enables StableSR in Comgfyui. Ported work of sd-webui-stablesr. Original work for Auotmaatic1111 version of this module and StableSR credit to LIightChaser and Jianyi Wang.
"""
import folder_paths
import os
import sys

comfy_path = os.path.dirname(folder_paths.__file__)
impact_path = os.path.join(os.path.dirname(__file__))
modules_path = os.path.join(os.path.dirname(__file__), "modules")

sys.path.append(modules_path)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']