'''
    This file is modified from the multidiffusion-upscaler-for-automatic1111 TiledVAE attn.py, so that the StableSR can save much VRAM.
    Add modified further for Comfyui
'''
import math
import torch
import comfy.model_management

'''
# DELETE from sd-webui-stablesr version
from modules import shared, sd_hijack
from modules.sd_hijack_optimizations import get_available_vram, get_xformers_flash_attention_op, sub_quad_attention
'''

try:
    import xformers
    import xformers.ops
except ImportError:
    pass

# Simplified version of attn_forward switch -- only xformers for now -- see get_attn_forward original below to add capability
def sr_get_attn_func():
    if comfy.model_management.xformers_enabled():
        return sr_xformers_attnblock_forward
    
    return sr_attn_forward

# The following functions are all copied from stable-diffusion-webui modules.sd_hijack_optimizations
# However, the residual & normalization are removed and computed separately.
# And addapted for Comfyui

# Note no change from original attn_function
def sr_attn_forward(q, k, v):
    # compute attention
    # q: b,hw,c
    k = k.permute(0, 2, 1)  # b,c,hw
    c = k.shape[1]
    w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
    w_ = w_ * (int(c)**(-0.5))
    w_ = torch.nn.functional.softmax(w_, dim=2)

    # attend to values
    v = v.permute(0, 2, 1)   # b,c,hw
    w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
    # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
    h_ = torch.bmm(v, w_)

    return h_.permute(0, 2, 1)

def sr_xformers_attnblock_forward(q, k, v):
    return xformers.ops.memory_efficient_attention(q, k, v, op=sr_get_xformers_flash_attention_op(q, k, v))

# barrow get_xformers_flash_attention_op from auto1111
def sr_get_xformers_flash_attention_op(q, k, v):
    '''
    # this may be a problem
    if not shared.cmd_opts.xformers_flash_attention:
        return None
    '''
    try:
        flash_attention_op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
        fw, bw = flash_attention_op
        if fw.supports(xformers.ops.fmha.Inputs(query=q, key=k, value=v, attn_bias=None)):
            return flash_attention_op
    except Exception as e:
        print(f'[StableSR] Error getting Flash attention handler: {e}')
    return None


'''
The original sd-web-ui get_attn_func() and mapped handlers for reference
def get_attn_func():
    method = sd_hijack.model_hijack.optimization_method
    print(f"[StableSR] in get_attn_function - optimization method: {method}")
    if method is None:
        return attn_forward
    method = method.lower()
    # The method should be one of the following:
    # ['none', 'sdp-no-mem', 'sdp', 'xformers', ''sub-quadratic', 'v1', 'invokeai', 'doggettx']
    if method not in ['none', 'sdp-no-mem', 'sdp', 'xformers', 'sub-quadratic', 'v1', 'invokeai', 'doggettx']:
        print(f"[StableSR] Warning: Unknown attention optimization method {method}. Please try to update the extension.")
        return attn_forward
    
    if method == 'none':
        return attn_forward
    elif method == 'xformers':
        return xformers_attnblock_forward
    elif method == 'sdp-no-mem':
        return sdp_no_mem_attnblock_forward
    elif method == 'sdp':
        return sdp_attnblock_forward
    elif method == 'sub-quadratic':
        return sub_quad_attnblock_forward
    elif method == 'doggettx':
        return cross_attention_attnblock_forward
    
    return attn_forward


# The following functions are all copied from modules.sd_hijack_optimizations
# However, the residual & normalization are removed and computed separately.

def attn_forward(q, k, v):
    # compute attention
    # q: b,hw,c
    k = k.permute(0, 2, 1)  # b,c,hw
    c = k.shape[1]
    w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
    w_ = w_ * (int(c)**(-0.5))
    w_ = torch.nn.functional.softmax(w_, dim=2)

    # attend to values
    v = v.permute(0, 2, 1)   # b,c,hw
    w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
    # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
    h_ = torch.bmm(v, w_)

    return h_.permute(0, 2, 1)

def xformers_attnblock_forward(q, k, v):
    return xformers.ops.memory_efficient_attention(q, k, v, op=get_xformers_flash_attention_op(q, k, v))
   

def cross_attention_attnblock_forward(q, k, v):
    # compute attention
    k = k.permute(0, 2, 1)# b,c,hw
    v = v.permute(0, 2, 1)# b,c,hw
    c = k.shape[1]
    h_ = torch.zeros_like(k, device=q.device)

    mem_free_total = get_available_vram()

    tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
    mem_required = tensor_size * 2.5
    steps = 1

    if mem_required > mem_free_total:
        steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))

    slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
    for i in range(0, q.shape[1], slice_size):
        end = i + slice_size

        w1 = torch.bmm(q[:, i:end], k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w2 = w1 * (int(c)**(-0.5))
        del w1
        w3 = torch.nn.functional.softmax(w2, dim=2, dtype=q.dtype)
        del w2

        # attend to values
        w4 = w3.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        del w3

        h_[:, :, i:end] = torch.bmm(v, w4)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        del w4

    return h_.permute(0, 2, 1)

def sdp_no_mem_attnblock_forward(q, k, v):
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
        return sdp_attnblock_forward(q, k, v)

def sdp_attnblock_forward(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

def sub_quad_attnblock_forward(q, k, v):
    return sub_quad_attention(q, k, v, q_chunk_size=shared.cmd_opts.sub_quad_q_chunk_size, kv_chunk_size=shared.cmd_opts.sub_quad_kv_chunk_size, chunk_threshold=shared.cmd_opts.sub_quad_chunk_threshold, use_checkpoint=True)
'''