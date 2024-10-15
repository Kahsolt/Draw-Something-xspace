#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/12 

# 扩撒模型引擎 Pytorch  后端

from copy import deepcopy
from utils import *

import torch
from diffusers.pipelines import StableDiffusionXLPipeline
#from diffusers.schedulers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler, UniPCMultistepScheduler
from diffusers.models import UNet2DConditionModel, AutoencoderTiny
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16

base_repo = 'stabilityai/stable-diffusion-xl-base-1.0'
vae_repo  = 'madebyollin/taesdxl'
unet_repo = 'ByteDance/SDXL-Lightning'
unet_ckpt = 'sdxl_lightning_8step_unet.safetensors'


pipe: StableDiffusionXLPipeline = None
vae: AutoencoderTiny = None
vae_decode: Callable = None

def init_model():
  global pipe, vae, vae_decode
  if pipe and vae and vae_decode: return
  unet = UNet2DConditionModel.from_config(base_repo, subfolder='unet').to(dtype)
  unet.load_state_dict(load_file(hf_hub_download(unet_repo, unet_ckpt)))
  vae = AutoencoderTiny.from_pretrained(vae_repo, torch_dtype=dtype)
  pipe = StableDiffusionXLPipeline.from_pretrained(base_repo, vae=vae, unet=unet, torch_dtype=dtype, use_safetensors=True, variant='fp16', add_watermarker=False)
  #pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
  pipe.to(device)
  vae_decode = lambda latent: vae.decode(latent, return_dict=False)[0][0].permute(1, 2, 0).div_(2).add_(0.5).clamp_(0.0, 1.0).mul_(255).byte().cpu().numpy()


latents = []
def peep_callback(pipe, step, timestep, callback_kwargs):
  global latents
  if step in SD_PEEP_STEPS:
    latents.append(callback_kwargs['latents'])
  return callback_kwargs


def rand_image_set(prompt:str) -> List[npimg]:
  init_model()
  kwargs = deepcopy(SD_PIPE_CALL_KWARGS)
  kwargs['prompt'] = prompt

  with torch.inference_mode():
    pipe(**kwargs, callback_on_step_end=peep_callback)
    imgs = [vae_decode(latent) for latent in latents]
  latents.clear()
  return imgs
