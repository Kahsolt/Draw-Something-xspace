#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/12 

# 测试本地 diffusers 库的可用性和硬件性能

'''
| model \ time | 1024@20 | 1024@10 | 768@20 | 768@10 | 512@20 | 512@10 |
| :-: | :-: | :-: | :-: |
| sdxl | 16.0 | 8.4 | 8.7 | 4.6 | 4.2 | 2.2 |

| model \ time | 1024@8 | 1024@4 | 1024@2 | 768@8 | 768@4 | 512@8 | 512@4 |
| :-: | :-: | :-: | :-: |
| sdxl-turbo | 4.2 | 2.5 | 1.6 | 2.4 | 1.5 | 1.3 | 0.8 |
| sdxl-lit   | 3.8 | 2.3 | 1.5 | 2.3 | 1.3 | 1.1 | 0.7 |
'''

from time import time
import warnings ; warnings.filterwarnings(action='ignore', category=UserWarning)
import warnings ; warnings.filterwarnings(action='ignore', category=FutureWarning)

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16

kwargs = {
  'prompt': 'A cat holding a sign that says hello world',
  'negative_prompt': '',
  'height': 1024,
  'width': 1024,
  'num_inference_steps': 20,
  'guidance_scale': 7.0,
}
kwargs_turbo = {
  'prompt': 'A cat holding a sign that says hello world',
  'height': 1024,
  'width': 1024,
  'num_inference_steps': 8,
  'guidance_scale': 0.0,    # MUST be off
}


def go(repo:str, pipe:StableDiffusionPipeline, kwargs:dict):
  ts_start = time()
  images = pipe(**kwargs)
  ts_end = time()
  print(f'>> {repo}: {ts_end - ts_start:.3f}s')
  return images.images[0]


'''
https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
'''
if not 'sdxl':
  pipe = StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=dtype, use_safetensors=True, variant='fp16')
  pipe.to(device)
  for i in range(5):
    image = go('sdxl', pipe, kwargs)
  image.show()
  breakpoint()


'''
https://huggingface.co/stabilityai/sdxl-turbo
'''
if not 'sdxl-turbo':
  pipe = StableDiffusionXLPipeline.from_pretrained('stabilityai/sdxl-turbo', torch_dtype=dtype, variant='fp16')
  pipe.to(device)
  for i in range(5):
    image = go('sdxl-turbo', pipe, kwargs_turbo)
  image.show()
  breakpoint()


'''
https://huggingface.co/ByteDance/SDXL-Lightning
'''
if 'sdxl-lit':
  base = 'stabilityai/stable-diffusion-xl-base-1.0'
  repo = 'ByteDance/SDXL-Lightning'
  ckpt = 'sdxl_lightning_8step_unet.safetensors'

  unet = UNet2DConditionModel.from_config(base, subfolder='unet').to(device, dtype)
  unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
  pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=dtype, variant='fp16').to(device)
  pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing='trailing')
  pipe.to(device)
  for i in range(5):
    image = go('sdxl-lit', pipe, kwargs_turbo)
  image.show()
  breakpoint()


'''
https://huggingface.co/stabilityai/stable-diffusion-2
'''
if not 'sdv2':
  model_id = 'stabilityai/stable-diffusion-2'
  pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
  pipe.scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder='scheduler')
  pipe.to(device)
  for i in range(5):
    image = go('sdv2', pipe, kwargs)
  image.show()
  breakpoint()


'''
https://huggingface.co/stabilityai/stable-diffusion-2-1
'''
if not 'sdv2.1':
  model_id = 'stabilityai/stable-diffusion-2-1'
  pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
  pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
  pipe.to(device)
  for i in range(5):
    image = go('sdv2.1', pipe, kwargs)
  image.show()
  breakpoint()
