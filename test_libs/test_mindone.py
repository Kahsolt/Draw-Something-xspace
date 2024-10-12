#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/11 

# 通过 git 安装 mindone (对标 diffusers，支持 SD 模型)，测读库的可用性和硬件性能
# https://github.com/mindspore-lab/mindone
# https://github.com/mindspore-lab/mindone/tree/master/examples
# https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers

'''
| model | time cost (s/it) |
| :-: | :-: |
| sdxl       |   |
| sdxl-turbo |   |
| sdxl-lit   |   |
| sdv2       | ? |
| sdv2.1     | ? |
| sdv3       | ? |
'''

import os
os.environ['GLOG_v'] = '2'
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

import mindspore as ms
ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=0)
from mindone.diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, StableDiffusion3Pipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel
from openmind_hub import om_hub_download
from safetensors.torch import load_file

from time import time

dtype = ms.float16

kwargs = {
  'prompt': 'A cat holding a sign that says hello world',
  'negative_prompt': '',
  'height': 768,
  'width': 768,
  'num_inference_steps': 20,
  'guidance_scale': 7.0,
}
kwargs_turbo = {
  'prompt': 'A cat holding a sign that says hello world',
  'height': 768,
  'width': 768,
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

https://modelers.cn/models/MindSpore-Lab/stable-diffusion-xl-base-1_0
https://modelers.cn/models/PyTorch-NPU/stable-diffusion-xl-base-1.0
https://modelers.cn/models/yuazhenxiao/stable-diffusion-xl-base-1.0
'''
if not 'sdxl':
  #model_path = 'stabilityai/stable-diffusion-xl-base-1.0'
  model_path = './sd_xl_base_1.0.safetensors'
  pipe = StableDiffusionXLPipeline.from_pretrained(model_path, mindspore_dtype=dtype, use_safetensors=True, variant='fp16')
  for i in range(5):
    image = go('sdxl', pipe, kwargs)
  image.show()
  breakpoint()


'''
https://modelers.cn/models/PyTorch-NPU/sdxl-turbo
'''
if not 'sdxl-turbo':
  pipe = StableDiffusionXLPipeline.from_pretrained('stabilityai/sdxl-turbo', mindspore_dtype=dtype, variant='fp16')
  for i in range(5):
    image = go('sdxl-turbo', pipe, kwargs_turbo)
  image.show()
  breakpoint()


'''
https://modelers.cn/models/PyTorch-NPU/SDXL-Lightning
https://modelers.cn/models/Beijing-Ascend/SDXL-Lightning
https://modelers.cn/models/yuazhenxiao/SDXL-Lightning
https://modelers.cn/models/zhoubj/SDXL-Lightning
'''
if not 'sdxl-lit':
  base = 'stabilityai/stable-diffusion-xl-base-1.0'
  repo = 'ByteDance/SDXL-Lightning'
  ckpt = 'sdxl_lightning_8step_unet.safetensors'

  unet = UNet2DConditionModel.from_config(base, subfolder='unet').to(dtype)
  unet.load_state_dict(load_file(om_hub_download(repo, ckpt)))
  pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, mindspore_dtype=dtype, variant='fp16')
  pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing='trailing')
  for i in range(5):
    image = go('sdxl-lit', pipe, kwargs_turbo)
  image.show()
  breakpoint()


'''
https://modelers.cn/models/MindSpore-Lab/stable-diffusion-v2
'''
if not 'sdv2':
  model_id = 'stabilityai/stable-diffusion-2'
  pipe = StableDiffusionPipeline.from_pretrained(model_id, mindspore_dtype=dtype)
  pipe.scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder='scheduler')
  for i in range(5):
    image = go('sdv2', pipe, kwargs)
  image.show()
  breakpoint()


'''
https://modelers.cn/models/KunLun/stable-diffusion-2-1
https://modelers.cn/models/KunLun/stable-diffusion-2-1-base
https://modelers.cn/models/State_Cloud/stable-diffusion-2-1
'''
if not 'sdv2.1':
  model_id = 'stabilityai/stable-diffusion-2-1'
  pipe = StableDiffusionPipeline.from_pretrained(model_id, mindspore_dtype=dtype)
  pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
  for i in range(5):
    image = go('sdv2.1', pipe, kwargs)
  image.show()
  breakpoint()


'''
https://modelers.cn/models/MindSpore-Lab/stable_diffusion_3
'''
if not 'sdv3':
  pipe = StableDiffusion3Pipeline.from_pretrained('stabilityai/stable-diffusion-3-medium-diffusers', mindspore_dtype=dtype)
  for i in range(5):
    image = go('sdv3', pipe, kwargs)
  image.show()
  breakpoint()


'''
https://modelers.cn/models/MindSpore-Lab/LatentDiffusion_LDMTextToImagePipeline
https://modelers.cn/models/MindSpore-Lab/LatentDiffusion_LDMSuperResolutionPipeline
https://modelers.cn/models/MindSpore-Lab/StableDiffusion_upscale
https://modelers.cn/models/MindSpore-Lab/StableDiffusionXL_instruct_pix2pix
https://modelers.cn/models/MindSpore-Lab/stable_diffusion_gligen
'''
