#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/12 

# 扩撒模型引擎 Mindspore 后端

from copy import deepcopy
from utils import *  # must before `import mindone`

import mindspore as ms
from mindone.diffusers.pipelines import StableDiffusionXLPipeline
from mindone.diffusers.schedulers import EulerDiscreteScheduler
from mindone.diffusers.models import ModelMixin, UNet2DConditionModel, AutoencoderTiny
from mindone.diffusers.models.modeling_utils import load_state_dict
from openmind_hub import om_hub_download, snapshot_download

# NOTE: 图模式 ms.GRAPH_MODE 下非常慢，可能触发了动态shape导致反复编译 :(
ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU' if IS_WIN else 'Ascend', device_id=0)
dtype = ms.float16

# repo path
base_repo  = 'PyTorch-NPU/stable-diffusion-xl-base-1.0'
unet_repo  = 'PyTorch-NPU/SDXL-Lightning'
vae_repo   = 'OpenSource/taesdxl'
mdl_fn     = 'diffusion_pytorch_model.safetensors'
unet_fn    = 'sdxl_lightning_8step_unet.safetensors'
cfg_fn     = 'config.json'
unet_cfg   = om_hub_download(base_repo, cfg_fn,  revision='main', subfolder='unet')
unet_cache = om_hub_download(unet_repo, unet_fn, revision='main')
vae_cache  = snapshot_download(vae_repo,  revision='main', ignore_patterns=[
  'examples/*', '*.md', '.git*',
  '*.bin', 'taesdxl_*', 'fusion_result.json',
])
base_cache = snapshot_download(base_repo, revision='main', ignore_patterns=[
  'examples/*', '*.md', '*.png', '.mv', '.mdl', '.msc', '.git*',
  '*lora*', 'openvino*', '*.onnx',
  '*/model.safetensors',  # since we use varinat='fp16'
  'unet/*', 'vae*/*', 
])

pipe: StableDiffusionXLPipeline = None
vae: AutoencoderTiny = None
vae_decode: Callable = None

# modifed from ModelMixin.from_pretrained()
def load_pretrained_unet(model_file:str):
  model: ModelMixin = UNet2DConditionModel.from_config(unet_cfg).to(dtype)
  state_dict = load_state_dict(model_file, variant='fp16')
  model._convert_deprecated_attention_blocks(state_dict)
  model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = UNet2DConditionModel._load_pretrained_model(model, state_dict, model_file, unet_repo)
  loading_info = {
    'missing_keys': missing_keys,
    'unexpected_keys': unexpected_keys,
    'mismatched_keys': mismatched_keys,
    'error_msgs': error_msgs,
  }
  print(f'[UNet2DConditionModel loading_info]')
  print(loading_info)
  model.register_to_config(_name_or_path=unet_repo)
  model.set_train(False)
  return model

def init_model():
  global pipe, vae, vae_decode
  if pipe and vae and vae_decode: return
  vae  = AutoencoderTiny.from_pretrained(vae_cache).to(dtype)
  unet = load_pretrained_unet(unet_cache)
  pipe = StableDiffusionXLPipeline.from_pretrained(base_cache, vae=vae, unet=unet, mindspore_dtype=dtype, use_safetensors=True, variant='fp16', add_watermarker=False)
  pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing='trailing')
  vae_decode = lambda latent: vae.decode(latent, return_dict=False)[0][0].permute(1, 2, 0).div(2).add(0.5).clamp(0.0, 1.0).mul(255).astype(ms.uint8).numpy()


@timer
def rand_image_set(prompt:str) -> List[npimg]:
  init_model()
  kwargs = deepcopy(SD_PIPE_CALL_KWARGS)
  kwargs['prompt'] = prompt

  latents = []
  def peep_callback(pipe, step, timestep, callback_kwargs):
    nonlocal latents
    if step in SD_PEEP_STEPS:
      latents.append(callback_kwargs['latents'])
    return callback_kwargs

  pipe(**kwargs, callback_on_step_end=peep_callback)
  imgs = [vae_decode(latent) for latent in latents]
  latents.clear()
  return imgs


if __name__ == '__main__':
  from PIL import Image
  from time import time

  for _ in range(10):
    ts_start = time()
    imgs = rand_image_set('a cute cat holding a sign saying hello world')
    ts_end = time()
    print(f'>> time cost: {ts_end - ts_start:.3f}s')  # ~6.5s

  for i, im in enumerate(imgs):
    Image.fromarray(im).save(f'{i}.jpg')
