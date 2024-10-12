#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/12 

# 游戏原型验证
# RTX 3060 benchmark stats:
# - time_cost: 0.35 sec/step
# - vram_cost: 8.4G

import math
from time import time
import warnings ; warnings.filterwarnings(action='ignore', category=UserWarning)
import warnings ; warnings.filterwarnings(action='ignore', category=FutureWarning)
from typing import List, Tuple, Union

import torch
from diffusers.pipelines import StableDiffusionXLPipeline
from diffusers.schedulers import (
  DDIMScheduler,
  DDPMScheduler,
  DPMSolverMultistepScheduler,
  EulerAncestralDiscreteScheduler,
  EulerDiscreteScheduler,
  HeunDiscreteScheduler,
  UniPCMultistepScheduler,
)
from diffusers.models import UNet2DConditionModel, AutoencoderTiny
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import matplotlib.pyplot as plt ; plt.ion()

if not 'plt maximize window':
  # https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window
  K = plt.get_backend()
  mng = plt.get_current_fig_manager()
  if K == 'TkAgg':
    # mng.resize(*mng.window.maxsize()) # works on Ubuntu??? >> did NOT working on windows
    mng.window.state('zoomed') # works fine on Windows!
  elif K == 'wxAgg':
    mng.frame.Maximize(True)
  elif K == 'Qt4Agg':
    mng.window.showMaximized()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16

pipe_call_kwargs = {
  #'prompt': 'A cat holding a sign that says hello world',
  'prompt': 'masterpiece, highres, detailed scene, japanese anime CG; egyptian, 1boy, cute child, cat ears, white hair, red eyes, yellow bell, red cloak, barefoot, flying angel',
  'height': 1024,
  'width': 1024,
  'num_inference_steps': 10,
  'guidance_scale': 0.0,    # MUST be off
  'output_type': 'latent',
  'return_dict': False,
}

base_repo = 'stabilityai/stable-diffusion-xl-base-1.0'
vae_repo  = 'madebyollin/taesdxl'
unet_repo = 'ByteDance/SDXL-Lightning'
unet_ckpt = 'sdxl_lightning_8step_unet.safetensors'

unet = UNet2DConditionModel.from_config(base_repo, subfolder='unet').to(dtype)
unet.load_state_dict(load_file(hf_hub_download(unet_repo, unet_ckpt)))
vae = AutoencoderTiny.from_pretrained(vae_repo, torch_dtype=dtype)
pipe = StableDiffusionXLPipeline.from_pretrained(base_repo, vae=vae, unet=unet, torch_dtype=dtype, use_safetensors=True, variant='fp16', add_watermarker=False)
#pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe.to(device)


# ~torchvision.utils.make_grid
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Tuple[int, int] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> torch.Tensor:
    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps: break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid

def vae_decode(latent):
  return (vae.decode(latent, return_dict=False)[0] / 2 + 0.5).clamp_(0.0, 1.0).float().cpu()


PEEP_STEPS = [3, 5, 7, 9]
latents = []
def peep_callback(pipe, step, timestep, callback_kwargs):
  global latents
  if step in PEEP_STEPS:
    latents.append(callback_kwargs['latents'])
  return callback_kwargs


with torch.inference_mode():
  try:
    while True:
      ts_start = time()
      pipe(**pipe_call_kwargs, callback_on_step_end=peep_callback)
      ts_end = time()
      imgs = [vae_decode(latent) for latent in latents]
      print(f'>> Time Cost: {ts_end - ts_start:.3f}s')
      latents.clear()

      if 'show':
        plt.imshow(make_grid(torch.cat(imgs, dim=0)).permute(1, 2, 0))
        plt.tight_layout()
        plt.pause(0.01)
      imgs.clear()
  except KeyboardInterrupt:
    pass
