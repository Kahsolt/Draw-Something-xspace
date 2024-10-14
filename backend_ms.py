#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/12 

# 扩撒模型引擎 Mindspore 后端

import mindspore as ms
ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=0)

from utils import *


def rand_image_set(prompt:str) -> List[npimg]:
  pass
