#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/11 

# 云平台 modelers 环境中预装了 mindformers (对标 transformers，主要支持 LLM 模型)，可以测读硬件性能
# https://github.com/mindspore-lab/mindformers
# https://github.com/mindspore-lab/mindformers/blob/master/docs/model_support_list.md

'''
| model | speed (token/s) |
| :-: | :-: |
| gpt2      | 84 |
| gpt2_xl   | 17 |
| glm2_6b   | 22 |
| llama_7b  |  5 |
| llama2_7b | 66 |
'''

import os
os.environ['GLOG_v'] = '2'
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

import mindspore as ms
ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=0)
from mindformers.pipeline import pipeline


AVAILABLE_MODELS = [
  'baichuan_7b',
  'baichuan2_7b',
  'baichuan2_13b',
  'bloom_560m',
  'bloom_7.1b',
  'bloom_65b',
  'bloom_176b',
  'codegeex2_6b',
  'codellama_34b',
  'deepseek_33b',
  'glm_6b',
  'glm_6b_chat',
  'glm_6b_lora',
  'glm_6b_lora_chat',
  'glm2_6b',
  'glm2_6b_lora',
  'glm3_6b',
  'gpt2',
  'gpt2_lora',
  'gpt2_xl',
  'gpt2_xl_lora',
  'gpt2_13b',
  'llama_7b',
  'llama_7b_lora',
  'llama_13b',
  'llama_65b',
  'llama2_7b',
  'llama2_13b',
  'llama2_70b',
  'internlm_7b',
  'internlm_7b_lora',
  'pangualpha_2_6b',
  'pangualpha_13b',
  'skywork_13b',
  'ziya_13b',
]


runner = pipeline(task='text_generation', model='llama2_7b', max_length=256)
go = lambda prompt: print(runner(prompt, do_sample=False)[0]['text_generation_text'][0])
go("What's your name?")
go("Tell me a famous greek fable.")
go("What is quantum computing?")
go("How far can generative AI models go?")


try:
  while True:
    s = input('>> Input: ').strip()
    if not s: continue
    go(s)
except KeyboardInterrupt:
  pass
