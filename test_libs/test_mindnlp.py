#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/11

# 通过 git 安装 mindnlp (对标 transformers，支持 LLM/NLP/CV 模型)，测读库的可用性和硬件性能
# https://github.com/mindspore-lab/mindnlp
# https://mindnlpdocs-hwy.readthedocs.io/zh-cn/latest/

'''
| model | speed (token/s) |
| :-: | :-: |
| llama2_7b | 15 |
'''

import os
os.environ['GLOG_v'] = '2'
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

from time import time
import mindspore as ms
try:
  # fix mindformers issue :(
  import mindnlp.accelerate.utils.imports as IMP
  IMP.is_mindformers_available = lambda: False
except ImportError:
  pass
from mindnlp.transformers import LlamaForCausalLM, LlamaTokenizer


model_path = 'shakechen/Llama-2-7b-hf'

tokenizer = LlamaTokenizer.from_pretrained(model_path, mirror='modelscope')
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained(model_path, mirror='modelscope', ms_dtype=ms.float16)
gen_cfg = model.generation_config
gen_cfg.max_length = 256
gen_cfg.do_sample = False
gen_cfg.pad_token_id=tokenizer.pad_token_id


def go(prompt:str):
  inputs = tokenizer([prompt], return_tensors='ms')
  ts_start = time()
  outputs = model.generate(**inputs, generation_config=gen_cfg)
  ts_end = time()
  n_token = len(outputs[0]) - len(inputs.input_ids[0])
  print(f'>> speed: {n_token / (ts_end - ts_start):.3f} tok/s')
  print(tokenizer.decode(outputs[0]))

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
