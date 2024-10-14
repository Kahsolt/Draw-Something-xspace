#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/12 

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

import os
import sys
import json
from uuid import uuid4
from hashlib import md5
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Set, Dict, Callable, Any

from numpy import ndarray

npimg = ndarray

BASE_PATH = Path(__file__).parent
RUN_PATH = BASE_PATH / 'run' ; RUN_PATH.mkdir(exist_ok=True)
RECORD_FILE = RUN_PATH / 'record.json'


''' Env '''

IS_WIN = sys.platform = 'win32'

def init_env():
  if os.getenv('INIT_FLAG'): return

  os.environ['GLOG_v'] = '2'
  if IS_WIN:    # local debug
    os.system('SET HF_ENDPOINT=https://hf-mirror.com')
  else:
    os.system('export HF_ENDPOINT=https://hf-mirror.com')
    os.system('python -m pip install --upgrade pip')
    os.system('pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple')

  os.environ['INIT_FLAG'] = '1'

init_env()


''' Misc Utils '''

GUETST_USERNAME_PREFIX = 'guest-'

def now_ts() -> int:
  return int(datetime.now().timestamp())

def rand_username() -> str:
  return f'{GUETST_USERNAME_PREFIX}{rand_gid()[:8]}'

def rand_gid() -> str:
  return md5(str(uuid4()).encode()).hexdigest()

def rand_ans() -> str:
  return 'a red apple'

def rand_prompt(ans:str) -> str:
  return ans


def load_json(fp:Path, default:Callable=dict) -> Any:
  if not fp.exists():
    return default()
  with open(fp, 'r', encoding='utf-8') as fh:
    return json.load(fh)

def save_json(data:Any, fp:Path):
  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, indent=2, ensure_ascii=False)


''' Game Const '''

# 每张图可以猜多少个答案
GAME_GUESS_MAX_COUNT = 3
# 猜对第k张图时的得分
GAME_GUESS_SCORES = [10, 5, 3, 1]
# 猜第k张图可用的猜想数
GAME_GUESS_CHOICES = [3, 2, 1, 1]
# 放弃游戏惩罚性禁用
GAME_ABORT_BAN_TOL  = 3
GAME_ABORT_BAN_TEMP = 5 * 60

# SD 绘图参数
SD_PIPE_CALL_KWARGS = {
  'prompt': None,  # place holder
  'height': 1024,
  'width': 1024,
  'num_inference_steps': 10,
  'guidance_scale': 0.0,    # MUST be off
  'output_type': 'latent',
  'return_dict': False,
}
SD_PEEP_STEPS = [3, 5, 7, 9]
