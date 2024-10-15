#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/12 

# 游戏总控

import gc
from dataclasses import dataclass, astuple
from utils import *

try:
  import mindspore
  from backend_ms import *
  print('>> Backend: Mindspore')
except ImportError:
  from backend_pt import *
  print('>> Backend: Pytorch')


''' Data '''

@dataclass
class State:
  username: str      # 玩家昵称
  ans: str           # 正确答案
  text: str          # 随机谜题文本
  imgs: List[npimg]  # 随机谜题图像
  info: str = ''     # 评测信息 (用于状态恢复)
  guest: bool = True # 游客身份

  @property
  def round(self) -> int:   # zero-based
    return GAME_GUESS_ROUND - len(self.imgs)
  @property
  def img(self) -> npimg:
    return self.imgs[0]
  @property
  def img_final(self) -> npimg:
    return self.imgs[-1]
  @property
  def info_round(self) -> str:
    return f'请根据图片猜 {len(self.ans)} 个字 (第 {1 + self.round} / {GAME_GUESS_ROUND} 轮)'
  @property
  def info_puzzle(self) -> str:
    return f'这幅画所用的提示词为：{self.text}'


@dataclass
class Record:
  count: int =  0   # 游玩次数
  score: int =  0   # 累计得分
  ts:    int = -1   # 最后在线时间

  @property
  def mean_score(self) -> float:
    return (self.score / self.count) if self.count else 0.0
  @property
  def dt(self) -> str:
    return datetime.fromtimestamp(self.ts).strftime(r'%Y-%m-%d')

States  = Dict[str, State]    # gid -> state
Records = Dict[str, Record]   # username -> record

def load_records() -> Records:
  data = load_json(RECORD_FILE)
  records: dict = data.get('records', {})
  return {k: Record(*v) for k, v in records.items()}

def save_records():
  global records
  data = {
    'save_ts': now_ts(),
    'records': {k: astuple(v) for k, v in records.items()},
  }
  save_json(data, RECORD_FILE)

def query_rank(username:str) -> int:
  ranklist = sorted([(v.mean_score, k) for k, v in records.items()], reverse=True)
  namelist = [k for _, k in ranklist]
  return namelist.index(username) + 1

def make_ranklist() -> list:
  reclist = [(k, v.score, round(v.mean_score, 2), v.dt) for k, v in records.items()]
  reclist.sort(key=lambda x: (-x[-2], x[-1]))
  return reclist[:50]

states: States = None
records: Records = None

def init_globals():
  global states, records
  if states     is None: states = {}
  if records    is None: records = load_records()

init_globals()


''' Logic '''

def _make_tx_guess_list(gid:str) -> tuple:
  tx_guess_list = [None for _ in range(GAME_GUESS_CHOCES_MAX)]
  can_use = GAME_GUESS_CHOICES[states[gid].round] if game_exists(gid) else GAME_GUESS_CHOCES_MAX
  for idx, tx_guess in enumerate(tx_guess_list):
    tx_guess_list[idx] = {'__type__': 'update', 'visible': idx < can_use, 'value': tx_guess}
  return tx_guess_list

def game_exists(gid:str) -> bool:
  return gid in states

def game_create(username:str) -> tuple:
  ''' ~return: tx_username, tx_gid, tx_info, tx_info_round, img_sd, *tx_guess_list '''

  username = username.strip()
  is_guest = not username or username.startswith(GUETST_USERNAME_PREFIX)
  realname = username or rand_username()

  # 关闭旧局
  for gid, inst in states.items():
    if inst.username == realname:
      game_destroy(gid)
      break

  # 新开一局
  gid   = rand_gid()
  words = rand_words()
  text  = rand_prompt(words[1])  # en
  imgs  = rand_image_set(text)
  inst  = State(realname, words[0], text, imgs, guest=is_guest)
  if IS_WIN:
    print(f'[gid-{gid}]')
    print('  words:', words)
    print('  text:', text)
  states[gid] = inst

  return realname, gid, inst.info, inst.info_round, inst.img, *_make_tx_guess_list(gid)

def game_destroy(gid:str):
  del states[gid] ; gc.collect()

def game_restore(gid:str) -> tuple:
  ''' ~return: tx_username, tx_info, tx_info_round, img_sd, *tx_guess_list '''

  gid = gid.strip()
  inst = states.get(gid)
  if inst is None: return None, 'Error: 错误的 gid', None, None, *_make_tx_guess_list(gid)

  return inst.username, inst.info, inst.info_round, inst.img, *_make_tx_guess_list(gid)

def game_guess(gid:str, *guess_list:str) -> tuple:
  ''' ~return: tx_gid, tx_info, tx_info_round, img_sd, *tx_guess_list '''

  gid = gid.strip()
  guess_list = [s.strip() for s in guess_list]
  inst = states.get(gid)
  if inst is None: return None, 'Error: 错误的 gid', None, None, *_make_tx_guess_list(gid)

  # 更新注册玩家的最近在线时间戳
  rec = None
  if not inst.guest:
    if inst.username not in records:
      records[inst.username] = Record()
    rec = records[inst.username]
    rec.ts = now_ts()

  def is_answer_match(answer:str, candidates:List[str]) -> bool:
    if answer in candidates:
      return True
    for indv in candidates:
      if not indv: continue
      if answer in indv or indv in answer:
        return True
    return False

  def get_iou(answer:str, guess:str) -> float:
    vset, pset = set(answer), set(guess)
    return len(vset & pset) / len(vset | pset)

  # 猜对了吗猜对了吗猜对了吗？？
  info_score = ''
  valid_guesses = guess_list[:GAME_GUESS_CHOICES[inst.round]]
  if is_answer_match(inst.ans, valid_guesses):
    game_destroy(gid)
    if not inst.guest:
      score = GAME_GUESS_SCORES[inst.round]
      rec.score += score
      rec.count += 1
      save_records()
      rank = query_rank(inst.username)
      info_score = f'玩家【{inst.username}】获得 {score} 分，累计积分：{rec.score}，平均分排名第 {rank}。'
    return '', f'恭喜你猜对啦，答案就是 【{inst.ans}】！{info_score}\n{inst.info_puzzle}', '', inst.img_final, *_make_tx_guess_list(gid)

  if inst.round == GAME_GUESS_ROUND:  # 强制结束
    game_destroy(gid)
    if not inst.guest:
      rec.count += 1
      save_records()
      rank = query_rank(inst.username)
      info_score = f'玩家【{inst.username}】累计积分：{rec.score}，平均分排名第 {rank}。'
    return '', f'好像没猜中哦……正确答案是 【{inst.ans}】，下次好运吧！{info_score}\n{inst.info_puzzle}', '', inst.img_final, *_make_tx_guess_list(gid)
  
  # 继续
  match_scores = {guess: int(100 * get_iou(inst.ans, guess)) for guess in valid_guesses}
  inst.info = f'答案好像不太对哦，匹配程度: {match_scores}'
  inst.imgs.pop(0)      # 切下一张图
  return gid, inst.info, inst.info_round, inst.img, *_make_tx_guess_list(gid)
