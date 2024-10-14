#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/14 

# 制作问题列表
# https://blog.csdn.net/weixin_42453761/article/details/106460524

#import warnings
#warnings.filterwarnings(action='ignore', category=UserWarning)
#warnings.filterwarnings(action='ignore', category=FutureWarning)

from pathlib import Path
import jieba.posseg as pseg
from translate import Translator
from tqdm import tqdm

BASE_PATH = Path(__file__).parent
ASSET_PATH = BASE_PATH / 'assets'

WORDLIST_FILE = ASSET_PATH / 'wordlist.txt'
WORDS_FILE = ASSET_PATH / 'words.txt'

with open(WORDLIST_FILE, 'r', encoding='utf-8') as fh:
  words = fh.read().strip().split('\n')

words = [w for w in words if len(w) <= 5]
words_new = []
tags = set()
T = Translator(from_lang='chinese', to_lang='english')
for word in tqdm(words):
  segs = pseg.lcut(word)
  if len(segs) > 1: continue
  word, tag = segs[0]
  if tag not in ['nr', 'nrfg', 'nrt', 'ns', 'nz', 'a', 'v', 'i', 'j', 'l', 'm']:
    tags.add(tag)
    try:
      it = f'{word} {T.translate(word).lower()}'
      print(f'>> add item: {it}')
      words_new.append(it)
    except:
      print(f'>> cannot translate {word}, ignored')
words = words_new
words = list(set(words))
words.sort(key=lambda s: (len(s), s))
print('>> len(words):', len(words))
print('>> tags collected:', tags)

with open(WORDS_FILE, 'w', encoding='utf-8') as fh:
  fh.write('\n'.join(words))
