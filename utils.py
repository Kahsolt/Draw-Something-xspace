#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/12 

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

import os
import sys
import json
import random
from uuid import uuid4
from hashlib import md5
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Set, Dict, Callable, Any

from numpy import ndarray

npimg = ndarray

BASE_PATH = Path(__file__).parent
ASSET_PATH = BASE_PATH / 'assets'
RUN_PATH = BASE_PATH / 'run' ; RUN_PATH.mkdir(exist_ok=True)
RECORD_FILE = RUN_PATH / 'record.json'
WORDS_FILE = ASSET_PATH / 'words-revised.txt'


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

words: List[Tuple[str, str]] = None  # (cn, en)

def rand_words() -> Tuple[str, str]:
  global words
  if words is None:
    with open(WORDS_FILE, 'r', encoding='utf-8') as fh:
      lines = [ln for ln in fh.read().strip().split('\n')]
    words = []
    for line in lines:
      cp = line.find(' ')
      words.append((line[:cp], line[cp+1:].lower()))
  return random.choice(words)

rand_words()  # warm up

# https://www.bilibili.com/read/cv26963160/
PROMPT_TEMPLATE = [
  'professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting',
  'analog film photo {prompt} . faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage',
  'anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed',
  'cinematic film still {prompt} . shallow depth of field, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy',
  'comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed',
  'play-doh style {prompt} . sculpture, clay art, centered composition, Claymation',
  'concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed',
  'breathtaking {prompt} . award-winning, professional, highly detailed',
  'ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy',
  'isometric style {prompt} . vibrant, beautiful, crisp, detailed, ultra detailed, intricate',
  'line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics',
  'low-poly style {prompt} . low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition',
  'neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional',
  'origami style {prompt} . paper art, pleated paper, folded, origami art, pleats, cut and fold, centered composition',
  'cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed',
  'pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics',
  'texture {prompt} top down close-up',
  'Advertising poster style {prompt} . Professional, modern, product-focused, commercial, eye-catching, highly detailed',
  'Automotive advertisement style {prompt} . Sleek, dynamic, professional, commercial, vehicle-focused, high-resolution, highly detailed',
  'Corporate branding style {prompt} . Professional, clean, modern, sleek, minimalist, business-oriented, highly detailed',
  'Fashion editorial style {prompt} . High fashion, trendy, stylish, editorial, magazine style, professional, highly detailed',
  'Food photography style {prompt} . Appetizing, professional, culinary, high-resolution, commercial, highly detailed',
  'Luxury product style {prompt} . Elegant, sophisticated, high-end, luxurious, professional, highly detailed',
  'Real estate photography style {prompt} . Professional, inviting, well-lit, high-resolution, property-focused, commercial, highly detailed',
  'Retail packaging style {prompt} . Vibrant, enticing, commercial, product-focused, eye-catching, professional, highly detailed',
  'abstract style {prompt} . non-representational, colors and shapes, expression of feelings, imaginative, highly detailed',
  'abstract expressionist painting {prompt} . energetic brushwork, bold colors, abstract forms, expressive, emotional',
  'Art Deco style {prompt} . geometric shapes, bold colors, luxurious, elegant, decorative, symmetrical, ornate, detailed',
  'Art Nouveau style {prompt} . elegant, decorative, curvilinear forms, nature-inspired, ornate, detailed',
  'constructivist style {prompt} . geometric shapes, bold colors, dynamic composition, propaganda art style',
  'cubist artwork {prompt} . geometric shapes, abstract, innovative, revolutionary',
  'expressionist {prompt} . raw, emotional, dynamic, distortion for emotional effect, vibrant, use of unusual colors, detailed',
  'graffiti style {prompt} . street art, vibrant, urban, detailed, tag, mural',
  'hyperrealistic art {prompt} . extremely high-resolution details, photographic, realism pushed to extreme, fine texture, incredibly lifelike',
  'impressionist painting {prompt} . loose brushwork, vibrant color, light and shadow play, captures feeling over form',
  'pointillism style {prompt} . composed entirely of small, distinct dots of color, vibrant, highly detailed',
  'Pop Art style {prompt} . bright colors, bold outlines, popular culture themes, ironic or kitsch',
  'psychedelic style {prompt} . vibrant colors, swirling patterns, abstract forms, surreal, trippy',
  'Renaissance style {prompt} . realistic, perspective, light and shadow, religious or mythological themes, highly detailed',
  'steampunk style {prompt} . antique, mechanical, brass and copper tones, gears, intricate, detailed',
  'surrealist art {prompt} . dreamlike, mysterious, provocative, symbolic, intricate, detailed',
  'typographic art {prompt} . stylized, intricate, detailed, artistic, text-based',
  'watercolor painting {prompt} . vibrant, beautiful, painterly, detailed, textural, artistic',
  'biomechanical style {prompt} . blend of organic and mechanical elements, futuristic, cybernetic, detailed, intricate',
  'biomechanical cyberpunk {prompt} . cybernetics, human-machine fusion, dystopian, organic meets artificial, dark, intricate, highly detailed',
  'cybernetic style {prompt} . futuristic, technological, cybernetic enhancements, robotics, artificial intelligence themes',
  'cybernetic robot {prompt} . android, AI, machine, metal, wires, tech, futuristic, highly detailed',
  'cyberpunk cityscape {prompt} . neon lights, dark alleys, skyscrapers, futuristic, vibrant colors, high contrast, highly detailed',
  'futuristic style {prompt} . sleek, modern, ultramodern, high tech, detailed',
  'retro cyberpunk {prompt} . 80\'s inspired, synthwave, neon, vibrant, detailed, retro futurism',
  'retro-futuristic {prompt} . vintage sci-fi, 50s and 60s style, atomic age, vibrant, highly detailed',
  'sci-fi style {prompt} . futuristic, technological, alien worlds, space themes, advanced civilizations',
  'vaporwave style {prompt} . retro aesthetic, cyberpunk, vibrant, neon colors, vintage 80s and 90s style, highly detailed',
  'Bubble Bobble style {prompt} . 8-bit, cute, pixelated, fantasy, vibrant, reminiscent of Bubble Bobble game',
  'cyberpunk game style {prompt} . neon, dystopian, futuristic, digital, vibrant, detailed, high contrast, reminiscent of cyberpunk genre video games',
  'fighting game style {prompt} . dynamic, vibrant, action-packed, detailed character design, reminiscent of fighting video games',
  'GTA-style artwork {prompt} . satirical, exaggerated, pop art style, vibrant colors, iconic characters, action-packed',
  'Super Mario style {prompt} . vibrant, cute, cartoony, fantasy, playful, reminiscent of Super Mario series',
  'Minecraft style {prompt} . blocky, pixelated, vibrant colors, recognizable characters and objects, game assets',
  'Pokémon style {prompt} . vibrant, cute, anime, fantasy, reminiscent of Pokémon series',
  'retro arcade style {prompt} . 8-bit, pixelated, vibrant, classic video game, old school gaming, reminiscent of 80s and 90s arcade games',
  'retro game art {prompt} . 16-bit, vibrant colors, pixelated, nostalgic, charming, fun',
  'role-playing game (RPG) style fantasy {prompt} . detailed, vibrant, immersive, reminiscent of high fantasy RPG games',
  'strategy game style {prompt} . overhead view, detailed map, units, reminiscent of real-time strategy video games',
  'Street Fighter style {prompt} . vibrant, dynamic, arcade, 2D fighting game, highly detailed, reminiscent of Street Fighter series',
  'Legend of Zelda style {prompt} . vibrant, fantasy, detailed, epic, heroic, reminiscent of The Legend of Zelda series',
  'architectural style {prompt} . clean lines, geometric shapes, minimalist, modern, architectural drawing, highly detailed',
  'disco-themed {prompt} . vibrant, groovy, retro 70s style, shiny disco balls, neon lights, dance floor, highly detailed',
  'dreamscape {prompt} . surreal, ethereal, dreamy, mysterious, fantasy, highly detailed',
  'dystopian style {prompt} . bleak, post-apocalyptic, somber, dramatic, highly detailed',
  'fairy tale {prompt} . magical, fantastical, enchanting, storybook style, highly detailed',
  'gothic style {prompt} . dark, mysterious, haunting, dramatic, ornate, detailed',
  'grunge style {prompt} . textured, distressed, vintage, edgy, punk rock vibe, dirty, noisy',
  'horror-themed {prompt} . eerie, unsettling, dark, spooky, suspenseful, grim, highly detailed',
  'kawaii style {prompt} . cute, adorable, brightly colored, cheerful, anime influence, highly detailed',
  'lovecraftian horror {prompt} . eldritch, cosmic horror, unknown, mysterious, surreal, highly detailed',
  'macabre style {prompt} . dark, gothic, grim, haunting, highly detailed',
  'manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style',
  'metropolis-themed {prompt} . urban, cityscape, skyscrapers, modern, futuristic, highly detailed',
  'minimalist style {prompt} . simple, clean, uncluttered, modern, elegant',
  'monochrome {prompt} . black and white, contrast, tone, texture, detailed',
  'nautical-themed {prompt} . sea, ocean, ships, maritime, beach, marine life, highly detailed',
  'space-themed {prompt} . cosmic, celestial, stars, galaxies, nebulas, planets, science fiction, highly detailed',
  'stained glass style {prompt} . vibrant, beautiful, translucent, intricate, detailed',
  'techwear fashion {prompt} . futuristic, cyberpunk, urban, tactical, sleek, dark, highly detailed',
  'tribal style {prompt} . indigenous, ethnic, traditional patterns, bold, natural colors, highly detailed',
  'zentangle {prompt} . intricate, abstract, monochrome, patterns, meditative, highly detailed',
  'collage style {prompt} . mixed media, layered, textural, detailed, artistic',
  'flat papercut style {prompt} . silhouette, clean cuts, paper, sharp edges, minimalist, color block',
  'kirigami representation of {prompt} . 3D, paper folding, paper cutting, Japanese, intricate, symmetrical, precision, clean lines',
  'paper mache representation of {prompt} . 3D, sculptural, textured, handmade, vibrant, fun',
  'paper quilling art of {prompt} . intricate, delicate, curling, rolling, shaping, coiling, loops, 3D, dimensional, ornamental',
  'papercut collage of {prompt} . mixed media, textured paper, overlapping, asymmetrical, abstract, vibrant',
  '3D papercut shadow box of {prompt} . layered, dimensional, depth, silhouette, shadow, papercut, handmade, high contrast',
  'stacked papercut art of {prompt} . 3D, layered, dimensional, depth, precision cut, stacked layers, papercut, high contrast',
  'thick layered papercut art of {prompt} . deep 3D, volumetric, dimensional, depth, thick paper, high stack, heavy texture, tangible layers',
  'alien-themed {prompt} . extraterrestrial, cosmic, otherworldly, mysterious, sci-fi, highly detailed',
  'film noir style {prompt} . monochrome, high contrast, dramatic shadows, 1940s style, mysterious, cinematic',
  'HDR photo of {prompt} . High dynamic range, vivid, rich details, clear shadows and highlights, realistic, intense, enhanced contrast, highly detailed',
  'long exposure photo of {prompt} . Blurred motion, streaks of light, surreal, dreamy, ghosting effect, highly detailed',
  'neon noir {prompt} . cyberpunk, dark, rainy streets, neon signs, high contrast, low light, vibrant, highly detailed',
  'silhouette style {prompt} . high contrast, minimalistic, black and white, stark, dramatic',
  'tilt-shift photo of {prompt} . Selective focus, miniature effect, blurred background, highly detailed, vibrant, perspective control',
]

def rand_prompt(word_en:str) -> str:
  return random.choice(PROMPT_TEMPLATE).format(prompt=word_en)


def load_json(fp:Path, default:Callable=dict) -> Any:
  if not fp.exists():
    return default()
  with open(fp, 'r', encoding='utf-8') as fh:
    return json.load(fh)

def save_json(data:Any, fp:Path):
  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, indent=2, ensure_ascii=False)

''' Game Const '''

# 文本常量
TEXT_HELP_INFO = '''
### 我画你猜 (via Stable-Diffusion)

⚪ 游戏规则
- 点击开始游戏按钮即可新开一轮游戏，自动生成的 **游戏唯一id标识** 可用于（未结束的）游戏状态恢复
  - 输入 **玩家昵称** 可用于历史成绩记录并加入排行榜，否则将保持游客身份
- 在每游戏轮中，玩家有 **4** 次机会猜测给出的图片内容所对应的文本描述（中文作答，模糊匹配即算正确）
  - 猜对：游戏结束并累计积分，获得积分随轮次数递减: **10/5/3/1** 分
  - 猜错：系统评定给出文本相似度评分，并切换下一张图
- 难度设计 & 提示
  - 题面图像包含的噪声随着轮次数而降低，并会给出正确答案的字数
  - 玩家可以给出的猜测数随轮次数而降低: **3/2/2/1** 个
- 当前游戏未完成而直接开始新游戏时，会视为放弃游戏
  - 每连续放弃超过 **3** 次，玩家账号将会被惩罚性地停用 **5 min**

⚪ 资源链接
- github: https://github.com/Kahsolt/Draw-Something-xspace
- online demo: https://modelers.cn/spaces/kahsolt/Draw-Something
'''.strip()
TEXT_BTN_MAIN_START = '开始游戏🎮'
TEXT_BTN_MAIN_GUESS = '就决定是你们了！🚀'

# 多少次猜的机会
GAME_GUESS_ROUND = 4
# 猜对第k张图时的得分
GAME_GUESS_SCORES = [10, 5, 3, 1]
# 猜第k张图可用的猜想数
GAME_GUESS_CHOICES = [3, 2, 2, 1]
GAME_GUESS_CHOCES_MAX = max(GAME_GUESS_CHOICES)
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
SD_PEEP_STEPS = [4, 6, 7, 9]
