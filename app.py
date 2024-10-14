#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/08 

import gradio as gr

from game import *

HELP_INFO = '''
### 我画你猜 (via Stable-Diffusion)

⚪ 游戏规则
- 点击开始游戏按钮即可新开一轮游戏，自动生成的 **游戏唯一id标识** 可用于（未结束的）游戏状态恢复
  - 输入 **玩家昵称** 可用于历史成绩记录并加入排行榜，否则将保持游客身份
- 在每游戏轮中，玩家有 **4** 次机会猜测给出的图片内容所对应的文本描述（中文作答，严格匹配）
  - 猜对：游戏结束并累计积分，获得积分随轮次数递减: **10/5/3/1** 分
  - 猜错：系统评定给出文本相似度评分，并切换下一张图
- 难度设计 & 提示
  - 题面图像包含的噪声随着轮次数而降低，并会给出正确答案的字数
  - 玩家可以给出的猜测数随轮次数而降低: **3/2/1/1** 个
- 当前游戏未完成而直接开始新游戏时，会视为放弃游戏
  - 每连续放弃超过 **3** 次，玩家账号将会被惩罚性地停用 **5 min**

⚪ 资源链接
- github: https://github.com/Kahsolt/Draw-Something-xspace
- online demo: https://modelers.cn/spaces/kahsolt/Draw-Something
'''.strip()


with gr.Blocks() as app:
  # Tab 1: game
  with gr.Tab('游戏'):
    with gr.Row():
      with gr.Column():
        tx_username = gr.Textbox(label='玩家昵称', placeholder='填入玩家昵称，开始一局新游戏', max_lines=1)
      with gr.Column():
        tx_gid = gr.Textbox(label='游戏唯一id标识 (开局自动生成)', placeholder='填入游戏唯一id标识，恢复游戏会话进度', max_lines=1)

    with gr.Row():
      with gr.Column():
        btn_game_start = gr.Button('开始游戏🎮', variant='primary')
      with gr.Column():
        btn_game_restore = gr.Button('恢复游戏↺')

    with gr.Row():
      with gr.Column():
        img_sd = gr.Image(label='猜猜我是什么？🤔', width=512, height=512)    # NOTE: size only for display

      with gr.Column():
        tx_info_round = gr.HTML()    # 猜?个字 (第?轮)

        tx_guess_list = []
        GAME_GUESS_HINT = ['我寻思这是...'] + ['也可能是...'] * (GAME_GUESS_MAX_COUNT - 1)
        for i in range(GAME_GUESS_MAX_COUNT):
          tx_guess = gr.Textbox(label=f'猜测-{i+1}', placeholder=GAME_GUESS_HINT[i], max_lines=1)
          tx_guess_list.append(tx_guess)

        btn_game_guess = gr.Button('就决定是你们了！🚀', variant='primary')

    with gr.Row():
      tx_info = gr.Textbox(label='评定结果💭', max_lines=1)

  btn_game_start  .click(game_create,  inputs=[tx_username],            outputs=[tx_username, tx_gid, tx_info, tx_info_round, img_sd, *tx_guess_list])
  btn_game_restore.click(game_restore, inputs=[tx_gid],                 outputs=[tx_username,         tx_info, tx_info_round, img_sd, *tx_guess_list])
  btn_game_guess  .click(game_guess,   inputs=[tx_gid, *tx_guess_list], outputs=[             tx_gid, tx_info, tx_info_round, img_sd, *tx_guess_list])

  # Tab 2: ranklist
  with gr.Tab('排行榜'):
    btn_refresh = gr.Button('刷新🔄', variant='primary')

    ls_rank = gr.List(
      value=make_ranklist,
      headers=['玩家昵称', '累计得分', '平均得分', '上次游玩时间'],
      col_count=(4, 'fixed'),
    )

    btn_refresh.click(make_ranklist, outputs=ls_rank)

  # Tab 3: help info
  with gr.Tab('说明'):
    gr.Markdown(HELP_INFO)


app.launch(max_threads=30)
