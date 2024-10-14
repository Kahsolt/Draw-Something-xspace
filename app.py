#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/08 

import gradio as gr

from game import *


with gr.Blocks() as app:
  # Tab 1: game
  with gr.Tab('游戏'):
    with gr.Row():
      with gr.Column():
        tx_username = gr.Textbox(label='玩家昵称', placeholder='填入玩家昵称，开始一局新游戏', max_lines=1)
      with gr.Column():
        tx_gid = gr.Textbox(label='游戏唯一id标识 (开局自动生成)', placeholder='填入游戏唯一id并按回车键，恢复游戏会话进度', max_lines=1)

    with gr.Row():
      with gr.Column():
        img_sd = gr.Image(label='猜猜我是什么？🤔', width=512, height=512)    # NOTE: size only for display

      with gr.Column():
        tx_info_round = gr.HTML()    # 猜?个字 (第?轮)

        tx_guess_list = []
        GAME_GUESS_HINT = ['我寻思这是...'] + ['也可能是...'] * (GAME_GUESS_CHOCES_MAX - 1)
        for i in range(GAME_GUESS_CHOCES_MAX):
          tx_guess = gr.Textbox(label=f'猜测-{i+1}', placeholder=GAME_GUESS_HINT[i], max_lines=1)
          tx_guess_list.append(tx_guess)

        btn_main = gr.Button(TEXT_BTN_MAIN_START, variant='primary')

    with gr.Row():
      tx_info = gr.Textbox(label='评定结果💭', lines=2, max_lines=2)

  def game_API_dispatcher(username:str, gid:str, *guess_list:str) -> tuple:
    ''' btn_main, tx_username, tx_gid, tx_info, tx_info_round, img_sd, *tx_guess_list '''
    if gid:
      apt_ret = game_guess(gid, *guess_list)
      apt_ret = (username, *apt_ret)
    else:
      apt_ret = game_create(username)
      gid = apt_ret[1]  # eager update
    btn_main = TEXT_BTN_MAIN_GUESS if game_exists(gid) else TEXT_BTN_MAIN_START
    return btn_main, *apt_ret

  def game_wrap_restore(gid:str) -> tuple:
    ''' btn_main, tx_username, tx_info, tx_info_round, img_sd, *tx_guess_list '''
    apt_ret = game_restore(gid)
    btn_main = TEXT_BTN_MAIN_GUESS if game_exists(gid) else TEXT_BTN_MAIN_START
    return btn_main, *apt_ret

  tx_gid.submit (game_wrap_restore,   inputs=[tx_gid],                              outputs=[btn_main, tx_username,         tx_info, tx_info_round, img_sd, *tx_guess_list])
  btn_main.click(game_API_dispatcher, inputs=[tx_username, tx_gid, *tx_guess_list], outputs=[btn_main, tx_username, tx_gid, tx_info, tx_info_round, img_sd, *tx_guess_list])

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
    gr.Markdown(TEXT_HELP_INFO)


app.launch(max_threads=30)
