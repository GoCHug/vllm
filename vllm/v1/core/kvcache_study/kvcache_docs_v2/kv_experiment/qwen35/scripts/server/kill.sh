#!/bin/bash
# 清理残留进程（暴杀版, 不分先后;
#   显存残留时先精杀 VLLM::EngineCore 再杀 serve —— 见 README §4.6-2）
pkill -9 VLLM*
pkill -9 python*
