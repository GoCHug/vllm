#!/bin/bash
# kvc practice scripts (run inside gggtest pod)
# 1) kill service
# pkill -9 VLLM*
# pkill -9 python*
# 2) start service (per user cmd.sh lines 6-9), run in background with fresh log:
# cd /a3_inference/itask/workdir/gch02599191/kvc && setsid nohup vllm serve ... > ./llama.log 2>&1 < /dev/null &
