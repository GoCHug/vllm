#!/bin/bash
# tp:异常数据
# gp:标杆数据

msprobe compare -tp ./2048_dump/step1/rank0/dump.json -gp ./1024_dump/step1/rank0/dump.json -o ./output
