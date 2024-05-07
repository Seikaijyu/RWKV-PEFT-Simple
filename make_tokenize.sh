#!/bin/bash

# 参数
IN_FILE="data/$1"
N_EPOCH=$2
CTX_LEN=$3

# 获取输入文件的目录路径
TARGET_DIR=$(dirname $IN_FILE)

# 执行命令
python3 make_tokenize.py $IN_FILE $N_EPOCH $CTX_LEN

# 获取输入文件的基本名称（没有扩展名）
BASE_NAME=$(basename $IN_FILE .jsonl)
# 判断文件是否存在
if [ ! -f "${BASE_NAME}.bin" ]; then
    exit 1
fi
# 移动生成的文件到目标目录
mv ${BASE_NAME}.bin ${TARGET_DIR}
if [ ! -f "${BASE_NAME}.idx" ]; then
    exit 1
fi
mv ${BASE_NAME}.idx ${TARGET_DIR}