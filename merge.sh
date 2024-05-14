# 模型路径
MODEL_PATH=RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth
# 训练回合数，由命令行传入，具体微调回复数可以查看output目录下的文件，例如：rwkv-7.pth表示微调7回合后的模型
# 使用方式：./merge.sh {微调的回合数量}
PISSA_EPOCH=$1
# 训练使用的量化精度，可用参数为：none,4bit,nf4,fp4
QUANT="none"
# 训练使用的微调模式，可用参数为：lora, pissa, state
TRAIN_TYPE="pissa"
# LORA_ALPHA参数，仅lora微调时需要设置，其它模型模式微调时不需要对应
LORA_ALPHA=256















































# ---------------------源代码---------------------
FILE_NAME=$(basename $MODEL_PATH .pth)
OUT_TYPE=$(echo "$TRAIN_TYPE" | tr '[:lower:]' '[:upper:]')
if [ ! -d "merge" ]; then
  mkdir merge
fi

case "$QUANT" in
"4bit"|"nf4"|"fp4")
    echo "-------------使用$QUANT精度量化的lora合并-------------"
    ;;
"none")
    echo "-------------不使用量化的lora合并-------------"
    ;;
*)
    echo "!!!!!!!!!!!!!不支持的量化精度参数$QUANT的lora合并，仅支持none,4bit, nf4, fp4!!!!!!!!!!!!!"
    exit 1
    ;;
esac

case "$TRAIN_TYPE" in
"pissa"|"lora"|"state")
    echo "-------------使用$TRAIN_TYPE模式合并-------------"
    ;;
*)
    echo "!!!!!!!!!!!!!不支持的训练模式参数$TRAIN_TYPE，仅支持lora, pissa!!!!!!!!!!!!!"
    exit 1
    ;;
esac

if [ "$TRAIN_TYPE" = "state" ]; then
    python3 merge/merge_state.py \
        --base_model model/$MODEL_PATH \
        --state_checkpoint output/rwkv-$PISSA_EPOCH.pth \
        --output merge/$FILE_NAME-$OUT_TYPE-$PISSA_EPOCH.pth
else
    python3 merge/merge.py \
        --quant $QUANT \
        --lora_alpha $LORA_ALPHA \
        --type $TRAIN_TYPE \
        --base_model model/$MODEL_PATH \
        --lora_init output/init_lora.pth \
        --lora_checkpoint output/rwkv-$PISSA_EPOCH.pth \
        --output merge/$FILE_NAME-$OUT_TYPE-$PISSA_EPOCH.pth
fi