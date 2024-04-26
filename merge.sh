# 模型路径
MODEL_PATH=RWKV-x060-World-3B-v2.1-Claude-nsfw.pth
# 训练回合数，由命令行传入
PISSA_EPOCH=$1

# 量化精度，可用参数为：none,4bit,nf4,fp4
QUANT="none"
# 微调模式，可用参数为：lora,pissa
TRAIN_TYPE="pissa"
# LORA_ALPHA参数，仅用于lora模式，pissa模式微调时不需要对应
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
"pissa")
    echo "-------------使用$TRAIN_TYPE模式合并-------------"
    ;;
*)
    echo "!!!!!!!!!!!!!不支持的训练模式参数$TRAIN_TYPE，仅支持lora, pissa!!!!!!!!!!!!!"
    exit 1
    ;;
esac

python3 merge.py \
    --quant $QUANT \
    --lora_alpha $LORA_ALPHA \
    --type $TRAIN_TYPE \
    --base_model model/$MODEL_PATH \
    --lora_init output/init_lora.pth \
    --lora_checkpoint output/rwkv-$PISSA_EPOCH.pth \
    --output merge/$FILE_NAME-$OUT_TYPE-$PISSA_EPOCH.pth