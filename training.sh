
# ------------------通用参数----------------------
# 微调方式，可选值为：lora, pissa，lisa
FINETUNE_MODE="pissa"
# 训练的RWKV模型版本，可选值为：v5, v6
MODEL_VERSION="v6"
# 量化方式，可选值为：none, 4bit, nf4, fp4
QUANT="none"
# 模型路径
MODEL_PATH=model/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth
# 数据路径
DATA_PATH=data/trainx
# 输出路径
OUTPUT_PATH=output
# 训练的回合数
EPOCH_COUNT=20
# 回合步数
EPOCH_STEPS=2128
# 上下文长度
CTX_LEN=4096
# 精度，可选值为：fp32, bf16, fp16
PRECISION=bf16
# 初始学习率
LR_INIT=5e-5
# 最终学习率
LR_FINAL=5e-5
# 显卡数量
GPU_COUNT=1
# 微批次大小
MICRO_BSZ=1
# 模型保存间隔
EPOCH_SAVE=1
# 前缀网络预处理
PRE_FFN=1
# 梯度累计
MINI_BSZ=8
# 优化策略, 可选值为：deepspeed_stage_1, deepspeed_stage_2, deepspeed_stage_3
DEEPSPEED_STRATEGY=deepspeed_stage_2
# 梯度复制
GRAD_CP=1
# 数据集获取，可选值为：get，pad，only
DATASET_GET="only"
# ------------------不常用训练参数----------------------
# 开始训练的回合，可以用来恢复训练
EPOCH_BEGIN=0
# 词表大小
VOCAB_SIZE=65536
# 嵌入维度
EMBD_SIZE=2560
# 嵌入层
N_LAYER=32
# Head QK
HEAD_QK=0
# Bata1
BETA1=0.9
# Bata2
BETA2=0.999
# 预热步数
WARMUP_STEPS=0
# ADAM epsilon
ADAM_EPS=1e-8

# ------------------Lora和Pissa设置参数----------------------
# lora_parts
lora_parts=att,ffn,time,ln
# LORA模型路径，代表从哪个LORA模型开始微调
lora_load="rwkv-0"
# LORA模型的r值
lora_r=96
# LORA模型的alpha值
lora_alpha=192
# LORA模型的dropout值 
lora_dropout=0.01
# pissa的快速奇异值分解的迭代次数，迭代次数越高损失越低，但是速度越慢
svd_niter=96

# ------------------lisa设置参数----------------------
# LISA模型的r值，代表采样的层数
lisa_r=2
# LISA模型的k值，代表LISA采样的频率
lisa_k=100



















































# ---------------------源代码---------------------
case "$MODEL_VERSION" in
"v6")
    echo "-------------RWKV6微调模式-------------"
    V6_TRAIN="--my_testing x060"
    ;;
"v5")
    echo "-------------RWKV5微调模式-------------"
    V6_TRAIN=""
    ;;
*)
    echo "!!!!!!!!!!!!!不支持的模型版本$MODEL_VERSION，仅支持v5, v6，另外v4已经过时，不建议使用!!!!!!!!!!!!!"
    exit 1
    ;;
esac

# 如果是lisa模式，则不允许量化微调
if [ "$QUANT" != "none" ]; then
    if [ "$FINETUNE_MODE" = "lisa" ]; then
        echo "!!!!!!!!!!!!!LISA微调不支持量化!!!!!!!!!!!!!"
        exit 1
    fi
    case "$QUANT" in
    "4bit"|"nf4"|"fp4")
        echo "-------------使用$QUANT精度量化微调-------------"
        ;;
    *)
        echo "!!!!!!!!!!!!!不支持的量化精度参数$QUANT，仅支持4bit, nf4, fp4!!!!!!!!!!!!!"
        exit 1
        ;;
    esac
fi

case "$DATASET_GET" in
"pad"|"get"|"only")
    echo "-------------使用$DATASET_GET模式读取数据-------------"
    ;;
*)
    echo "!!!!!!!!!!!!!不支持的数据集获取参数$DATASET_GET，仅支持pad, get, only!!!!!!!!!!!!!"
    exit 1
    ;;
esac


if [ "$FINETUNE_MODE" = "lora" ]; then
   python3 train.py --load_model $MODEL_PATH \
    --proj_dir $OUTPUT_PATH --data_file $DATA_PATH \
    --data_type binidx --vocab_size $VOCAB_SIZE \
    --ctx_len $CTX_LEN --epoch_steps $EPOCH_STEPS --epoch_count $EPOCH_COUNT --epoch_begin $EPOCH_BEGIN --epoch_save $EPOCH_SAVE --micro_bsz $MICRO_BSZ \
    --n_layer $N_LAYER --n_embd $EMBD_SIZE \
    --pre_ffn $PRE_FFN --head_qk $HEAD_QK --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps $WARMUP_STEPS --beta1 $BETA1 --beta2 $BETA2 --adam_eps 1e-8 \
    --accelerator gpu --devices $GPU_COUNT --precision $PRECISION --strategy $DEEPSPEED_STRATEGY --grad_cp $GRAD_CP \
    --accumulate_grad_batches $MINI_BSZ --dataset_get $DATASET_GET\
    --lora_load $lora_load --lora --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout $lora_dropout --lora_parts=$lora_parts $V6_TRAIN
else if [ "$FINETUNE_MODE" = "lisa" ]; then
   python3 train.py --load_model $MODEL_PATH \
    --proj_dir $OUTPUT_PATH --data_file $DATA_PATH \
    --data_type binidx --vocab_size $VOCAB_SIZE \
    --ctx_len $CTX_LEN --epoch_steps $EPOCH_STEPS --epoch_count $EPOCH_COUNT --epoch_begin $EPOCH_BEGIN --epoch_save $EPOCH_SAVE --micro_bsz $MICRO_BSZ \
    --n_layer $N_LAYER --n_embd $EMBD_SIZE \
    --pre_ffn $PRE_FFN --head_qk $HEAD_QK --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps $WARMUP_STEPS --beta1 $BETA1 --beta2 $BETA2 --adam_eps $ADAM_EPS \
    --accelerator gpu --devices $GPU_COUNT --precision $PRECISION --strategy $DEEPSPEED_STRATEGY --grad_cp $GRAD_CP \
    --accumulate_grad_batches $MINI_BSZ --dataset_get $DATASET_GET\
    --LISA --lisa_r $lisa_r --lisa_k $lisa_k $V6_TRAIN
else if [ "$FINETUNE_MODE" = "pissa" ]; then
   python3 train.py --load_model $MODEL_PATH \
    --proj_dir $OUTPUT_PATH --data_file $DATA_PATH \
    --data_type binidx --vocab_size $VOCAB_SIZE \
    --ctx_len $CTX_LEN --epoch_steps $EPOCH_STEPS --epoch_count $EPOCH_COUNT --epoch_begin $EPOCH_BEGIN --epoch_save $EPOCH_SAVE --micro_bsz $MICRO_BSZ \
    --n_layer $N_LAYER --n_embd $EMBD_SIZE \
    --pre_ffn $PRE_FFN --head_qk $HEAD_QK --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps $WARMUP_STEPS --beta1 $BETA1 --beta2 $BETA2 --adam_eps $ADAM_EPS \
    --accelerator gpu --devices $GPU_COUNT --precision $PRECISION --strategy $DEEPSPEED_STRATEGY --grad_cp $GRAD_CP \
    --accumulate_grad_batches $MINI_BSZ --dataset_get $DATASET_GET \
    --lora_load $lora_load --lora --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout $lora_dropout --lora_parts=$lora_parts \
    --PISSA --svd_niter $svd_niter $V6_TRAIN
else
    echo "!!!!!!!!!!!!!不支持的微调模式$FINETUNE_MODE，仅支持lora, pissa, lisa!!!!!!!!!!!!!"
    exit 1
fi
fi
fi