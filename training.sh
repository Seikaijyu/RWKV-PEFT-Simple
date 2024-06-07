# 使用方式：调整参数后使用`./training.sh`运行
# ------------------通用参数----------------------
# 微调方式，可选值为：lora, pissa，lisa
# lora: 使用标准lora微调，但是lora微调速度较慢，效果一般，更推荐pissa
# pissa: lora的改进版本，使用快速奇异值分解，收敛速度更快，效果更好，推荐使用
# lisa: lisa使用类似全量微调的方式，冻结多层，每次只选择几层进行微调，微调的每个epoch直接输出模型，但是目前还有一些小问题，不推荐使用
FINETUNE_MODE="pissa"
# 微调附加类型，可选值为：none, state, infctx
# state: 最新的实验性质微调，微调init state，微调速度更快，占用显存更低，但是暂时不够稳定，state微调暂时不作为微调附加类型
# 开启后会覆盖微调方式设置，后续将允许作为附加类型使用，目前暂时未支持
# infctx: infctx使用时间换内存进行训练，当显存不足但语料过长时建议开启
# 使用infctx时必须将DATALOAD修改为pad模式，否则会导致nan
# 开启后使用更长上下文微调时，不会导致显存使用量增加，开启时最好同时开启FLA以启用triton算子，因为cuda算子的梯度有点问题
TRAIN_TYPE="none"
# 训练的RWKV模型版本，可选值为：v5, v6
MODEL_VERSION="v6"
# 量化方式，可选值为：none, 4bit, nf4, fp4
# 一般推荐使用nf4，分布更均匀
QUANT="none"
# 微调embedding层，可选值为：0（关闭）, 1（开启）
# 开启时不会冻结embedding层和head层，这两层和lora_r没有任何关系，调高也不会影响这两层的微调
# embedding层是模型理解输入数据的基础，而head层则直接影响模型的输出
# 但是开启后会需要更多显存，同时会增加训练时间（仅在lora和pissa微调下有效）
EMB_FINETUNE=0
# 启用triton算子，可选值为：0（关闭）, 1（开启）
# MICRO_BSZ越小越推荐开启，训练速度更快
FLA=0
# 模型路径
# 对应的是在model文件夹下需要微调的模型的文件名
MODEL_PATH=RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth
# 数据路径
# 对应的是在data文件夹下需要微调的使用数据的文件名
DATA_PATH=train
# 训练的回合数，达到回合数后会停止训练
# 仅在数据读取模式为pad和only时生效
EPOCH_COUNT=20
# 回合步数
# 应该根据训练数据的条数和微批次大小调整，公式为：数据集条数/微批次大小=回合步数
EPOCH_STEPS=1344
# 上下文长度
# 使用./make_tokenize.sh {数据集名称}.jsonl {训练回合数}脚本进行数据分词时能得到如：### max_length = 208 这样的输出
# 其中208就是数据中最长的数据的长度，在pad模式和only模式中，应该填入此数值以保证数据能够完全被训练
# 如果数据过长无法训练，建议降低上下文长度并使用get模式读取数据，可以节省资源
# 使用./make_tokenize.sh {数据集名称}.jsonl 1 进行数据分词即可
CTX_LEN=6604
# 开启微调附加项的infctx参数后启用的设置，此设置用于确定在infctx中单次训练的上下文长度，此参数越高，消耗的显存越多
# 相当于不开启infctx时的CTX_LEN参数，一般建议能开多大开多大（仅在infctx启用时有效）
CHUNK_CTX=512
# 精度，可选值为：fp32, bf16, fp16，通常建议使用bf16，节省显存同时保证了训练精度
PRECISION=bf16
# 如果使用state模式微调，lr最好调高，建议使用动态学习率，从1到0.01，使用其它模式建议5e-5到1e-4之间，优先选择5e-5
# 初始学习率
LR_INIT=5e-5
# 最终学习率
# 通常建议和初始学习率一致，除非需要动态学习率
# 动态学习率的下降速度与你设定的训练回合数和训练步数有关
LR_FINAL=5e-5
# 预热步数
# 此配置与学习率相关，如果使用不同的初始学习率和最终学习率，建议调整此配置
# 此配置规定了在训练开始阶段使用以下公式的学习率作为预热初始学习率：预热初始学习率=初始学习率/5 
# 然后在训练过程中基于设定的预热步数训练到指定的实际步数，计算公式为：实际步数=梯度累计x训练步数
# 此时lr将为初始学习率，然后再从初始学习率开始逐渐迭代到最终学习率
# 开启时建议严格控制训练步数和回合数，以达到最优效果，模型尺寸越大预热步数应当越多
WARMUP_STEPS=0
# 显卡数量
GPU_COUNT=1
# 微批次大小，此配置项越大，显存占用越大，但是训练速度越快
# 此配置项非1时应该跟随数据集条数调整，计算公式为：数据集条数/微批次大小=回合步数
# 例如：数据集条数为10000，微批次大小为10，回合步数应该设置为1000
# 微批次大小并不是越大越好，更大的微批次大小会导致学不到东西，建议根据数据集数量调整
# 一般来说，数据集条数小于5k条时候，微批次大小建议最大为8，但是和梯度累计不同，微批次大小可以适当提高，视需求而定
MICRO_BSZ=1
# 模型保存间隔，每隔多少回合保存一次模型
EPOCH_SAVE=1
# 梯度累计，如果显存不够无法调整微批次大小
# 建议基于以下公式设置参数：微批次大小x梯度累计 <= 8
# 梯度累计并不是越大越好，更大的梯度累计会导致学不到东西，一般来说，数据集条数小于5k条时候，梯度累计建议最大为8
MINI_BSZ=8
# 优化策略, 可选值为：deepspeed_stage_1, deepspeed_stage_2, deepspeed_stage_3
# 建议使用deepspeed_stage_2节省显存的同时也能保证微调速度
# deepspeed_stage_1: 完全使用显卡内存，不适用于显存较小的显卡，在显存足够的时候速度较快，使用量化微调时建议开启以加速训练
# deepspeed_stage_2: 使用显卡内存和RAM内存，适用于显存较小的显卡，能节省显存的同时保证速度
# deepspeed_stage_3: 使用显卡内存和RAM内存，硬盘内存，适用于显存极小的显卡，速度最慢，除非RAM和显存都不足，否则不建议使用
DEEPSPEED_STRATEGY=deepspeed_stage_1
# 梯度复制，通常建议开启以节省显存
GRAD_CP=1
# 数据集获取，可选值为：get，pad，only
# get: 从数据中随机截取一段基于上下文长度的数据进行训练，适用于数据集较大但是上下文长度无法调高的情况
# pad: 从数据最开始到结束进行训练，如果数据长度小于上下文长度，则填充上下文，适用于微批次大小大于1的配置，建议使用此配置时根据最长数据调整上下文长度
# only: 从数据最开始到结束进行训练，即使数据集长度超过上下文长度，也会从最开始截取到上下文长度的数据进行训练，适用于微批次大小为1的配置，建议使用此配置时根据最长数据调整上下文长度
# 在上下文长度允许的情况下，更推荐使用pad（微批次大于1）或者only（微批次为1）模式，可以更好的学习到数据的连贯特征
DATALOAD="pad"
# ------------------不常用训练参数----------------------
# 开始训练的回合，可以用来恢复训练
EPOCH_BEGIN=0
# 词表大小
# 此参数应该根据tokenizer/rwkv_vocab_v20230424.txt的词表大小进行调整，更改词表数量后应该修改此参数
VOCAB_SIZE=65536
# 嵌入维度
# 此参数应该根据模型的参数进行调整：
# 14B EMBD_SIZE = 4096
# 7B EMBD_SIZE = 4096
# 3B EMBD_SIZE = 2560
# 1.5B、1.6B、0.43B EMBD_SIZE = 2048
# 0.17B EMBD_SIZE = 768
EMBD_SIZE=2560
# 嵌入层
# 此参数应该根据模型的参数进行调整：
# 14B N_LAYER = 61
# 7B N_LAYER = 32 
# 3B N_LAYER = 32 
# 1.5B、1.6B、0.43B N_LAYER = 24 
# 0.17B  N_LAYER = 12 
N_LAYER=32
# Bata1
BETA1=0.9
# Bata2
BETA2=0.999
# ADAM epsilon
ADAM_EPS=1e-8

# ------------------Lora和Pissa设置参数----------------------
# lora_parts
lora_parts=att,ffn,time,ln
# LORA模型路径，代表从哪个LORA模型开始微调
lora_load="rwkv-0"
# LORA模型的r值
# 越大的r值，微调的效果越好，同时也能让pissa微调的奇异值分解精度越高，但是显存占用越高，建议最大128
# 如果显存或者RAM不足，应该调低此值，一般训练使用32或者64即可，实在不够也可以用16
lora_r=64
# LORA模型的alpha值
# 此值应该配合r值调整
# 计算公式为：lora_alpha=lora_r*2
lora_alpha=128
# LORA模型的dropout值 
lora_dropout=0.01
# pissa的快速奇异值分解的迭代次数，迭代次数越高损失越低，但是初始化的速度就越慢
# 如果需要速度，可以适当调低此值，但是损失会增加
# 一般来说svd_niter=16后就已经非常接近奇异值分解的结果了，往后的迭代次数对结果影响不会很大
# 此外，lora_r对于svd_niter的影响也很大，lora_r越大，训练损失越低，但是svd_niter也需要相应增加
svd_niter=16

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

INFCTX=""
case "$TRAIN_TYPE" in
"state")
    echo "-------------使用$TRAIN_TYPE附加模式微调-------------"
    ;;
"infctx")
    echo "-------------使用$TRAIN_TYPE附加模式微调-------------"
    INFCTX="--train_type infctx"
    ;;
"none")
    ;;
*)
    echo "!!!!!!!!!!!!!不支持的附加微调模式$TRAIN_TYPE，仅支持none, state, infctx!!!!!!!!!!!!!"
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

if [ "$EMB_FINETUNE" = 0 ]; then
    case "$FINETUNE_MODE" in
    "lora"|"pissa")
        echo "-------------不微调embedding层-------------"
        ;;
    *)
        ;;
    esac
    EMB=""
else if [ "$EMB_FINETUNE" = 1 ]; then
    echo "-------------微调embedding层-------------"
    EMB="--emb"
else
    echo "!!!!!!!!!!!!!不支持的微调embedding层参数$EMB_FINETUNE，仅支持0（关闭）, 1（开启）!!!!!!!!!!!!!"
    exit 1
fi
fi

if [ "$FLA" = 0 ]; then
    FLA=""
else if [ "$FLA" = 1 ]; then
    echo "-------------使用triton算子-------------"
    FLA="--fla"
else
    echo "!!!!!!!!!!!!!不支持的FLA参数$FLA，仅支持0（关闭）, 1（开启）!!!!!!!!!!!!!"
    exit 1
fi
fi

case "$DATALOAD" in
"pad"|"get"|"only")
    echo "-------------使用$DATALOAD模式读取数据-------------"
    ;;
*)
    echo "!!!!!!!!!!!!!不支持的数据集获取参数$DATALOAD，仅支持pad, get, only!!!!!!!!!!!!!"
    exit 1
    ;;
esac

if [ "$TRAIN_TYPE" = "state" ]; then
   python3 train.py --load_model model/$MODEL_PATH \
    --proj_dir output --data_file data/$DATA_PATH \
    --data_type binidx --vocab_size $VOCAB_SIZE \
    --ctx_len $CTX_LEN --epoch_steps $EPOCH_STEPS --epoch_count $EPOCH_COUNT --epoch_begin $EPOCH_BEGIN --epoch_save $EPOCH_SAVE --micro_bsz $MICRO_BSZ \
    --n_layer $N_LAYER --n_embd $EMBD_SIZE \
    --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps $WARMUP_STEPS --beta1 $BETA1 --beta2 $BETA2 --adam_eps $ADAM_EPS \
    --accelerator gpu --devices $GPU_COUNT --precision $PRECISION --strategy $DEEPSPEED_STRATEGY --grad_cp $GRAD_CP \
    --accumulate_grad_batches $MINI_BSZ --dataload $DATALOAD --chunk_ctx $CHUNK_CTX \
    --train_type $TRAIN_TYPE $V6_TRAIN $FLA
else if [ "$FINETUNE_MODE" = "lora" ]; then
   python3 train.py --load_model model/$MODEL_PATH \
    --proj_dir output --data_file data/$DATA_PATH \
    --data_type binidx --vocab_size $VOCAB_SIZE \
    --ctx_len $CTX_LEN --epoch_steps $EPOCH_STEPS --epoch_count $EPOCH_COUNT --epoch_begin $EPOCH_BEGIN --epoch_save $EPOCH_SAVE --micro_bsz $MICRO_BSZ \
    --n_layer $N_LAYER --n_embd $EMBD_SIZE \
    --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps $WARMUP_STEPS --beta1 $BETA1 --beta2 $BETA2 --adam_eps 1e-8 \
    --accelerator gpu --devices $GPU_COUNT --precision $PRECISION --strategy $DEEPSPEED_STRATEGY --grad_cp $GRAD_CP \
    --accumulate_grad_batches $MINI_BSZ --dataload $DATALOAD --chunk_ctx $CHUNK_CTX \
    --lora_load $lora_load --lora --lora_r $lora_r --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout --lora_parts=$lora_parts $INFCTX $V6_TRAIN $EMB $FLA
else if [ "$FINETUNE_MODE" = "lisa" ]; then
   python3 train.py --load_model model/$MODEL_PATH \
    --proj_dir output --data_file data/$DATA_PATH \
    --data_type binidx --vocab_size $VOCAB_SIZE \
    --ctx_len $CTX_LEN --epoch_steps $EPOCH_STEPS --epoch_count $EPOCH_COUNT --epoch_begin $EPOCH_BEGIN --epoch_save $EPOCH_SAVE --micro_bsz $MICRO_BSZ \
    --n_layer $N_LAYER --n_embd $EMBD_SIZE \
    --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps $WARMUP_STEPS --beta1 $BETA1 --beta2 $BETA2 --adam_eps $ADAM_EPS \
    --accelerator gpu --devices $GPU_COUNT --precision $PRECISION --strategy $DEEPSPEED_STRATEGY --grad_cp $GRAD_CP \
    --accumulate_grad_batches $MINI_BSZ --dataload $DATALOAD --chunk_ctx $CHUNK_CTX \
    --LISA --lisa_r $lisa_r --lisa_k $lisa_k $INFCTX $V6_TRAIN $EMB $FLA
else if [ "$FINETUNE_MODE" = "pissa" ]; then
   python3 train.py --load_model model/$MODEL_PATH \
    --proj_dir output --data_file data/$DATA_PATH \
    --data_type binidx --vocab_size $VOCAB_SIZE \
    --ctx_len $CTX_LEN --epoch_steps $EPOCH_STEPS --epoch_count $EPOCH_COUNT --epoch_begin $EPOCH_BEGIN --epoch_save $EPOCH_SAVE --micro_bsz $MICRO_BSZ \
    --n_layer $N_LAYER --n_embd $EMBD_SIZE \
    --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps $WARMUP_STEPS --beta1 $BETA1 --beta2 $BETA2 --adam_eps $ADAM_EPS \
    --accelerator gpu --devices $GPU_COUNT --precision $PRECISION --strategy $DEEPSPEED_STRATEGY --grad_cp $GRAD_CP \
    --accumulate_grad_batches $MINI_BSZ --dataload $DATALOAD --chunk_ctx $CHUNK_CTX \
    --lora_load $lora_load --lora --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout $lora_dropout --lora_parts=$lora_parts \
    --PISSA --svd_niter $svd_niter $INFCTX $V6_TRAIN $EMB $FLA
else
    echo "!!!!!!!!!!!!!不支持的微调模式$FINETUNE_MODE，仅支持lora, pissa, lisa!!!!!!!!!!!!!"
    exit 1
fi
fi
fi
fi