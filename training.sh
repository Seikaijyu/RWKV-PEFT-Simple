# 使用方式：调整参数后使用`./training.sh`运行

# ------------------通用参数----------------------
# 微调方式，可选值为：lora, pissa，bone, state_tuning
#
# lora: 使用标准lora微调，但是lora微调速度较慢，效果一般，更推荐pissa
# pissa: lora的改进版本，使用快速奇异值分解，收敛速度更快，效果更好，推荐使用
# bone: 不同于lora系列的全新微调方法。bone_b（类似lora中的lora_r）越大，微调效果越好，泛化越强
# bat: 与bone类似，区别在于bat会让模型学到更多的特征，但是训练速度较慢，效果更好
# state_tuning: 微调init state，微调速度更快，占用显存更低。
#   此微调不会让模型学到没有在预训练中学过的数据，如有需要请使用其它微调方式。
FINETUNE_MODE="bat"

# 微调附加类型，可选值为：none, infctx
#
# infctx: infctx使用时间换内存进行训练，当显存不足但语料过长时建议开启
#  使用infctx时必须将DATALOAD修改为pad模式，否则会导致nan
#  开启后使用更长上下文微调时，不会导致显存使用量增加。
#  开启时最好同时开启FLA以启用triton算子，因为cuda算子的梯度有点问题
TRAIN_TYPE="none"

# 机器学习实验跟踪平台 wandb (Weights & Biases)
#
# 可以用于查看训练的每一步loss，并且可以进行不同训练的loss对比，还有EMA（移动指数平滑）等各种图标展示功能
# 还可以查看设定的训练参数，如需使用，需要在[https://wandb.ai/]注册账号，并复制key
# 在此参数中设定wandb的名字（任意）后根据命令行提示粘贴key（命令行输入key时不显示任何内容是正常的，粘贴后直接回车即可）
# 绑定后即可使用并在每次训练后查看数据图和远程关闭训练等操作
WANDB="wandb"

# 训练的RWKV模型的架构版本，可选值为：v5, v6, v7
MODEL_VERSION="v7"

# 量化方式，可选值为：none, 4bit, nf4, fp4, int8, fp8
#
# 4位量化一般推荐使用nf4，分布更均匀，8位量化则推荐int8，更推荐8位量化，损失更小
# int8量化在pissa中损失更小，而在bone中和非量化结果基本一致
# fp8损失更大，但是训练效率更高
QUANT="none"

# 启用其它算子，可选值为：cuda, fla, triton（v7可用）
#
# MICRO_BSZ越小越推荐开启，训练速度更快
OP="cuda"

# 模型路径
#
# 对应的是在model文件夹下需要微调的模型的文件名
MODEL_PATH=RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth

# 数据路径
#
# 对应的是在data文件夹下需要微调的使用数据的文件名
DATA_PATH=b

# 训练的回合数，达到回合数后会停止训练
#
# 仅在数据读取模式为pad和only时生效
EPOCH_COUNT=10

# 训练数据自动洗牌，从第一个epoch开始打乱训练数据排列
#
# 如果开启此项，使用`make_tokenize.sh`进行分词时训练回合数应该为1，训练回合数最好大于1
# 多个回合的训练会在每个训练回合训练的开始从头读取数据，如果不打乱数据，则每次都会读取到一样的数据排列，影响泛化
#
# 如果关闭此项，使用`make_tokenize.sh`进行分词时训练回合数最好大于1，训练回合数应该为1
# 单个回合的训练会在训练的开始打乱数据，如果打乱数据，则会导致多个回合的数据重复被刷到一块，导致错误的拟合
DATA_SHUFFLE=1

# 回合步数
# 应该根据训练数据的条数和微批次大小调整，公式为：数据集条数/微批次大小=回合步数
EPOCH_STEPS=3280

# loss掩码，可选值为：none, pad, qa, se
#
# 此参数用于指示模型在计算损失时应该关注哪些位置，带来的好处是LLM不会学习和模仿Mask部分说话
# 让模型可以更快地学习到有用的特征，并减少不必要的计算
#
# qa: 忽略掉名为"User"的角色回复的loss不纳入计算
#  如果找不到"User"和"Assistant"（对话不使用User:xxx\n\nAssistant:xxx）格式
#  则自动降级为pad模式
#
# pad: 忽略掉所有超出当条数据长度的部分不做loss计算
#  如CTX_LEN=200，但是某一条输入数据只有100，则忽略超出实际数据长度部分loss计算
#
# se: 目前有问题，等待上游更新
LOSS_MASK="qa"

# 上下文长度
#
# 使用./make_tokenize.sh {数据集名称}.jsonl脚本进行数据分词时能得到如：### max_length = 208 这样的输出
# 其中208就是数据中最长的数据的长度，在pad模式和only模式中，应该填入此数值以保证数据能够完全被训练
# 如果数据过长无法训练，建议降低上下文长度并使用get模式读取数据，可以节省资源
# 使用./make_tokenize.sh {数据集名称}.jsonl 1 进行数据分词即可
CTX_LEN=7872

# 开启微调附加项的infctx参数后启用的设置，此设置用于确定在infctx中单次训练的上下文长度，此参数越高，消耗的显存越多
#
# 相当于不开启infctx时的CTX_LEN参数，一般建议能开多大开多大（仅在infctx启用时有效）
# CTX_LEN必须大于或者等于此参数
CHUNK_CTX=512

# 精度，可选值为：fp32, bf16, fp16，通常建议使用bf16，节省显存同时保证了训练精度
PRECISION=bf16

# 初始学习率
#
# 如果使用state模式微调，lr最好调高，建议使用动态学习率，从1到0.01，使用其它模式建议5e-5到1e-4之间，优先选择5e-5
LR_INIT=5e-5

# 最终学习率
#
# 通常建议和初始学习率一致，除非需要动态学习率
# 动态学习率的下降速度与你设定的训练回合数和训练步数有关
LR_FINAL=5e-5

# 学习率衰减策略，可选值为：cos, wsd
# cos: 余弦衰减策略：
#  初始lr定义了学习率的最大值，通常在训练开始时使用。
#  最终lr定义了学习率的最小值，通常在训练结束时达到。
#  学习率会按照余弦函数的形状从初始lr平滑地降低到最终lr。
#  这种策略提供了一个从高到低的平滑过渡，有助于在训练初期快速学习，后期微调。
#
# wsd: 余弦退火策略：
#  初始lr和最终lr同样定义了学习率变化的总体范围。
#  但学习率可能会在这个范围内周期性地上下波动。
#  每个周期可能从接近初始lr的值开始，然后降低到接近最终lr的值。
#  不同周期的最大值可能会逐渐降低，最终趋近于最终lr。
#  这种策略提供了更动态的学习率调整，可能有助于模型跳出局部最优解。
LR_SCHEDULE="cos"

# 预热步数
#
# 此配置与学习率相关，如果使用不同的初始学习率和最终学习率，建议调整此配置
#
# 此配置规定了在训练开始阶段使用以下公式的学习率作为预热初始学习率：预热初始学习率=初始学习率/5 
# 然后在训练过程中基于设定的预热步数训练到指定的实际步数，计算公式为：实际步数=梯度累计x训练步数
# 此时lr将为初始学习率，然后再从初始学习率开始逐渐迭代到最终学习率
# 开启时建议严格控制训练步数和回合数，以达到最优效果，模型尺寸越大预热步数应当越多
WARMUP_STEPS=20

# 显卡数量
# 
# 此配置项大于1时DATA_SHUFFLE（数据洗牌）参数会被主动关闭，此时只能训练一回合
# 此时应该时候此命令进行分词 `./make_tokenize.sh {data目录中的文件名称，包括.jsonl} {训练的回合数}`
#
# 应该遵守以下公式修改训练步数（在梯度累计和微批次大小计算好训练步数后的基础上进行计算）：
# 训练步数/GPU数量=训练步数
GPU_COUNT=1

# 微批次大小，此配置项越大，显存占用越大，但是训练速度越快
#
# 此配置项非1时应该跟随数据集条数调整，计算公式为：数据集条数/微批次大小=回合步数
# 例如：数据集条数为10000，微批次大小为10，回合步数应该设置为1000
# 微批次大小并不是越大越好，更大的微批次大小会导致学不到东西，建议根据数据集数量调整
# 一般来说，数据集条数小于5k条时候，微批次大小建议最大为8
# 如果数据量大，可以设置到16或者32以及更高
MICRO_BSZ=1

# 模型保存间隔，每隔多少回合保存一次模型
EPOCH_SAVE=1

# 梯度累计，如果显存不够无法调整微批次大小
#
# 此配置项非1时应该跟随数据集条数调整，但是和微批次大小不同，梯度累计不能让训练步数降低
# 例如：数据集条数为10000，微批次大小为10，梯度累计为2
# 回合步数应该设置为1000而不是500，梯度累计不能降低每个训练步数的数量
#
# 梯度累计可以理解为微批次大小的下位替代，效果会差一点
# 一般来说，数据集条数小于5k条时候，梯度累计建议最大为8
# 如果数据量大，可以设置到16或者32以及更高
MINI_BSZ=16

# 优化策略, 可选值为：deepspeed_stage_1, deepspeed_stage_2, deepspeed_stage_3
#
# 建议使用deepspeed_stage_2节省显存的同时也能保证微调速度
# deepspeed_stage_1: 完全使用显卡内存，不适用于显存较小的显卡，在显存足够的时候速度较快，使用量化微调时建议开启以加速训练
# deepspeed_stage_2: 使用显卡内存和RAM内存，适用于显存较小的显卡，能节省显存的同时保证速度
# deepspeed_stage_3: 使用显卡内存和RAM内存，硬盘内存，适用于显存极小的显卡，速度最慢，除非RAM和显存都不足，否则不建议使用
DEEPSPEED_STRATEGY=deepspeed_stage_1

# 梯度复制，通常建议开启以节省显存
GRAD_CP=1

# 数据集获取，可选值为：get，pad，only
#
# get: 从数据中随机截取一段基于上下文长度的数据进行训练，适用于数据集较大但是上下文长度无法调高的情况
# pad: 从数据最开始到结束进行训练，如果数据长度小于上下文长度，则填充上下文，适用于微批次大小大于1的配置，建议使用此配置时根据最长数据调整上下文长度
# only: 从数据最开始到结束进行训练，即使数据集长度超过上下文长度，也会从最开始截取到上下文长度的数据进行训练，适用于微批次大小为1的配置，建议使用此配置时根据最长数据调整上下文长度
# 在上下文长度允许的情况下，更推荐使用pad（微批次大于1）或者only（微批次为1）模式，可以更好的学习到数据的连贯特征
DATALOAD="pad"

# ------------------不常用训练参数----------------------
# 开始训练的回合，可以用来恢复训练
EPOCH_BEGIN=0

# 词表大小
#
# 此参数应该根据tokenizer/rwkv_vocab_v20230424.txt的词表大小进行调整，更改词表数量后应该修改此参数
VOCAB_SIZE=65536

# 嵌入维度
#
# 此参数应该根据模型的参数进行调整：
# v6
# 14B EMBD_SIZE = 4096
# 7B EMBD_SIZE = 4096
# 3B EMBD_SIZE = 2560
# 1.5B、1.6B、0.43B EMBD_SIZE = 2048
# 0.17B EMBD_SIZE = 768

# v7
# 0.4B EMBD_SIZE = 1024
EMBD_SIZE=1024

# 嵌入层
#
# 此参数应该根据模型的参数进行调整：
# v6
# 14B N_LAYER = 61
# 7B N_LAYER = 32 
# 3B N_LAYER = 32 
# 1.5B、1.6B、0.43B N_LAYER = 24 
# 0.17B  N_LAYER = 12

# v7
# 0.4B N_LAYER = 24
N_LAYER=24

# Bata1
BETA1=0.9

# Bata2
BETA2=0.95

# ADAM epsilon
ADAM_EPS=1e-18

# ------------------LoRA设置参数----------------------

# LoRA训练模型路径
#
# 代表从哪个LoRA模型开始微调，格式一般为
# "rwkv-0.pth" "rwkv-10.pth"这样即可
lora_load=""

# LoRA模型的r值
#
# 如果显存或者RAM不足，应该调低此值，一般训练使用32或者64即可，实在不够也可以用16
lora_r=64


# LORA模型的alpha值
#
# 此值应该配合r值调整
# 计算公式为：lora_alpha=lora_r*2
lora_alpha=128

# LORA模型的dropout值
#
# lora_dropout在训练过程中随机丢弃一部分神经元来防止过拟合。
# lora_dropout=0.01 表示在训练过程中，每次更新时有 1% 的神经元会被随机丢弃。这样可以提高模型的泛化能力。
# 但同时也会降低训练效率，因此在训练过程中，lora_dropout的值一般设置为0.01
lora_dropout=0.01


# ------------------PiSSA设置参数----------------------
# PiSSA初始化的模型路径
#
# 代表从哪个PiSSA模型的初始化开始加载PiSSA
# 如果需要继续训练，一般设置为
# "init_pissa.pth"即可
pissa_init=""

# PiSSA训练模型路径
#
# 代表从哪个PiSSA模型开始微调，格式一般为
# "rwkv-0.pth" "rwkv-10.pth"这样即可
pissa_load=""

# PiSSA模型的r值
#
# 越大的r值，微调的效果越好，同时也能让PiSSA微调的奇异值分解精度越高，但是显存占用越高，建议最大128
# 如果显存或者RAM不足，应该调低此值，一般训练使用32或者64即可，实在不够也可以用16
pissa_r=64

# PiSSA的快速奇异值分解的迭代次数，迭代次数越高损失越低，但是初始化的速度就越慢
#
# 如果需要快速初始化（如微调14B），可以适当调低此值，但是损失会增加
# 一般来说svd_niter=16后就已经非常接近奇异值分解的结果了，往后的迭代次数对结果影响不会很大
# 此外，lora_r对于svd_niter的影响也很大，lora_r越大，训练损失越低，但是svd_niter也需要相应增加
svd_niter=16


# ------------------DiSHA设置参数----------------------
# DiSHA训练模型路径
#
# 代表从哪个检查点开始微调，格式一般为
# "rwkv-0.pth" "rwkv-10.pth"这样即可
disha_load=""

# DiSHA的r值
#
# 类似lora的r，disha_r=128相当于lora_r=64的占用，越大的值微调的参数量越多
# disha_r必须能被维度整除
disha_r=1024



















































# ---------------------源代码---------------------
case "$MODEL_VERSION" in
"v7")
    echo "-------------RWKV7微调模式-------------"
    TRAIN_VERSION="--my_testing x070"
    ;;
"v6")
    echo "-------------RWKV6微调模式-------------"
    TRAIN_VERSION="--my_testing x060"
    ;;
"v5")
    echo "-------------RWKV5微调模式-------------"
    TRAIN_VERSION=""
    ;;
*)
    echo "!!!!!!!!!!!!!不支持的模型版本$MODEL_VERSION，仅支持v5, v6，另外v4已经过时，不建议使用!!!!!!!!!!!!!"
    exit 1
    ;;
esac

INFCTX=""
case "$TRAIN_TYPE" in
"infctx")
    if [ "$FINETUNE_MODE" = "state_tuning" ]; then
        echo "!!!!!!!!!!!!!state_tuning不支持的附加微调模式$TRAIN_TYPE!!!!!!!!!!!!!"
        exit 1
    fi
    echo "-------------使用$TRAIN_TYPE附加模式微调-------------"
    INFCTX="--train_type infctx"
    ;;
"none")
    ;;
*)
    echo "!!!!!!!!!!!!!不支持的附加微调模式$TRAIN_TYPE，仅支持none, infctx!!!!!!!!!!!!!"
    exit 1
    ;;
esac


case "$LOSS_MASK" in
    "qa"|"pad"|"se")
        echo "-------------使用$LOSS_MASK模式-------------"
        ;;
"none")
    ;;
*)
    echo "!!!!!!!!!!!!!不支持的loss掩码参数$LOSS_MASK，仅支持none, qa, pad, se!!!!!!!!!!!!!"
    exit 1
    ;;
esac

case "$QUANT" in
"4bit"|"nf4"|"fp4"|"int8"|"fp8")
    echo "-------------使用$QUANT精度量化微调-------------"
    ;;
"none")
    echo "-------------不使用量化微调-------------"
    ;;
*)
    echo "!!!!!!!!!!!!!不支持的量化精度参数$QUANT，仅支持none, 4bit, nf4, fp4, int8, fp8!!!!!!!!!!!!!"
    exit 1
    ;;
esac



if [ "$OP" = "cuda" ]; then
    echo "-------------使用cuda算子-------------"
elif [ "$OP" = "fla" ]; then
    echo "-------------使用fla算子-------------"
elif [ "$OP" = "triton" ]; then
    echo "-------------使用triton算子-------------"
else
    echo "!!!!!!!!!!!!!不支持的OP参数$OP，仅支持cuda, fla, triton（v7可用）!!!!!!!!!!!!!"
    exit 1
fi

case "$LR_SCHEDULE" in
"cos"|"wsd")
    echo "-------------使用$LR_SCHEDULE学习率衰减策略-------------"
    ;;
*)
    echo "!!!!!!!!!!!!!不支持的学习率衰减策略参数$LR_SCHEDULE，仅支持cos, wsd!!!!!!!!!!!!!"
    exit 1
    ;;
esac

case "$DATALOAD" in
"pad"|"get"|"only")
    echo "-------------使用$DATALOAD模式读取数据-------------"
    ;;
*)
    echo "!!!!!!!!!!!!!不支持的数据集获取参数$DATALOAD，仅支持pad, get, only!!!!!!!!!!!!!"
    exit 1
    ;;
esac

if [ "$FINETUNE_MODE" = "state_tuning" ]; then
   COMMAND="python3 train.py --load_model 'model/${MODEL_PATH}' \
    --proj_dir 'output' --data_file 'data/${DATA_PATH}' \
    --data_type binidx --vocab_size ${VOCAB_SIZE} \
    --ctx_len ${CTX_LEN} --epoch_steps ${EPOCH_STEPS} --epoch_count ${EPOCH_COUNT} --epoch_begin ${EPOCH_BEGIN} --epoch_save ${EPOCH_SAVE} --micro_bsz ${MICRO_BSZ} \
    --n_layer ${N_LAYER} --n_embd ${EMBD_SIZE} \
    --lr_init ${LR_INIT} --lr_final ${LR_FINAL} --warmup_steps ${WARMUP_STEPS} --beta1 ${BETA1} --beta2 ${BETA2} --adam_eps ${ADAM_EPS} \
    --accelerator gpu --devices ${GPU_COUNT} --precision ${PRECISION} --strategy ${DEEPSPEED_STRATEGY} --grad_cp ${GRAD_CP} \
    --accumulate_grad_batches ${MINI_BSZ} --dataload ${DATALOAD} --chunk_ctx ${CHUNK_CTX} --data_shuffle ${DATA_SHUFFLE} \
    --quant ${QUANT} --loss_mask ${LOSS_MASK} --train_type 'state' --op ${OP} --lr_schedule ${LR_SCHEDULE}\
    --wandb \"${WANDB}\" ${INFCTX} ${TRAIN_VERSION}"
    echo "-------------使用StateTuning方法微调-------------"
elif [ "$FINETUNE_MODE" = "lora" ]; then
    printf -v lora_config '{"lora_load":"%s","lora_r":%s,"lora_alpha":%s,"lora_dropout":%s}' "$lora_load" "$lora_r" "$lora_alpha" "$lora_dropout"
   COMMAND="python3 train.py --load_model 'model/${MODEL_PATH}' \
    --proj_dir 'output' --data_file 'data/${DATA_PATH}' \
    --data_type binidx --vocab_size ${VOCAB_SIZE} \
    --ctx_len ${CTX_LEN} --epoch_steps ${EPOCH_STEPS} --epoch_count ${EPOCH_COUNT} --epoch_begin ${EPOCH_BEGIN} --epoch_save ${EPOCH_SAVE} --micro_bsz ${MICRO_BSZ} \
    --n_layer ${N_LAYER} --n_embd ${EMBD_SIZE} \
    --lr_init ${LR_INIT} --lr_final ${LR_FINAL} --warmup_steps ${WARMUP_STEPS} --beta1 ${BETA1} --beta2 ${BETA2} --adam_eps 1e-8 \
    --accelerator gpu --devices ${GPU_COUNT} --precision ${PRECISION} --strategy ${DEEPSPEED_STRATEGY} --grad_cp ${GRAD_CP} \
    --accumulate_grad_batches ${MINI_BSZ} --dataload ${DATALOAD} --chunk_ctx ${CHUNK_CTX} --data_shuffle ${DATA_SHUFFLE} \
    --peft lora --lora_config '${lora_config}' \
    --wandb \"${WANDB}\" --quant ${QUANT} --loss_mask ${LOSS_MASK} ${INFCTX} ${TRAIN_VERSION} --op ${OP} --lr_schedule ${LR_SCHEDULE}"
    echo "-------------使用LoRA方法微调-------------"
elif [ "$FINETUNE_MODE" = "bone" ]; then
    printf -v disha_config '{"mode":"bone","load":"%s","r":%s}' "$disha_load" "$disha_r"
   COMMAND="python3 train.py --load_model 'model/${MODEL_PATH}' \
    --proj_dir 'output' --data_file 'data/${DATA_PATH}' \
    --data_type binidx --vocab_size ${VOCAB_SIZE} \
    --ctx_len ${CTX_LEN} --epoch_steps ${EPOCH_STEPS} --epoch_count ${EPOCH_COUNT} --epoch_begin ${EPOCH_BEGIN} --epoch_save ${EPOCH_SAVE} --micro_bsz ${MICRO_BSZ} \
    --n_layer ${N_LAYER} --n_embd ${EMBD_SIZE} \
    --lr_init ${LR_INIT} --lr_final ${LR_FINAL} --warmup_steps ${WARMUP_STEPS} --beta1 ${BETA1} --beta2 ${BETA2} --adam_eps ${ADAM_EPS} \
    --accelerator gpu --devices ${GPU_COUNT} --precision ${PRECISION} --strategy ${DEEPSPEED_STRATEGY} --grad_cp ${GRAD_CP} \
    --accumulate_grad_batches ${MINI_BSZ} --dataload ${DATALOAD} --chunk_ctx ${CHUNK_CTX} --data_shuffle ${DATA_SHUFFLE} \
    --wandb \"${WANDB}\" --quant ${QUANT} --peft disha --disha_config '${disha_config}' --loss_mask ${LOSS_MASK}  --op ${OP} --lr_schedule ${LR_SCHEDULE} ${INFCTX} ${TRAIN_VERSION} ${EMB}"
    echo "-------------使用Bone方法微调-------------"
elif [ "$FINETUNE_MODE" = "bat" ]; then
    printf -v disha_config '{"mode":"bat","load":"%s","r":%s}' "$disha_load" "$disha_r"
   COMMAND="python3 train.py --load_model 'model/${MODEL_PATH}' \
    --proj_dir 'output' --data_file 'data/${DATA_PATH}' \
    --data_type binidx --vocab_size ${VOCAB_SIZE} \
    --ctx_len ${CTX_LEN} --epoch_steps ${EPOCH_STEPS} --epoch_count ${EPOCH_COUNT} --epoch_begin ${EPOCH_BEGIN} --epoch_save ${EPOCH_SAVE} --micro_bsz ${MICRO_BSZ} \
    --n_layer ${N_LAYER} --n_embd ${EMBD_SIZE} \
    --lr_init ${LR_INIT} --lr_final ${LR_FINAL} --warmup_steps ${WARMUP_STEPS} --beta1 ${BETA1} --beta2 ${BETA2} --adam_eps ${ADAM_EPS} \
    --accelerator gpu --devices ${GPU_COUNT} --precision ${PRECISION} --strategy ${DEEPSPEED_STRATEGY} --grad_cp ${GRAD_CP} \
    --accumulate_grad_batches ${MINI_BSZ} --dataload ${DATALOAD} --chunk_ctx ${CHUNK_CTX} --data_shuffle ${DATA_SHUFFLE} \
    --wandb \"${WANDB}\" --quant ${QUANT} --peft disha --disha_config '${disha_config}' --loss_mask ${LOSS_MASK}  --op ${OP}  --lr_schedule ${LR_SCHEDULE} ${INFCTX} ${TRAIN_VERSION} ${EMB}"
    echo "-------------使用Bat方法微调-------------"
elif [ "$FINETUNE_MODE" = "pissa" ]; then
    printf -v pissa_config '{"pissa_load":"%s","pissa_init":"%s","pissa_r":%s,"svd_niter":%s}' "$pissa_load" "$pissa_init" "$pissa_r" "$svd_niter"
   COMMAND="python3 train.py --load_model 'model/${MODEL_PATH}' \
    --proj_dir 'output' --data_file 'data/${DATA_PATH}' \
    --data_type binidx --vocab_size ${VOCAB_SIZE} \
    --ctx_len ${CTX_LEN} --epoch_steps ${EPOCH_STEPS} --epoch_count ${EPOCH_COUNT} --epoch_begin ${EPOCH_BEGIN} --epoch_save ${EPOCH_SAVE} --micro_bsz ${MICRO_BSZ} \
    --n_layer ${N_LAYER} --n_embd ${EMBD_SIZE} \
    --lr_init ${LR_INIT} --lr_final ${LR_FINAL} --warmup_steps ${WARMUP_STEPS} --beta1 ${BETA1} --beta2 ${BETA2} --adam_eps ${ADAM_EPS} \
    --accelerator gpu --devices ${GPU_COUNT} --precision ${PRECISION} --strategy ${DEEPSPEED_STRATEGY} --grad_cp ${GRAD_CP} \
    --accumulate_grad_batches ${MINI_BSZ} --dataload ${DATALOAD} --chunk_ctx ${CHUNK_CTX} --data_shuffle ${DATA_SHUFFLE} \
    --peft pissa --pissa_config '${pissa_config}' \
    --wandb \"${WANDB}\" --quant ${QUANT} --loss_mask ${LOSS_MASK} --op ${OP} --lr_schedule ${LR_SCHEDULE} ${INFCTX} ${TRAIN_VERSION}"
    echo "-------------使用PiSSA方法微调-------------"
else
    echo "!!!!!!!!!!!!!不支持的微调方法$FINETUNE_MODE，仅支持state_tuning, lora, pissa, bone, bat!!!!!!!!!!!!!"
    exit 1
fi
CURRENT_DATE_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "# $CURRENT_DATE_TIME" >> history_run_command.sh.log
echo "$COMMAND" >> history_run_command.sh.log
echo "" >> history_run_command.sh.log
eval "$COMMAND" 