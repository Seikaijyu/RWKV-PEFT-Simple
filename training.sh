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
FINETUNE_MODE="bone"

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
WANDB=""

# 训练的RWKV模型的架构版本，可选值为：v5, v6, v7
MODEL_VERSION="v7"

# 量化方式，可选值为：none, 4bit, nf4, fp4, int8, fp8
#
# 4位量化一般推荐使用nf4，分布更均匀，8位量化则推荐int8，更推荐8位量化，损失更小
# int8量化在pissa中损失更小，而在bone中和非量化结果基本一致
# fp8损失更大，但是训练效率更高
# 多卡微调中不建议进行任何程度的量化，因为会有一显卡的显存被占满，其它卡则占用很低的情况发生
# 无法均衡负载微调
QUANT="none"

# 启用其它算子，可选值为：cuda, fla, triton（v7可用）
#
# 如需进行state_tuning微调，必须使用fla算子，否则建议使用cuda算子（前提是你的显卡是NVIDIA的）
OP="cuda"

# 模型路径
#
# 对应的是在model文件夹下需要微调的模型的文件名
MODEL_PATH=RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth

# 嵌入维度
#
# 此参数应该根据模型的参数量进行调整：
# v6
# 14B EMBD_SIZE = 4096
# 7B EMBD_SIZE = 4096
# 3B EMBD_SIZE = 2560
# 1.5B、1.6B、0.43B EMBD_SIZE = 2048
# 0.17B EMBD_SIZE = 768

# v7
# 3B EMBD_SIZE = 2560
# 1.5B EMBD_SIZE = 2048
# 0.4B EMBD_SIZE = 1024
EMBD_SIZE=2048

# 嵌入层
#
# 此参数应该根据模型的参数量进行调整：
# v6
# 14B N_LAYER = 61
# 7B N_LAYER = 32 
# 3B N_LAYER = 32 
# 1.5B、1.6B、0.43B N_LAYER = 24 
# 0.17B  N_LAYER = 12

# v7
# 3B N_LAYER = 32
# 1.5B N_LAYER = 24
# 0.4B N_LAYER = 24
N_LAYER=24

# 数据路径
#
# 对应的是在data文件夹下需要微调的使用数据的文件名
DATA_PATH=sample


# 上下文长度
#
# 使用./make_tokenize.sh {数据集名称}.jsonl脚本进行数据分词时能得到如：### max_length = 208 这样的输出
# 其中208就是数据中最长的数据的长度，在pad模式和only模式中，应该填入此数值以保证数据能够完全被训练
# 如果数据过长无法训练，建议降低上下文长度并使用get模式读取数据，可以节省资源
# 使用./make_tokenize.sh {数据集名称}.jsonl 1 进行数据分词即可
CTX_LEN=608

# 训练的回合数，达到回合数后会停止训练
#
# 仅在数据读取模式为pad和only时生效
EPOCH_COUNT=3

# 回合步数
# 应该根据训练数据的条数和微批次大小调整，公式为：数据集条数/微批次大小=回合步数
EPOCH_STEPS=16

# 模型保存间隔，每隔多少回合保存一次模型
EPOCH_SAVE=1

# 初始学习率
#
# 如果使用state模式微调，lr最好调高，建议使用动态学习率，从1到0.01，使用其它模式建议8e-5到1e-5之间
# 优先选择2e-5到3e-5之间的范围作为初始学习率
LR_INIT=2e-5

# 最终学习率
#
# 通常建议和初始学习率一致，除非需要动态学习率
# 动态学习率的下降速度与你设定的训练回合数和训练步数有关
LR_FINAL=2e-5

# 学习率衰减策略，可选值为：cos, wsd
#
# cos: 余弦衰减策略：
#   初始lr定义了学习率的最大值，通常在训练开始时使用。
#   最终lr定义了学习率的最小值，通常在训练结束时达到。
#   学习率会按照余弦函数的形状从初始lr平滑地降低到最终lr。
#   这种策略提供了一个从高到低的平滑过渡，有助于在训练初期快速学习，后期微调。

# wsd: 预热-稳定-衰减 (Warmup-Stable-Decay) 策略：
#   这个策略，可以理解成让模型先进行一个预热阶段，调整到最佳状态，然后再全力进行训练，最后平稳结束。它通常包含以下几个阶段：
#
#   1. 预热 (Warmup) 阶段：
#      在训练的初始阶段，学习率并不会直接采用你设定的“初始学习率 (LR_INIT)”，而是从一个极小的值（例如0，或者一个低到可以忽略的值）开始。
#      随后，在指定的“预热步数” (warmup steps) 范围内，学习率会逐渐线性或按其他方式增加，直至达到你设定的“初始学习率 (LR_INIT)”。
#      这样做的目的是为了在训练初期稳定模型的参数更新，避免因初始学习率设置过高导致训练过程出现震荡或发散。
#
#   2. 稳定 (Stable) 阶段 (此阶段为可选)：
#      当学习率通过预热达到“初始学习率 (LR_INIT)”之后，可能会在该学习率水平上保持一段固定的训练步数。
#      这个稳定期的长度由你配置，如果不需要，也可以将其设置为0以跳过此阶段。
#
#   3. 衰减 (Decay) 阶段：
#      在预热阶段（以及可能的稳定阶段）完成后，学习率将从“初始学习率 (LR_INIT)”开始，
#      按照预设的方式（例如线性、余弦等）逐渐降低，直至达到你设定的“最终学习率 (LR_FINAL)”。这个最终学习率通常是一个非常小的值，或者为0，标志着学习率调整的结束。
#
#   在 `wsd` 策略中：
#   “初始lr (LR_INIT)”：通常指的是预热阶段完成时达到、并且是后续衰减阶段开始时的目标学习率。
#   “最终lr (LR_FINAL)”：是学习率在训练末期将降低到的最低数值。（目前PEFT还有问题，Decay步骤不正确）
LR_SCHEDULE="cos"

# WARMUP_STEPS: 预热阶段的优化器步数。
#
# 这个参数控制着学习率从一个低点爬升到 LR_INIT 所需要的优化器迭代次数。
# 预热期间的学习率具体是这么变化的：
#
# 1. 起始学习率：
#    训练刚开始的时候，学习率并不是直接就用 LR_INIT，而是会从一个更低的值开始，具体来说，就是 LR_INIT / 5。
#
# 2. 学习率爬升：
#    在 WARMUP_STEPS 指定的这么多优化器步数内，学习率会从上面那个 LR_INIT / 5 的起始值，稳步地、通常是线性地增长，直到达到 LR_INIT。
#
# 3. 预热完成，进入下一阶段：
#    一旦完成了这 WARMUP_STEPS 步的预热，学习率就正好达到了 LR_INIT。从这个时候开始，学习率就会按照你选定的衰减策略（比如 cos、wsd ）从 LR_INIT 逐渐向 LR_FINAL 过渡。
#
# 此参数的步数需要基于以下公式设定：数据集条数（DATA_LINE_COUNT） / 微批次大小（MICRO_BSZ） / 梯度累计（MINI_BSZ） = 总预热步数
# 关于怎么设置 WARMUP_STEPS，给你点实用建议：
# - 这个值不是孤立的，你需要把它和你的总训练步数、LR_INIT、LR_FINAL 这些参数一起考虑，让它们互相配合好，才能发挥最大效用。
# - 一般的经验是，如果你的模型特别大，或者你计划的训练总步数比较多，那么适当增加 WARMUP_STEPS 通常是个好主意。这能帮助训练在开始时更稳定，不容易跑偏，有时候还能帮助模型找到更好的收敛点。
# - 特别注意：当你设定的 LR_INIT 比较高，或者 LR_INIT 和 LR_FINAL 之间的差距比较大的时候，一个设计合理的预热阶段就显得尤其重要了。
WARMUP_STEPS=20

# 显卡数量
GPU_COUNT=1

# 微批次大小，此配置项越大，显存占用越大，但是训练速度越快，同时配合梯度累计（MINI_BSZ）让训练更平稳
#
# 此配置项非1时应该跟随数据集条数调整回合步数（EPOCH_STEPS）参数，计算公式为：数据集条数（DATA_LINE_COUNT） / 微批次大小（MICRO_BSZ） = 回合步数（EPOCH_STEPS）
# 
# 例如：数据集条数为10000，微批次大小为10，回合步数应该设置为1000
# 微批次大小并不是越大越好，更大的微批次大小会导致学不到东西，同时增加显存占用和降低微调速度，建议根据数据集数量调整
# 一般来说，数据集条数小于5k条时，微批次大小（MICRO_BSZ）*梯度累计（MINI_BSZ）建议为8或16
# 数据集条数大于1w条时，微批次大小（MICRO_BSZ）*梯度累计（MINI_BSZ）建议为32或64
# 如果过数据小于1k甚至200条，微批次大小（MICRO_BSZ）*梯度累计（MINI_BSZ）建议为4到8
# 再少就太离谱了点，如果你真有这么少的数据，就学学sample数据集微调的调参吧（当然这只是给你跑测试的，这么少的数据无法保证效果）
MICRO_BSZ=1

# 梯度累计，如果显存不够无法调整微批次大小
#
# 此配置项非1时应该跟随数据集条数调整，但是和微批次大小不同，梯度累计不能让训练步数降低
# 例如：数据集条数（DATA_LINE_COUNT）为10000，微批次大小（MINI_BSZ）为10，梯度累计（MINI_BSZ）为2
# 回合步数（EPOCH_STEPS）应该设置为1000而不是500
# 梯度累计不能降低每个训练步数的数量，但是可以稳定loss，可以作为微批次大小的下位替代
#
# 梯度累计可以理解为微批次大小的下位替代，效果会差一点
# 梯度累计并不是越大越好，更大的梯度累计会导致学不到东西，建议根据数据集数量调整
# 一般来说，数据集条数小于5k条时，微批次大小（MICRO_BSZ）*梯度累计（MINI_BSZ）建议为8或16
# 数据集条数大于1w条时，微批次大小（MICRO_BSZ）*梯度累计（MINI_BSZ）建议为32或64
# 如果过数据小于1k甚至200条，微批次大小（MICRO_BSZ）*梯度累计（MINI_BSZ）建议为4到8
# 再少就太离谱了点，如果你真有这么少的数据，就学学sample数据集微调的调参吧（当然这只是给你跑测试的，这么少的数据无法保证效果）
MINI_BSZ=2

# 数据集读取模式，可选值为：get，pad，only
#
# get: 从数据中随机截取一段基于上下文长度的数据进行训练，适用于数据集较大但是上下文长度无法调高的情况
# pad: 从数据最开始到结束进行训练，如果数据长度小于上下文长度，则填充上下文，适用于微批次大小大于1的配置，建议使用此配置时根据最长数据调整上下文长度
# only: 从数据最开始到结束进行训练，即使数据集长度超过上下文长度，也会从最开始截取到上下文长度的数据进行训练，适用于微批次大小为1的配置，建议使用此配置时根据最长数据调整上下文长度
# 在上下文长度允许的情况下，更推荐使用pad（微批次大于1）或者only（微批次为1）模式，可以更好的学习到数据的连贯特征
DATALOAD="pad"

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

# 开启微调附加项的infctx参数后启用的设置，此设置用于确定在infctx中单次训练的上下文长度，此参数越高，消耗的显存越多
#
# 相当于不开启infctx时的CTX_LEN参数，一般建议能开多大开多大（仅在infctx启用时有效）
# CTX_LEN必须大于或者等于此参数
CHUNK_CTX=512

# 训练数据自动洗牌，从第一个epoch开始打乱训练数据排列
#
# 如果开启此项，使用`make_tokenize.sh`进行分词时训练回合数应该为1，训练回合数最好大于1
# 多个回合的训练会在每个训练回合训练的开始从头读取数据，如果不打乱数据，则每次都会读取到一样的数据排列，影响泛化
#
# 如果关闭此项，使用`make_tokenize.sh`进行分词时训练回合数最好大于1，训练回合数应该为1
# 单个回合的训练会在训练的开始打乱数据，如果打乱数据，则会导致多个回合的数据重复被刷到一块，导致错误的拟合
DATA_SHUFFLE=1

# 优化策略, 可选值为：deepspeed_stage_1, deepspeed_stage_2, deepspeed_stage_3
#
# 建议使用deepspeed_stage_2节省显存的同时也能保证微调速度
# deepspeed_stage_1: 完全使用显卡内存，不适用于显存较小的显卡，在显存足够的时候速度较快，使用量化微调时建议开启以加速训练
# deepspeed_stage_2: 使用显卡内存和RAM内存，适用于显存较小的显卡，能节省显存的同时保证速度
# deepspeed_stage_3: 使用显卡内存和RAM内存，硬盘内存，适用于显存极小的显卡，速度最慢，除非RAM和显存都不足，否则不建议使用
DEEPSPEED_STRATEGY=deepspeed_stage_1

# ------------------不常用训练参数----------------------
# 精度，可选值为：fp32, bf16, fp16，通常建议使用bf16，节省显存同时保证了训练精度
PRECISION=bf16

# 梯度复制，通常建议开启以节省显存
GRAD_CP=1

# 开始训练的回合，可以用来恢复训练
EPOCH_BEGIN=0

# 词表大小
#
# 此参数应该根据tokenizer/rwkv_vocab_v20230424.txt的词表大小进行调整，更改词表数量后应该修改此参数
VOCAB_SIZE=65536


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
# 通常64或者128即可，实在不够也可以用32，128已经几乎可以释放DiSHA的全部潜力了
# 除非你的微调场景非常特殊，甚至模型根本没见过，否则不建议使用更大的值
# disha_r必须能被维度整除
disha_r=128


















































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