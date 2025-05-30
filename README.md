# 更新公告
#### 依赖项目[RWKV-PEFT](https://github.com/Seikaijyu/RWKV-PEFT)更新到了`15693e5`（2025/5/2）版本，此版本完全支持RWKV7微调，同时允许修改学习率衰减模式（衰减模式已修复）
#### 不管怎么样，推荐点此下载fork的[RWKV-PEFT](https://github.com/Seikaijyu/RWKV-PEFT)，此版本一定经过验证，嗯...至少比RWKV-PEFT的原始地址验证的多一些（你如果经常更新版本可能会发现原始地址提供的RWKV-PEFT可能不可用）原始地址提供的版本更像是dev版，而此[fork版本](https://github.com/Seikaijyu/RWKV-PEFT)则只会在尝试版本可用后更新，或者会因为新版本出现问题而回退，特别推荐！
#### 现在，你每次运行的微调脚本将被记录在`history_run_command.sh.log`文件中。嗯......或许可以在遇到有关RWKV-PEFT的问题时把`history_run_command.sh.log`文件中使用的脚本参数作为证据提供给J.L，或者在RWKV-PEFT项目开一个issue后把参数作为证据提供给J.L，谁知道呢~
#### 依赖项目RWKV-PEFT现在需要`torch>=2.5.1`（同时别忘了安装GPU版本，如`2.5.1+cu121`，使用`pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu12`安装，如果你需要此版本的话）
#### 如需在torch==2.5.1+cu121环境下进行state tuning微调，需要使用此命令`pip install rwkv-fla==0.1.202502070725`将rwkv-fla安装至老版本，同时使用此命令`pip install triton==3.1.0`将triton安装为符合torch2.5.1的版本后即可进行state tuning
#### 为gpu服务器微调者提供了一个快捷环境安装脚本`gpu_server_installer.sh`，可以一键安装环境（包括wandb），不过模型和数据集还是要自己下载和安装
#### 提供了更多参数设置和使用的例子，查看`training_disha_example.sh`和`training_stat
e_tuning_example.sh`文件以确定如何使用`RWKV-PEFT-Simple`为你带来便捷！祝微调顺利~
---
# RWKV-PEFT-Simple

**此项目是为[JL-er](https://github.com/JL-er)的RWKV-PEFT项目提供的快捷微调包**
**注意：本项目默认你有一定的动手和学习能力**
#### 1. 准备工作（linux和windows通用）
* 此项目依赖于`RWKV-PEFT`仓库，必须下载后并覆盖到根目录，也就是`RWKV-PEFT-Simple`解压目录一起使用，建议跟随此步骤一步一步学习
* 首先进入依赖项目[RWKV-PEFT](https://github.com/Seikaijyu/RWKV-PEFT)下载RWKV-PEFT包，此时应该能得到名为`RWKV-PEFT-main.zip`的压缩包
* 将压缩包中的`RWKV-PEFT-main`文件夹解压到本地
* 下载本项目包，此时应该能得到名为`RWKV-PEFT-Simple-main.zip`的压缩包
* 双击进入`RWKV-PEFT-Simple-main.zip`压缩包，并进入压缩包内的`RWKV-PEFT-Simple-main`文件夹，使用鼠标拖动或者`ctrl+a`全选文件，拖动到`RWKV-PEFT-main`文件夹内
* 安装`wsl2`以及`Ubuntu 22.04.3 LTS`，这里不说明安装方法，有需要可以搜索
* 为Ubuntu系统中的python3解释器安装pip，这里不说明安装方法，有需要可以搜索
* 完成后找到之前的`RWKV-PEFT-main`文件夹，进入后右键鼠标选择`在终端中打开`然后在打开的终端中输入`wsl`进入`Ubuntu 22.04.3 LTS`系统
* 现在为Python安装必要的环境，默认Python在Ubuntu下应该为`python3`，并且如果你正确执行以上步骤，你应该能在根目录下找到`requirements.txt`文件，现在在此文件夹根目录使用以上步骤提供的方法打开Ubuntu终端，执行`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`即可安装环境，安装环境耗时较长，请耐心等待命令行停止输出
#### 2. 准备数据
* 你可以在`data/sample.jsonl`找到我提供的示例数据，训练数据的格式大致如此，你必须严格遵循格式才能进行训练
* 你可以根据自己想法做一些遵循此格式的训练数据，或者去国外的[huggingface](https://huggingface.co/)找一些公开的开源数据，国内也有类似的网站，比如[modelscope](https://modelscope.cn/)也可以找到一些开源数据
* 所有数据都应该放入`/data`目录中
#### 3. 数据分词
* 现在你得到了数据，进入之前在文件夹打开的ubuntu终端中，使用以下格式`./make_tokenize.sh {data目录中的文件名称，包括.jsonl} {训练的回合数}`进行数据分词，一般情况下建议直接使用`./make_tokenize.sh {data目录中的文件名称，包括.jsonl}`，同等于`./make_tokenize.sh {data目录中的文件名称，包括.jsonl} 1`
* 你可能会注意到，数据分词后会出现如`### The first ten factors of the five numbers nearby (±10):`这样的输出，下面跟着一批输出，这其实是根据此数据数量的10个值范围中推荐填入的`MICRO_BSZ`以及`EPOCH_STEPS`，使用此值可以让你在完整训练每一轮数据的同时加速训练时间（但同时也需要更多的显存），所以设置与否应该和你的显存挂钩
* 如果你的数据条数刚好支持你显存支持的更多`MICRO_BSZ`，我推荐你尽可能多开，如果你提供的数据条数不支持你多开或者你的显存不足以开到此`MICRO_BSZ`，我推荐你找到输出的`MINI_BSZ`并且你的显存`负担的起`的数据条数，然后对数据条数进行适当的增加或者删除并再次使用命令行进行数据分词，然后根据输出的推荐值调整参数中的`MICRO_BSZ`和`EPOCH_STEPS`即可
* 如果你的显卡甚至不支持将`MICRO_BSZ`设置为2，我推荐你依然使用输出的`MICRO_BSZ`值，找到一个较小的数值并修改`MINI_BSZ`参数，注意是设置`MINI_BSZ`参数而不是设置`MICRO_BSZ`参数，`MICRO_BSZ`设置为1即可，这样可以让训练数据的分布更均匀，所以不要设置`EPOCH_STEPS`，只是把一次性训练的数据分布到多次训练后更新梯度，所以这并不会降低训练训练时间
* 如果你希望使用`get`模式读取数据，则应该使用`./make_tokenize.sh {data目录中的文件名称，包括.jsonl}`进行数据分词，并且不需要设置epoch（设置了也不会停止）
* 在无法开启更大的`MICRO_BSZ`时，推荐开启`FLA`参数，此时`CTX_LEN`参数应该跟随分词脚本输出的`### max_length_power_of_two`项设置而不是`### max_length`，`FLA`启用时会加速`MICRO_BSZ`<=8的微调，但是要求`CTX_LEN`和`CHUNK_CTX`参数必须为n的二次幂，即，`"512, 1024，2048，4096，8192"`等，并且启用`infctx`进行超长上下文训练时也必须开启`FLA`参数
* 如果你无法训练数据（使用的数据token过长）你可以使用`./make_tokenize.sh {data目录中的文件名称，包括.jsonl} {训练回合数} {过滤的最大tokens}`以过滤掉超过最后一个参数设置的tokens的数据，可以降低显卡的训练负担
* 因为是示例，现在你可以输入`./make_tokenize.sh sample.jsonl`进行分词测试
#### 4. 调整参数
* 你已经完成了数据分词，现在使用文本编辑器（vscode或者其他文本编辑器）打开当前目录下的`training.sh`文件，里面的所有参数设置已经使用数据标注好，你应该调整其中的参数进行训练准备，`training.sh`已经设置默认值，可根据参数注释修改
* 调整好了参数后，在Ubuntu终端中运行`./training.sh`即可开始训练，只需要等待训练回合达到你期望的回合或者loss达到你的期望值即可
#### 5. 合并模型
* 现在你得到了微调后的模型，state tuning训练不需要这个步骤，因为它直接得到一个state额外state挂载到模型而不是权重合并文件，我这里只讲lora和pissa，bone训练
* 找到`merge.sh`文件并进入，调整对应训练的参数后，在Ubuntu命令行中运行`./merge.sh {训练的回合数}`如`./merge.sh 0`即可合并得到模型文件，文件存放在`/merge`文件夹内
#### 6. 运行测试
* 此时你已经完成了微调的所有步骤，现在只需要找到`/merge`文件夹内合并好的模型文件，放到[RWKV-Runner](https://github.com/josStorer/RWKV-Runner)或者[Ai00](https://github.com/Ai00-X/ai00_server)等RWKV推理项目中运行测试即可（这里我推荐RWKV-Runner）
* 测试时推荐多调整`Temperature`，`Top_P`，`Presence Penalty`，`Frequency Penalty`参数进行测试
## 如果在读完此教程后依然不理解如何使用，建议查看`training.sh`文件和`merge.sh`文件中的注释，能让你更快上手，也可以使用GPT阅读并询问GPT，弄清不理解的部分
### 祝您能成功调出自己喜欢的模型~

###### by Seikaijyu