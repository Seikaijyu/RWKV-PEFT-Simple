# RWKV-PEFT-Simple

**此项目是为[RWKV-PEFT](https://github.com/JL-er/RWKV-PEFT)提供的快捷微调包**

**注意：本项目默认你有一定的动手和学习能力**

#### 1. 准备工作（linux和windows通用）
* 进入以上RWKV-PEFT的github链接下载RWKV-PEFT包，此时应该能得到名为`RWKV-PEFT-main.zip`的压缩包
* 将压缩包中的`RWKV-PEFT-main`文件夹解压到本地
* 下载本项目包，此时应该能得到名为`RWKV-PEFT-Simple-main.zip`的压缩包
* 双击进入`RWKV-PEFT-Simple-main.zip`压缩包，并进入压缩包内的`RWKV-PEFT-Simple-main`文件夹，使用鼠标拖动或者`ctrl+a`全选文件，拖动到`RWKV-PEFT-main`文件夹内
* 安装`wsl2`以及`Ubuntu 22.04.3 LTS`，这里不说明安装方法，有需要可以搜索
* 为Ubuntu系统中的python3解释器安装pip以及必要的环境，这里不说明安装方法，有需要可以搜索
* 完成后找到之前的`RWKV-PEFT-main`文件夹，进入后右键鼠标选择`在终端中打开`然后在打开的终端中输入`wsl`进入`Ubuntu 22.04.3 LTS`系统
#### 2. 准备数据
* 你可以在`data/sample.jsonl`找到我提供的示例数据，训练数据的格式大致如此，你必须严格遵循格式才能进行训练
* 你可以根据自己想法做一些遵循此格式的训练数据，或者去国外的[huggingface](https://huggingface.co/)找一些公开的开源数据，国内也有类似的网站，比如[modelscope](https://modelscope.cn/)也可以找到一些开源数据
* 所有数据都应该放入`/data`目录中
#### 3. 数据分词
* 现在你得到了数据，进入之前在文件夹打开的ubuntu终端中，使用以下格式`./make_tokenize.sh {data目录中的文件名称，包括.jsonl} {训练的回合数} {上下文长度}`进行数据分词
* 因为是示例，现在你可以输入`./make_tokenize.sh sample.jsonl 4 512` 进行分词测试
#### 4. 调整参数
* 你已经完成了数据分词，现在使用文本编辑器（vscode或者其他文本编辑器）打开当前目录下的`training.sh`文件，里面的所有参数设置已经使用数据标注好，你应该调整其中的参数进行训练准备，参数调整说明会在其他地方（可能是github wiki或者知乎）详细说明，这里就不再多提
* 调整好了参数后，在Ubuntu终端中运行`./training.sh`即可开始训练，只需要等待训练回合达到你期望的回合或者loss达到你的期望值即可
#### 5. 合并模型
* 现在你得到了微调后的模型，lisa训练不需要这个步骤，因为它直接得到一个新的模型而不是权重合并文件，我这里只讲lora和pissa训练
* 找到`merge.sh`文件并进入，调整对应训练的参数后，在Ubuntu命令行中运行`./merge.sh {训练的回合数}`如`./merge.sh 0`即可合并得到模型文件，文件存放在`/merge`文件夹内
#### 6. 运行测试
* 此时你已经完成了微调的所有步骤，现在只需要找到`/merge`文件夹内合并好的模型文件，放到[RWKV-Runner](https://github.com/josStorer/RWKV-Runner)或者[Ai00](https://github.com/Ai00-X/ai00_server)等RWKV推理项目中运行测试即可（这里我推荐RWKV-Runner）
* 测试时推荐多调整`Temperature`，`Top_P`，`Presence Penalty`，`Frequency Penalty`参数进行测试
## 如果在读完此教程后依然不理解如何使用，建议查看`training.sh`文件和`merge.sh`文件中的注释，能让你更快上手，也可以使用GPT阅读并询问GPT，弄清不理解的部分
### 祝您能成功调出自己喜欢的模型~

###### by Seikaijyu