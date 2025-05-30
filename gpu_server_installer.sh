# 在GPU服务器上快捷安装依赖项的脚本
# 使用方式：在linux命令行或者wsl命令行输入 ./gpu_server_installer.sh
# 回车即可
git clone https://github.com/Seikaijyu/RWKV-PEFT
git clone https://github.com/Seikaijyu/RWKV-PEFT-Simple
mv RWKV-PEFT/* RWKV-PEFT-Simple/
rm -rf ./RWKV-PEFT
cd RWKV-PEFT-Simple
chmod +rwx ./*
pip install -r requirements.txt
pip install wandb