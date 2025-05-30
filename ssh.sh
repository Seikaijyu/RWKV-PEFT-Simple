# ssh创建持久化远程会话确保在退出ssh时不会关闭训练的方式，这是一个例子
# 创建会话
tmux new -s session_name
# 接入会话
tmux attach -t session_name
# 杀死会话
tmux kill-session -t session_name
# 列出会话
tmux ls
# 分离会话
tmux detach