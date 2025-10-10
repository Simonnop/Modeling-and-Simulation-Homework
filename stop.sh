# 查找所有后台运行的以 solver.py 为命令的进程并终止
ps aux | grep '[s]olver.py' | awk '{print $2}' | xargs -r kill -9
echo "已关闭所有 solver.py 的后台进程。"
