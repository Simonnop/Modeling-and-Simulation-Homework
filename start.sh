# 循环从1到100, 以步长10遍历每个区间的起始值
for start in $(seq 1 20 100); do
  # 计算当前区间的结束值（起始值加9）
  end=$((start + 19))
  # 后台运行 solver.py 脚本传递起始和结束参数及样本参数, 重定向输出到 /dev/null
  nohup python solver.py --sample_E 250 --sample_S_begin $start --sample_S_end $end > /dev/null 2>&1 &
done