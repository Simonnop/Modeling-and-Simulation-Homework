# 循环从1到100, 以步长50遍历每个区间的起始值
# for start in $(seq 1 100 100); do
#   # 计算当前区间的结束值（起始值加9）
#   end=$((start + 99))
#   # 后台运行 solver.py 脚本传递起始和结束参数及样本参数, 重定向输出到 /dev/null
#   nohup python solver.py --sample_E 300 --sample_S_begin $start --sample_S_end $end --sample_C 30 > /dev/null 2>&1 &
#   nohup python solver.py --sample_E 400 --sample_S_begin $start --sample_S_end $end --sample_C 40 > /dev/null 2>&1 &
#   nohup python solver.py --sample_E 500 --sample_S_begin $start --sample_S_end $end --sample_C 50 > /dev/null 2>&1 &
# done

nohup python solver2.py --sample_N 10000 --sample_E 11000 --sample_S_begin 4 --sample_S_end 4 --sample_C 1100 > /dev/null 2>&1 &

nohup python solver2.py --sample_N 10000 --sample_E 12500 --sample_S_begin 3 --sample_S_end 3 --sample_C 1250 > /dev/null 2>&1 &

nohup python solver2.py --sample_N 10000 --sample_E 15000 --sample_S_begin 5 --sample_S_end 5 --sample_C 1500 > /dev/null 2>&1 &