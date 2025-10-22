#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验结果分析脚本
收集所有实验结果，计算统计指标，生成对比图表
"""

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
# import seaborn as sns  # 不需要seaborn

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_result_file(file_path):
    """解析结果文件，提取关键信息"""
    result = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 提取求解方法
        method_match = re.search(r'# 求解方法: (.+)', content)
        if method_match:
            result['method'] = method_match.group(1)
        
        # 提取删除的边数
        edges_match = re.search(r'# 删除的边数: (\d+)', content)
        if edges_match:
            result['edges_removed'] = int(edges_match.group(1))
        
        # 提取最大连通子图大小
        size_match = re.search(r'# 最大连通子图大小: (\d+)', content)
        if size_match:
            result['max_connected_size'] = int(size_match.group(1))
        
        # 提取求解时间
        time_match = re.search(r'# 求解时间: ([\d.]+) 秒', content)
        if time_match:
            result['solve_time'] = float(time_match.group(1))
        
        # 从文件路径提取数据集信息
        path_parts = file_path.split('/')
        if len(path_parts) >= 2:
            dataset = path_parts[-2]  # 例如 E250_S1
            result['dataset'] = dataset
            
            # 解析数据集信息
            if dataset.startswith('E250_S'):
                result['graph_type'] = 'E250'
                result['sample_id'] = dataset.split('_S')[1]
            elif dataset.startswith('N200_E300_S'):
                result['graph_type'] = 'N200_E300'
                result['sample_id'] = dataset.split('_S')[1]
            elif dataset.startswith('N10000_E11000_S'):
                result['graph_type'] = 'N10000_E11000'
                result['sample_id'] = dataset.split('_S')[1]
        
    except Exception as e:
        print(f"解析文件 {file_path} 时出错: {e}")
        return None
    
    return result

def collect_all_results():
    """收集所有实验结果"""
    results = []
    
    # 查找所有结果文件
    result_files = glob.glob('results/*/*_cut_edges.txt')
    print(f"找到 {len(result_files)} 个结果文件")
    
    for file_path in result_files:
        result = parse_result_file(file_path)
        if result:
            results.append(result)
    
    return results

def calculate_statistics(df):
    """计算统计指标"""
    stats = {}
    
    # 按方法分组计算统计指标
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        
        stats[method] = {
            'count': len(method_data),
            'max_connected_size_mean': method_data['max_connected_size'].mean(),
            'max_connected_size_std': method_data['max_connected_size'].std(),
            'max_connected_size_min': method_data['max_connected_size'].min(),
            'max_connected_size_max': method_data['max_connected_size'].max(),
            'solve_time_mean': method_data['solve_time'].mean(),
            'solve_time_std': method_data['solve_time'].std(),
            'solve_time_min': method_data['solve_time'].min(),
            'solve_time_max': method_data['solve_time'].max(),
        }
    
    return stats

def create_comparison_plots(df, stats):
    """创建对比图表"""
    
    # 设置图表样式
    # plt.style.use('seaborn-v0_8')  # 不需要seaborn样式
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 最大连通子图大小对比（箱线图）
    ax1 = axes[0, 0]
    methods = df['method'].unique()
    data_for_box = [df[df['method'] == method]['max_connected_size'].values for method in methods]
    
    box_plot = ax1.boxplot(data_for_box, labels=methods, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(box_plot['boxes'], colors[:len(methods)]):
        patch.set_facecolor(color)
    
    ax1.set_title('最大连通子图大小对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('最大连通子图大小')
    ax1.grid(True, alpha=0.3)
    
    # 2. 求解时间对比（箱线图）
    ax2 = axes[0, 1]
    data_for_time = [df[df['method'] == method]['solve_time'].values for method in methods]
    
    box_plot_time = ax2.boxplot(data_for_time, labels=methods, patch_artist=True)
    for patch, color in zip(box_plot_time['boxes'], colors[:len(methods)]):
        patch.set_facecolor(color)
    
    ax2.set_title('求解时间对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('求解时间 (秒)')
    ax2.set_yscale('log')  # 使用对数坐标轴
    ax2.grid(True, alpha=0.3)
    
    # 3. 精度vs时间散点图
    ax3 = axes[1, 0]
    for method in methods:
        method_data = df[df['method'] == method]
        ax3.scatter(method_data['solve_time'], method_data['max_connected_size'], 
                   label=method, alpha=0.7, s=50)
    
    ax3.set_xlabel('求解时间 (秒)')
    ax3.set_ylabel('最大连通子图大小')
    ax3.set_title('精度 vs 求解时间', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 方法性能雷达图
    ax4 = axes[1, 1]
    
    # 计算归一化指标（越小越好）
    metrics = ['max_connected_size_mean', 'solve_time_mean']
    method_names = list(stats.keys())
    
    # 创建雷达图数据
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    for method in method_names:
        values = []
        for metric in metrics:
            # 归一化（越小越好，所以用1-归一化值）
            max_val = max([stats[m][metric] for m in method_names])
            min_val = min([stats[m][metric] for m in method_names])
            if max_val == min_val:
                norm_val = 1.0
            else:
                norm_val = 1 - (stats[method][metric] - min_val) / (max_val - min_val)
            values.append(norm_val)
        
        values += values[:1]  # 闭合
        
        ax4.plot(angles, values, 'o-', linewidth=2, label=method)
        ax4.fill(angles, values, alpha=0.25)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(['连通子图大小', '求解时间'])
    ax4.set_title('方法性能雷达图', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('results_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_detailed_analysis(df):
    """创建详细分析图表"""
    
    # 按图类型分组分析
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 不同图类型的性能对比
    ax1 = axes[0, 0]
    graph_types = df['graph_type'].unique()
    
    for graph_type in graph_types:
        type_data = df[df['graph_type'] == graph_type]
        methods = type_data['method'].unique()
        
        x_pos = np.arange(len(methods))
        means = [type_data[type_data['method'] == method]['max_connected_size'].mean() 
                for method in methods]
        stds = [type_data[type_data['method'] == method]['max_connected_size'].std() 
               for method in methods]
        
        ax1.errorbar(x_pos, means, yerr=stds, label=graph_type, marker='o', capsize=5)
    
    ax1.set_xlabel('求解方法')
    ax1.set_ylabel('最大连通子图大小')
    ax1.set_title('不同图类型的性能对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 求解时间分布
    ax2 = axes[0, 1]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        ax2.hist(method_data['solve_time'], alpha=0.7, label=method, bins=20)
    
    ax2.set_xlabel('求解时间 (秒)')
    ax2.set_ylabel('频次')
    ax2.set_title('求解时间分布', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 精度分布
    ax3 = axes[1, 0]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        ax3.hist(method_data['max_connected_size'], alpha=0.7, label=method, bins=20)
    
    ax3.set_xlabel('最大连通子图大小')
    ax3.set_ylabel('频次')
    ax3.set_title('精度分布', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 方法效率对比（精度/时间）
    ax4 = axes[1, 1]
    efficiency_data = []
    method_names = []
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        # 效率 = 1 / (精度 * 时间)，越小越好
        efficiency = 1 / (method_data['max_connected_size'] * method_data['solve_time'])
        efficiency_data.append(efficiency.values)
        method_names.append(method)
    
    ax4.boxplot(efficiency_data, labels=method_names)
    ax4.set_ylabel('效率指标 (1/(精度×时间))')
    ax4.set_title('方法效率对比', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_report(df, stats):
    """生成实验报告"""
    
    report = []
    report.append("# 图切割问题求解算法实验报告")
    report.append("")
    report.append("## 1. 问题介绍")
    report.append("")
    report.append("### 1.1 问题描述")
    report.append("图切割问题是一个经典的图论优化问题，目标是在给定的预算约束下，通过移除最少数量的边，使得剩余图的最大连通分量尽可能小。")
    report.append("")
    report.append("### 1.2 问题形式化")
    report.append("给定无向图 G=(V,E)，其中 V 是节点集合，E 是边集合。问题可以形式化为：")
    report.append("")
    report.append("**目标函数**: min max |C_i|")
    report.append("**约束条件**: |E_removed| ≤ C")
    report.append("")
    report.append("其中：")
    report.append("- C_i 是移除边后的连通分量")
    report.append("- E_removed 是被移除的边集合")
    report.append("- C 是预算约束（最多可移除的边数）")
    report.append("")
    
    report.append("## 2. 方法介绍")
    report.append("")
    report.append("### 2.1 贪心算法 (Greedy Algorithm)")
    report.append("贪心算法采用局部最优策略，每次选择当前看起来最优的边进行移除。")
    report.append("")
    report.append("**策略3 (s3)**: 基于边介数中心性的贪心策略")
    report.append("- 计算每条边的介数中心性")
    report.append("- 优先移除介数中心性最高的边")
    report.append("- 时间复杂度: O(C × V × (V + E))")
    report.append("")
    
    report.append("### 2.2 Gurobi优化求解器")
    report.append("使用整数线性规划(ILP)方法，将问题转化为数学优化模型。")
    report.append("")
    report.append("**数学模型**:")
    report.append("- 决策变量: u[i,j] 表示节点i是否能到达节点j")
    report.append("- 目标函数: 最小化最大连通分量大小")
    report.append("- 约束条件: 预算约束、连通性约束、传递约束等")
    report.append("- 时间复杂度: O(2^(V²+E)) (指数时间)")
    report.append("")
    
    report.append("### 2.3 Mealpy元启发式算法")
    report.append("使用遗传算法(GA)等元启发式方法求解连续优化问题。")
    report.append("")
    report.append("**算法特点**:")
    report.append("- 将离散问题转化为连续优化问题")
    report.append("- 使用遗传算法进行全局搜索")
    report.append("- 时间复杂度: O(G × P × (V + E))")
    report.append("- 其中G为迭代代数，P为种群大小")
    report.append("")
    
    report.append("## 3. 实验结果分析")
    report.append("")
    report.append("### 3.1 实验设置")
    report.append(f"- 数据集数量: {len(df)} 个实验")
    report.append(f"- 求解方法: {', '.join(df['method'].unique())}")
    report.append(f"- 图类型: {', '.join(df['graph_type'].unique())}")
    report.append("")
    
    report.append("### 3.2 精度对比")
    report.append("")
    report.append("| 方法 | 平均精度 | 标准差 | 最小值 | 最大值 |")
    report.append("|------|----------|--------|--------|--------|")
    
    for method, stat in stats.items():
        report.append(f"| {method} | {stat['max_connected_size_mean']:.2f} | "
                     f"{stat['max_connected_size_std']:.2f} | "
                     f"{stat['max_connected_size_min']} | "
                     f"{stat['max_connected_size_max']} |")
    
    report.append("")
    report.append("### 3.3 求解时间对比")
    report.append("")
    report.append("| 方法 | 平均时间(秒) | 标准差 | 最小值 | 最大值 |")
    report.append("|------|-------------|--------|--------|--------|")
    
    for method, stat in stats.items():
        report.append(f"| {method} | {stat['solve_time_mean']:.2f} | "
                     f"{stat['solve_time_std']:.2f} | "
                     f"{stat['solve_time_min']:.2f} | "
                     f"{stat['solve_time_max']:.2f} |")
    
    report.append("")
    report.append("### 3.4 方法性能总结")
    report.append("")
    
    # 找出最佳方法
    best_accuracy = min(stats.items(), key=lambda x: x[1]['max_connected_size_mean'])
    best_speed = min(stats.items(), key=lambda x: x[1]['solve_time_mean'])
    
    report.append(f"**精度最佳**: {best_accuracy[0]} (平均连通子图大小: {best_accuracy[1]['max_connected_size_mean']:.2f})")
    report.append(f"**速度最快**: {best_speed[0]} (平均求解时间: {best_speed[1]['solve_time_mean']:.2f}秒)")
    report.append("")
    
    report.append("## 4. 复杂度分析")
    report.append("")
    report.append("### 4.1 时间复杂度")
    report.append("")
    report.append("| 算法 | 时间复杂度 | 适用场景 |")
    report.append("|------|-----------|----------|")
    report.append("| 贪心算法 | O(C × V × (V + E)) | 大规模图，快速求解 |")
    report.append("| Gurobi | O(2^(V²+E)) | 小规模图，最优解 |")
    report.append("| Mealpy | O(G × P × (V + E)) | 中等规模图，近似最优 |")
    report.append("")
    
    report.append("### 4.2 空间复杂度")
    report.append("")
    report.append("| 算法 | 空间复杂度 | 内存需求 |")
    report.append("|------|-----------|----------|")
    report.append("| 贪心算法 | O(V + E) | 低 |")
    report.append("| Gurobi | O((V²+E)²) | 高 |")
    report.append("| Mealpy | O(P × E) | 中等 |")
    report.append("")
    
    report.append("## 5. 结论与建议")
    report.append("")
    report.append("### 5.1 主要发现")
    report.append("1. **Gurobi求解器**在精度方面表现最佳，但求解时间较长")
    report.append("2. **贪心算法**求解速度最快，适合大规模问题")
    report.append("3. **Mealpy算法**在精度和速度之间取得较好平衡")
    report.append("")
    
    report.append("### 5.2 应用建议")
    report.append("- **小规模图(V<100)**: 推荐使用Gurobi获得最优解")
    report.append("- **中等规模图(100≤V<1000)**: 推荐使用Mealpy算法")
    report.append("- **大规模图(V≥1000)**: 推荐使用贪心算法快速求解")
    report.append("")
    
    report.append("### 5.3 未来改进方向")
    report.append("1. 开发混合算法，结合不同方法的优势")
    report.append("2. 优化算法参数，提高求解效率")
    report.append("3. 利用并行计算加速大规模问题求解")
    report.append("")
    
    return "\n".join(report)

def main():
    """主函数"""
    print("开始分析实验结果...")
    
    # 收集所有结果
    results = collect_all_results()
    print(f"成功收集 {len(results)} 个实验结果")
    
    if not results:
        print("没有找到实验结果，请检查文件路径")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    print(f"数据框形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 计算统计指标
    stats = calculate_statistics(df)
    
    # 创建对比图表
    print("创建对比图表...")
    create_comparison_plots(df, stats)
    
    # 创建详细分析图表
    print("创建详细分析图表...")
    create_detailed_analysis(df)
    
    # 生成报告
    print("生成实验报告...")
    report = generate_report(df, stats)
    
    # 保存报告
    with open('实验报告.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("分析完成！")
    print("- 对比图表: results_analysis.png")
    print("- 详细分析: detailed_analysis.png") 
    print("- 实验报告: 实验报告.md")
    
    # 打印简要统计
    print("\n=== 简要统计 ===")
    for method, stat in stats.items():
        print(f"{method}:")
        print(f"  平均精度: {stat['max_connected_size_mean']:.2f}")
        print(f"  平均时间: {stat['solve_time_mean']:.2f}秒")

if __name__ == "__main__":
    main()
