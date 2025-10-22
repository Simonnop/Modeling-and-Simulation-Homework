#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建英文版本的实验分析图表
解决中文字体显示问题
"""

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 设置英文图表样式
plt.rcParams['font.family'] = 'DejaVu Sans'
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
        print(f"Error parsing file {file_path}: {e}")
        return None
    
    return result

def collect_all_results():
    """收集所有实验结果"""
    results = []
    
    # 查找所有结果文件
    result_files = glob.glob('results/*/*_cut_edges.txt')
    print(f"Found {len(result_files)} result files")
    
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

def create_english_plots(df, stats):
    """创建英文版本的对比图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Maximum Connected Component Size Comparison (Box Plot)
    ax1 = axes[0, 0]
    methods = df['method'].unique()
    data_for_box = [df[df['method'] == method]['max_connected_size'].values for method in methods]
    
    box_plot = ax1.boxplot(data_for_box, labels=methods, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(box_plot['boxes'], colors[:len(methods)]):
        patch.set_facecolor(color)
    
    ax1.set_title('Maximum Connected Component Size Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Maximum Connected Component Size')
    ax1.grid(True, alpha=0.3)
    
    # 2. Solving Time Comparison (Box Plot)
    ax2 = axes[0, 1]
    data_for_time = [df[df['method'] == method]['solve_time'].values for method in methods]
    
    box_plot_time = ax2.boxplot(data_for_time, labels=methods, patch_artist=True)
    for patch, color in zip(box_plot_time['boxes'], colors[:len(methods)]):
        patch.set_facecolor(color)
    
    ax2.set_title('Solving Time Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Solving Time (seconds)')
    ax2.set_yscale('log')  # Use logarithmic scale
    ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy vs Time Scatter Plot
    ax3 = axes[1, 0]
    for method in methods:
        method_data = df[df['method'] == method]
        ax3.scatter(method_data['solve_time'], method_data['max_connected_size'], 
                   label=method, alpha=0.7, s=50)
    
    ax3.set_xlabel('Solving Time (seconds)')
    ax3.set_ylabel('Maximum Connected Component Size')
    ax3.set_title('Accuracy vs Solving Time', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Method Performance Radar Chart
    ax4 = axes[1, 1]
    
    # Calculate normalized metrics (lower is better)
    metrics = ['max_connected_size_mean', 'solve_time_mean']
    method_names = list(stats.keys())
    
    # Create radar chart data
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    for method in method_names:
        values = []
        for metric in metrics:
            # Normalize (lower is better, so use 1-normalized value)
            max_val = max([stats[m][metric] for m in method_names])
            min_val = min([stats[m][metric] for m in method_names])
            if max_val == min_val:
                norm_val = 1.0
            else:
                norm_val = 1 - (stats[method][metric] - min_val) / (max_val - min_val)
            values.append(norm_val)
        
        values += values[:1]  # Close the circle
        
        ax4.plot(angles, values, 'o-', linewidth=2, label=method)
        ax4.fill(angles, values, alpha=0.25)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(['Connected Component Size', 'Solving Time'])
    ax4.set_title('Method Performance Radar Chart', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('english_results_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_detailed_english_analysis(df):
    """创建英文版本的详细分析图表"""
    
    # Analysis by graph type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Performance Comparison by Graph Type
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
    
    ax1.set_xlabel('Solving Method')
    ax1.set_ylabel('Maximum Connected Component Size')
    ax1.set_title('Performance Comparison by Graph Type', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Solving Time Distribution
    ax2 = axes[0, 1]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        ax2.hist(method_data['solve_time'], alpha=0.7, label=method, bins=20)
    
    ax2.set_xlabel('Solving Time (seconds)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Solving Time Distribution', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy Distribution
    ax3 = axes[1, 0]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        ax3.hist(method_data['max_connected_size'], alpha=0.7, label=method, bins=20)
    
    ax3.set_xlabel('Maximum Connected Component Size')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Accuracy Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Method Efficiency Comparison (Accuracy/Time)
    ax4 = axes[1, 1]
    efficiency_data = []
    method_names = []
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        # Efficiency = 1 / (accuracy * time), lower is better
        efficiency = 1 / (method_data['max_connected_size'] * method_data['solve_time'])
        efficiency_data.append(efficiency.values)
        method_names.append(method)
    
    ax4.boxplot(efficiency_data, labels=method_names)
    ax4.set_ylabel('Efficiency Metric (1/(Accuracy×Time))')
    ax4.set_title('Method Efficiency Comparison', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('english_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """主函数"""
    print("Creating English version analysis plots...")
    
    # 收集所有结果
    results = collect_all_results()
    print(f"Successfully collected {len(results)} experimental results")
    
    if not results:
        print("No experimental results found, please check file paths")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 计算统计指标
    stats = calculate_statistics(df)
    
    # 创建英文对比图表
    print("Creating English comparison plots...")
    create_english_plots(df, stats)
    
    # 创建英文详细分析图表
    print("Creating English detailed analysis plots...")
    create_detailed_english_analysis(df)
    
    print("English analysis completed!")
    print("- Comparison plots: english_results_analysis.png")
    print("- Detailed analysis: english_detailed_analysis.png")
    
    # 打印简要统计
    print("\n=== Summary Statistics ===")
    for method, stat in stats.items():
        print(f"{method}:")
        print(f"  Average Accuracy: {stat['max_connected_size_mean']:.2f}")
        print(f"  Average Time: {stat['solve_time_mean']:.2f} seconds")

if __name__ == "__main__":
    main()
