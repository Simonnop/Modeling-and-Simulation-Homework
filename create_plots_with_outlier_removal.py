#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建排除异常值的实验分析图表
使用IQR方法识别和排除异常值
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

def remove_outliers_iqr(data, column, factor=1.5):
    """使用IQR方法移除异常值"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    # 记录异常值信息
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    print(f"Removing {len(outliers)} outliers from {column} column")
    if len(outliers) > 0:
        print(f"  Outlier range: [{outliers[column].min():.2f}, {outliers[column].max():.2f}]")
        print(f"  Normal range: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # 返回去除异常值后的数据
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """使用Z-score方法移除异常值"""
    z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
    
    outliers = data[z_scores > threshold]
    print(f"Removing {len(outliers)} outliers from {column} column (Z-score > {threshold})")
    if len(outliers) > 0:
        print(f"  Outlier range: [{outliers[column].min():.2f}, {outliers[column].max():.2f}]")
    
    return data[z_scores <= threshold]

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

def create_plots_with_outlier_removal(df, stats):
    """创建排除异常值的对比图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Maximum Connected Component Size Comparison (Box Plot)
    ax1 = axes[0, 0]
    methods = df['method'].unique()
    
    # 为每个方法移除异常值
    data_for_box = []
    labels = []
    for method in methods:
        method_data = df[df['method'] == method]
        # 移除连通子图大小的异常值
        cleaned_data = remove_outliers_iqr(method_data, 'max_connected_size', factor=1.5)
        if len(cleaned_data) > 0:
            data_for_box.append(cleaned_data['max_connected_size'].values)
            labels.append(f"{method}\n(n={len(cleaned_data)})")
        else:
            # 如果没有数据，使用原始数据
            data_for_box.append(method_data['max_connected_size'].values)
            labels.append(f"{method}\n(n={len(method_data)})")
    
    box_plot = ax1.boxplot(data_for_box, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(box_plot['boxes'], colors[:len(methods)]):
        patch.set_facecolor(color)
    
    ax1.set_title('Maximum Connected Component Size Comparison\n(Outliers Removed)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Maximum Connected Component Size')
    ax1.grid(True, alpha=0.3)
    
    # 2. Solving Time Comparison (Box Plot)
    ax2 = axes[0, 1]
    data_for_time = []
    labels_time = []
    for method in methods:
        method_data = df[df['method'] == method]
        # 移除求解时间的异常值
        cleaned_data = remove_outliers_iqr(method_data, 'solve_time', factor=1.5)
        if len(cleaned_data) > 0:
            data_for_time.append(cleaned_data['solve_time'].values)
            labels_time.append(f"{method}\n(n={len(cleaned_data)})")
        else:
            data_for_time.append(method_data['solve_time'].values)
            labels_time.append(f"{method}\n(n={len(method_data)})")
    
    box_plot_time = ax2.boxplot(data_for_time, labels=labels_time, patch_artist=True)
    for patch, color in zip(box_plot_time['boxes'], colors[:len(methods)]):
        patch.set_facecolor(color)
    
    ax2.set_title('Solving Time Comparison\n(Outliers Removed)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Solving Time (seconds)')
    ax2.set_yscale('log')  # Use logarithmic scale
    ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy vs Time Scatter Plot (with outlier removal)
    ax3 = axes[1, 0]
    for method in methods:
        method_data = df[df['method'] == method]
        # 移除异常值
        cleaned_data = remove_outliers_iqr(method_data, 'max_connected_size', factor=1.5)
        cleaned_data = remove_outliers_iqr(cleaned_data, 'solve_time', factor=1.5)
        
        if len(cleaned_data) > 0:
            ax3.scatter(cleaned_data['solve_time'], cleaned_data['max_connected_size'], 
                       label=f"{method} (n={len(cleaned_data)})", alpha=0.7, s=50)
    
    ax3.set_xlabel('Solving Time (seconds)')
    ax3.set_ylabel('Maximum Connected Component Size')
    ax3.set_title('Accuracy vs Solving Time\n(Outliers Removed)', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Method Performance Radar Chart (with outlier removal)
    ax4 = axes[1, 1]
    
    # Calculate normalized metrics using cleaned data
    metrics = ['max_connected_size_mean', 'solve_time_mean']
    method_names = list(stats.keys())
    
    # Create radar chart data
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    for method in method_names:
        method_data = df[df['method'] == method]
        # 移除异常值后重新计算统计指标
        cleaned_data = remove_outliers_iqr(method_data, 'max_connected_size', factor=1.5)
        cleaned_data = remove_outliers_iqr(cleaned_data, 'solve_time', factor=1.5)
        
        if len(cleaned_data) > 0:
            values = []
            for metric in metrics:
                if metric == 'max_connected_size_mean':
                    metric_value = cleaned_data['max_connected_size'].mean()
                else:
                    metric_value = cleaned_data['solve_time'].mean()
                
                # 获取所有方法的该指标值进行归一化
                all_values = []
                for m in method_names:
                    m_data = df[df['method'] == m]
                    m_cleaned = remove_outliers_iqr(m_data, 'max_connected_size', factor=1.5)
                    m_cleaned = remove_outliers_iqr(m_cleaned, 'solve_time', factor=1.5)
                    if len(m_cleaned) > 0:
                        if metric == 'max_connected_size_mean':
                            all_values.append(m_cleaned['max_connected_size'].mean())
                        else:
                            all_values.append(m_cleaned['solve_time'].mean())
                
                if len(all_values) > 0:
                    max_val = max(all_values)
                    min_val = min(all_values)
                    if max_val == min_val:
                        norm_val = 1.0
                    else:
                        norm_val = 1 - (metric_value - min_val) / (max_val - min_val)
                    values.append(norm_val)
                else:
                    values.append(0.5)
            else:
                values = [0.5, 0.5]
        else:
            values = [0.5, 0.5]
        
        values += values[:1]  # Close the circle
        
        ax4.plot(angles, values, 'o-', linewidth=2, label=method)
        ax4.fill(angles, values, alpha=0.25)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(['Connected Component Size', 'Solving Time'])
    ax4.set_title('Method Performance Radar Chart\n(Outliers Removed)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('results_analysis_no_outliers.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_detailed_analysis_with_outlier_removal(df):
    """创建排除异常值的详细分析图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Performance Comparison by Graph Type (with outlier removal)
    ax1 = axes[0, 0]
    graph_types = df['graph_type'].unique()
    
    for graph_type in graph_types:
        type_data = df[df['graph_type'] == graph_type]
        methods = type_data['method'].unique()
        
        x_pos = np.arange(len(methods))
        means = []
        stds = []
        labels = []
        
        for method in methods:
            method_data = type_data[type_data['method'] == method]
            # 移除异常值
            cleaned_data = remove_outliers_iqr(method_data, 'max_connected_size', factor=1.5)
            
            if len(cleaned_data) > 0:
                means.append(cleaned_data['max_connected_size'].mean())
                stds.append(cleaned_data['max_connected_size'].std())
                labels.append(f"{method}\n(n={len(cleaned_data)})")
            else:
                means.append(method_data['max_connected_size'].mean())
                stds.append(method_data['max_connected_size'].std())
                labels.append(f"{method}\n(n={len(method_data)})")
        
        ax1.errorbar(x_pos, means, yerr=stds, label=graph_type, marker='o', capsize=5)
    
    ax1.set_xlabel('Solving Method')
    ax1.set_ylabel('Maximum Connected Component Size')
    ax1.set_title('Performance Comparison by Graph Type\n(Outliers Removed)', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Solving Time Distribution (with outlier removal)
    ax2 = axes[0, 1]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        # 移除异常值
        cleaned_data = remove_outliers_iqr(method_data, 'solve_time', factor=1.5)
        
        if len(cleaned_data) > 0:
            ax2.hist(cleaned_data['solve_time'], alpha=0.7, label=f"{method} (n={len(cleaned_data)})", bins=20)
        else:
            ax2.hist(method_data['solve_time'], alpha=0.7, label=f"{method} (n={len(method_data)})", bins=20)
    
    ax2.set_xlabel('Solving Time (seconds)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Solving Time Distribution\n(Outliers Removed)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy Distribution (with outlier removal)
    ax3 = axes[1, 0]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        # 移除异常值
        cleaned_data = remove_outliers_iqr(method_data, 'max_connected_size', factor=1.5)
        
        if len(cleaned_data) > 0:
            ax3.hist(cleaned_data['max_connected_size'], alpha=0.7, label=f"{method} (n={len(cleaned_data)})", bins=20)
        else:
            ax3.hist(method_data['max_connected_size'], alpha=0.7, label=f"{method} (n={len(method_data)})", bins=20)
    
    ax3.set_xlabel('Maximum Connected Component Size')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Accuracy Distribution\n(Outliers Removed)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Method Efficiency Comparison (with outlier removal)
    ax4 = axes[1, 1]
    efficiency_data = []
    method_names = []
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        # 移除异常值
        cleaned_data = remove_outliers_iqr(method_data, 'max_connected_size', factor=1.5)
        cleaned_data = remove_outliers_iqr(cleaned_data, 'solve_time', factor=1.5)
        
        if len(cleaned_data) > 0:
            # Efficiency = 1 / (accuracy * time), lower is better
            efficiency = 1 / (cleaned_data['max_connected_size'] * cleaned_data['solve_time'])
            efficiency_data.append(efficiency.values)
            method_names.append(f"{method}\n(n={len(cleaned_data)})")
        else:
            efficiency = 1 / (method_data['max_connected_size'] * method_data['solve_time'])
            efficiency_data.append(efficiency.values)
            method_names.append(f"{method}\n(n={len(method_data)})")
    
    ax4.boxplot(efficiency_data, labels=method_names)
    ax4.set_ylabel('Efficiency Metric (1/(Accuracy×Time))')
    ax4.set_title('Method Efficiency Comparison\n(Outliers Removed)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detailed_analysis_no_outliers.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_outlier_analysis_plots(df):
    """创建异常值分析图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 原始数据 vs 去除异常值后的数据对比
    ax1 = axes[0, 0]
    methods = df['method'].unique()
    
    original_means = []
    cleaned_means = []
    method_labels = []
    
    for method in methods:
        method_data = df[df['method'] == method]
        original_mean = method_data['max_connected_size'].mean()
        
        cleaned_data = remove_outliers_iqr(method_data, 'max_connected_size', factor=1.5)
        if len(cleaned_data) > 0:
            cleaned_mean = cleaned_data['max_connected_size'].mean()
        else:
            cleaned_mean = original_mean
        
        original_means.append(original_mean)
        cleaned_means.append(cleaned_mean)
        method_labels.append(method)
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax1.bar(x - width/2, original_means, width, label='Original Data', alpha=0.7)
    ax1.bar(x + width/2, cleaned_means, width, label='Outliers Removed', alpha=0.7)
    
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Average Connected Component Size')
    ax1.set_title('Original vs Cleaned Data Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 异常值检测可视化
    ax2 = axes[0, 1]
    for method in methods:
        method_data = df[df['method'] == method]
        
        # 计算IQR
        Q1 = method_data['max_connected_size'].quantile(0.25)
        Q3 = method_data['max_connected_size'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 标记异常值
        outliers = method_data[(method_data['max_connected_size'] < lower_bound) | 
                              (method_data['max_connected_size'] > upper_bound)]
        normal = method_data[(method_data['max_connected_size'] >= lower_bound) & 
                           (method_data['max_connected_size'] <= upper_bound)]
        
        ax2.scatter(normal['max_connected_size'], [method] * len(normal), 
                   alpha=0.6, s=30, label=f'{method} (normal)' if method == methods[0] else "")
        ax2.scatter(outliers['max_connected_size'], [method] * len(outliers), 
                   alpha=0.8, s=50, color='red', marker='x', 
                   label=f'{method} (outliers)' if method == methods[0] else "")
    
    ax2.set_xlabel('Maximum Connected Component Size')
    ax2.set_ylabel('Method')
    ax2.set_title('Outlier Detection Visualization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 异常值统计
    ax3 = axes[1, 0]
    outlier_counts = []
    total_counts = []
    
    for method in methods:
        method_data = df[df['method'] == method]
        cleaned_data = remove_outliers_iqr(method_data, 'max_connected_size', factor=1.5)
        outlier_count = len(method_data) - len(cleaned_data)
        outlier_counts.append(outlier_count)
        total_counts.append(len(method_data))
    
    x = np.arange(len(methods))
    ax3.bar(x, outlier_counts, alpha=0.7, color='red')
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Number of Outliers')
    ax3.set_title('Outlier Count by Method')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (outlier, total) in enumerate(zip(outlier_counts, total_counts)):
        ax3.text(i, outlier + 0.1, f'{outlier}/{total}', ha='center', va='bottom')
    
    # 4. 异常值比例
    ax4 = axes[1, 1]
    outlier_ratios = [outlier/total for outlier, total in zip(outlier_counts, total_counts)]
    
    ax4.bar(x, outlier_ratios, alpha=0.7, color='orange')
    ax4.set_xlabel('Method')
    ax4.set_ylabel('Outlier Ratio')
    ax4.set_title('Outlier Ratio by Method')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods)
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, ratio in enumerate(outlier_ratios):
        ax4.text(i, ratio + 0.01, f'{ratio:.2%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('outlier_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """主函数"""
    print("Creating plots with outlier removal...")
    
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
    
    # 创建排除异常值的对比图表
    print("Creating comparison plots with outlier removal...")
    create_plots_with_outlier_removal(df, stats)
    
    # 创建排除异常值的详细分析图表
    print("Creating detailed analysis plots with outlier removal...")
    create_detailed_analysis_with_outlier_removal(df)
    
    # 创建异常值分析图表
    print("Creating outlier analysis plots...")
    create_outlier_analysis_plots(df)
    
    print("Analysis with outlier removal completed!")
    print("- Comparison plots: results_analysis_no_outliers.png")
    print("- Detailed analysis: detailed_analysis_no_outliers.png")
    print("- Outlier analysis: outlier_analysis.png")
    
    # 打印异常值统计
    print("\n=== Outlier Statistics ===")
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        original_count = len(method_data)
        
        # 连通子图大小异常值
        cleaned_size = remove_outliers_iqr(method_data, 'max_connected_size', factor=1.5)
        size_outliers = original_count - len(cleaned_size)
        
        # 求解时间异常值
        cleaned_time = remove_outliers_iqr(method_data, 'solve_time', factor=1.5)
        time_outliers = original_count - len(cleaned_time)
        
        print(f"{method}:")
        print(f"  Total samples: {original_count}")
        print(f"  Size outliers: {size_outliers} ({size_outliers/original_count:.1%})")
        print(f"  Time outliers: {time_outliers} ({time_outliers/original_count:.1%})")

if __name__ == "__main__":
    main()
