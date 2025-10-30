import os
import re
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt


# 解析单个结果文件，返回(method, time_seconds, max_component)
# 中文注释：优先从文件头部注释读取方法、时间、最大连通子图；若缺失再从文件名回退解析
def parse_result_file(file_path: str) -> Tuple[str, float, int]:
    method = None
    time_seconds = None
    max_component = None

    # 中文注释：尝试从文件内容解析
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 中文注释：仅扫描前若干行注释，避免读取整文件
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith('#'):
                    # 中文注释：遇到非注释提前结束
                    break
                if '求解方法' in line:
                    # 例：# 求解方法: mealpy-ga
                    m = re.search(r':\s*(.+)$', line)
                    if m:
                        method = m.group(1).strip()
                elif '最大连通子图大小' in line:
                    m = re.search(r':\s*([0-9]+)', line)
                    if m:
                        try:
                            max_component = int(m.group(1))
                        except ValueError:
                            pass
                elif '求解时间' in line:
                    # 例：# 求解时间: 12.34 秒
                    m = re.search(r':\s*([0-9]+(?:\.[0-9]+)?)', line)
                    if m:
                        try:
                            time_seconds = float(m.group(1))
                        except ValueError:
                            pass
    except Exception:
        # 中文注释：读取失败时，保持为 None，稍后尝试从文件名解析
        pass

    # 中文注释：从文件名回退解析（形如 greedy-s3-81_cut_edges.txt / mealpy-ga-144_cut_edges.txt）
    file_name = os.path.basename(file_path)
    # 中文注释：优先匹配包含 “方法-数值_cut_edges.txt” 的模式
    m_name = re.match(r'(.+?)-([0-9]+)_cut_edges\.txt$', file_name)
    if m_name:
        if method is None:
            method = m_name.group(1)
        if max_component is None:
            try:
                max_component = int(m_name.group(2))
            except ValueError:
                pass
    else:
        # 中文注释：匹配无数值的模式，如 init_cut_edges.txt
        m_name2 = re.match(r'(.+?)_cut_edges\.txt$', file_name)
        if m_name2 and method is None:
            method = m_name2.group(1)

    return method, time_seconds, max_component


# 中文注释：聚合指定 N,E 下所有 S 的方法统计信息
def aggregate_results(results_root: str, N: int, E: int, allow_methods: List[str] = None) -> Dict[str, Dict[str, List[float]]]:
    agg: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"time": [], "quality": [], "efficiency": []})
    target_prefix = f"N{N}_E{E}_S"

    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"Results directory not found: {results_root}")

    # 中文注释：遍历所有匹配的 S 子目录
    for entry in os.listdir(results_root):
        if not entry.startswith(target_prefix):
            continue
        subdir = os.path.join(results_root, entry)
        if not os.path.isdir(subdir):
            continue

        # 中文注释：基线简化为 N，无需读取样本
        base_quality = N

        # 中文注释：扫描该 S 下的所有 *_cut_edges.txt
        for fname in os.listdir(subdir):
            if not fname.endswith('_cut_edges.txt'):
                continue
            fpath = os.path.join(subdir, fname)
            method, t_sec, q_val = parse_result_file(fpath)
            # 中文注释：方法名缺失则跳过
            if not method:
                continue
            # 中文注释：若传入方法白名单，则过滤
            if allow_methods is not None and len(allow_methods) > 0:
                if method not in allow_methods:
                    continue
            # 中文注释：记录可用数据
            if t_sec is not None:
                agg[method]["time"].append(t_sec)
            if q_val is not None:
                q = float(q_val)
                agg[method]["quality"].append(q)
                # 中文注释：效率=(N-裁剪后)/时间；需要时间
                if t_sec is not None and t_sec > 0:
                    eff = max(base_quality - q, 0.0) / t_sec
                    agg[method]["efficiency"].append(eff)

    return agg


# 中文注释：根据聚合结果绘制平均时间与平均质量（最大连通子图，越小越好）
def plot_aggregates(agg: Dict[str, Dict[str, List[float]]], out_dir: str, N: int, E: int) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 中文注释：Nature 风格 rcParams（接近 Nature 期刊图尺寸与线宽）
    def set_nature_rcparams():
        try:
            plt.style.use('seaborn-v0_8-whitegrid')  # 中文注释：轻网格背景，便于阅读
        except Exception:
            try:
                plt.style.use('seaborn-whitegrid')
            except Exception:
                pass
        mpl.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 8,
            'axes.titlesize': 9,
            'axes.labelsize': 8,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
            'axes.linewidth': 0.8,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.width': 0.6,
            'ytick.minor.width': 0.6,
            'grid.linewidth': 0.5,
            'grid.alpha': 0.15,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        })

    # 中文注释：坐标轴去装饰、美化（去除上右脊、仅 y 轴网格）
    def beautify_axes(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.15)
        ax.grid(axis='x', visible=False)

    # 中文注释：此处不使用对数刻度与科学计数法，保持线性与实际数值

    # 中文注释：动态尺寸（英寸），单栏 ~3.5in，最多双栏 7.2in
    def dynamic_figsize(n_items: int, base: float = 3.5, per_item: float = 0.18, max_w: float = 7.2, h: float = 2.6) -> Tuple[float, float]:
        w = per_item * max(n_items, 1) + (base - 1.5)
        w = min(max(w, base), max_w)
        return (w, h)

    # 中文注释：统一颜色映射（色盲安全、接近 Nature 常用色系）
    palette = [
        '#4C72B0',  # blue
        '#DD8452',  # orange
        '#55A868',  # green
        '#C44E52',  # red
        '#8172B2',  # purple
        '#937860',  # brown
        '#DA8BC3',  # pink
        '#8C8C8C',  # gray
        '#64B5CD',  # cyan
        '#4E9F50',  # dark green
    ]

    # 中文注释：仅保留有至少一个时间与质量数据的方法用于散点；柱状图则分别过滤
    methods = sorted(agg.keys())
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(methods)}

    set_nature_rcparams()

    # 中文注释：线性坐标无需对 0 值特殊处理
    def format_value(v: float) -> str:
        # 中文注释：默认 3 有效数字，适配大/小数量级
        try:
            return f"{float(v):.3g}"
        except Exception:
            return str(v)

    def add_bar_labels(ax, rects):
        # 中文注释：在柱顶标注数值
        y_min, y_max = ax.get_ylim()
        y_off = max(1e-12, (y_max - y_min) * 0.015)
        for r in rects:
            h = r.get_height()
            ax.text(
                r.get_x() + r.get_width() / 2.0,
                h + y_off,
                format_value(h),
                ha='center', va='bottom', fontsize=7
            )

    # ---- 平均时间柱状图 ----
    time_methods = [m for m in methods if len(agg[m]["time"]) > 0]
    avg_times = [sum(agg[m]["time"]) / len(agg[m]["time"]) for m in time_methods]

    if len(time_methods) > 0:
        fig_w, fig_h = dynamic_figsize(len(time_methods), h=2.4)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        colors = [color_map[m] for m in time_methods]
        x = list(range(len(time_methods)))
        bars = ax.bar(x, avg_times, color=colors, edgecolor='none')
        ax.set_ylabel('Average Solve Time (s)')  # 英文绘图
        ax.set_xlabel('Method')  # 英文绘图
        ax.set_title(f'Average Solve Time by Method (N={N}, E={E})')  # 英文绘图
        ax.set_xticks(x)
        ax.set_xticklabels(time_methods, rotation=30, ha='right')
        beautify_axes(ax)
        add_bar_labels(ax, bars)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'avg_time.png'))
        plt.close(fig)

    # ---- 平均质量（最大连通子图大小）柱状图：越小越好 ----
    qual_methods = [m for m in methods if len(agg[m]["quality"]) > 0]
    avg_quals = [sum(agg[m]["quality"]) / len(agg[m]["quality"]) for m in qual_methods]

    if len(qual_methods) > 0:
        fig_w, fig_h = dynamic_figsize(len(qual_methods), h=2.4)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        colors = [color_map[m] for m in qual_methods]
        x = list(range(len(qual_methods)))
        bars = ax.bar(x, avg_quals, color=colors, edgecolor='none')
        ax.set_ylabel('Average Max Component Size (smaller is better)')  # 英文绘图
        ax.set_xlabel('Method')  # 英文绘图
        ax.set_title(f'Average Quality by Method (N={N}, E={E})')  # 英文绘图
        ax.set_xticks(x)
        ax.set_xticklabels(qual_methods, rotation=30, ha='right')
        beautify_axes(ax)
        add_bar_labels(ax, bars)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'avg_quality.png'))
        plt.close(fig)

    # ---- 平均效率柱状图（质量/时间；越小越好）----
    eff_methods = [m for m in methods if len(agg[m]["efficiency"]) > 0]
    avg_effs = [sum(agg[m]["efficiency"]) / len(agg[m]["efficiency"]) for m in eff_methods]

    if len(eff_methods) > 0:
        fig_w, fig_h = dynamic_figsize(len(eff_methods), h=2.4)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        colors = [color_map[m] for m in eff_methods]
        x = list(range(len(eff_methods)))
        bars = ax.bar(x, avg_effs, color=colors, edgecolor='none')
        ax.set_ylabel('Average Efficiency (quality/time, lower is better)')  # 英文绘图
        ax.set_xlabel('Method')  # 英文绘图
        ax.set_title(f'Average Efficiency by Method (N={N}, E={E})')  # 英文绘图
        ax.set_xticks(x)
        ax.set_xticklabels(eff_methods, rotation=30, ha='right')
        beautify_axes(ax)
        add_bar_labels(ax, bars)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'avg_efficiency.png'))
        plt.close(fig)

    # ---- 时间-质量散点图 ----
    # 中文注释：需要两者均有均值
    scatter_methods = [m for m in methods if len(agg[m]["time"]) > 0 and len(agg[m]["quality"]) > 0]
    if len(scatter_methods) > 0:
        s_times = [sum(agg[m]["time"]) / len(agg[m]["time"]) for m in scatter_methods]
        s_quals = [sum(agg[m]["quality"]) / len(agg[m]["quality"]) for m in scatter_methods]

        fig, ax = plt.subplots(figsize=(3.5, 2.8))  # 中文注释：单栏散点图
        colors = [color_map[m] for m in scatter_methods]
        ax.scatter(s_times, s_quals, c=colors, s=24, linewidths=0.5, edgecolors='white')
        for xv, yv, label, c in zip(s_times, s_quals, scatter_methods, colors):
            ax.annotate(f"{label}: t={format_value(xv)}, q={format_value(yv)}",
                        (xv, yv), textcoords="offset points", xytext=(6, 6), fontsize=7, color=c)
        ax.set_xlabel('Average Solve Time (s)')  # 英文绘图
        ax.set_ylabel('Average Max Component Size (smaller is better)')  # 英文绘图
        ax.set_title(f'Time vs Quality (N={N}, E={E})')  # 英文绘图
        beautify_axes(ax)
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'time_vs_quality.png'))
        plt.close(fig)


# 中文注释：遍历 results 下所有 N,E 组合并聚合到方法×(N,E) 的均值
def aggregate_overall_by_NE(results_root: str, allow_methods: List[str] = None) -> Tuple[List[str], List[str], Dict[str, Dict[str, List[float]]]]:
    # 中文注释：返回 (NE 列表, 方法列表, 指标字典[metric][method] -> list 对齐 NE)
    # metric ∈ { 'time', 'quality', 'efficiency' }
    ne_to_method_vals: Dict[Tuple[int, int], Dict[str, Dict[str, List[float]]]] = {}

    if not os.path.isdir(results_root):
        return [], [], {"time": {}, "quality": {}, "efficiency": {}}

    for entry in os.listdir(results_root):
        # 目录名格式：N{N}_E{E}_S{S}
        m = re.match(r'^N(\d+)_E(\d+)_S(\d+)$', entry)
        if not m:
            continue
        N_val = int(m.group(1))
        E_val = int(m.group(2))
        subdir = os.path.join(results_root, entry)
        if not os.path.isdir(subdir):
            continue

        for fname in os.listdir(subdir):
            if not fname.endswith('_cut_edges.txt'):
                continue
            fpath = os.path.join(subdir, fname)
            method, t_sec, q_val = parse_result_file(fpath)
            if not method:
                continue
            if allow_methods is not None and len(allow_methods) > 0 and method not in allow_methods:
                continue

            key = (N_val, E_val)
            if key not in ne_to_method_vals:
                ne_to_method_vals[key] = {}
            if method not in ne_to_method_vals[key]:
                ne_to_method_vals[key][method] = {"time": [], "quality": [], "efficiency": []}

            if t_sec is not None:
                ne_to_method_vals[key][method]["time"].append(t_sec)
            if q_val is not None:
                q = float(q_val)
                ne_to_method_vals[key][method]["quality"].append(q)
                if t_sec is not None and t_sec > 0:
                    eff = max(N_val - q, 0.0) / t_sec
                    ne_to_method_vals[key][method]["efficiency"].append(eff)

    # 中文注释：整理为对齐的列表
    ne_keys = sorted(list(ne_to_method_vals.keys()), key=lambda x: (x[0], x[1]))
    ne_labels = [f'N{n}_E{e}' for n, e in ne_keys]
    # 方法集合
    method_set = set()
    for key in ne_keys:
        method_set.update(ne_to_method_vals[key].keys())
    methods_sorted = sorted(method_set) if (allow_methods is None or len(allow_methods) == 0) else [m for m in allow_methods if m in method_set]

    metrics = {"time": {}, "quality": {}, "efficiency": {}}
    for method in methods_sorted:
        for metric in metrics.keys():
            series: List[float] = []
            for key in ne_keys:
                vals = ne_to_method_vals.get(key, {}).get(method, {}).get(metric, [])
                if len(vals) == 0:
                    series.append(float('nan'))
                else:
                    series.append(sum(vals) / len(vals))
            metrics[metric][method] = series

    return ne_labels, methods_sorted, metrics


# 中文注释：绘制跨 N,E 的总体对比折线图
def plot_overall_by_NE(results_root: str, figs_root: str, allow_methods: List[str] = None) -> None:
    out_dir = os.path.join(figs_root, 'overall')
    os.makedirs(out_dir, exist_ok=True)

    # 风格与配色
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn-whitegrid')
        except Exception:
            pass
    mpl.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.15,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    palette = [
        '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2',
        '#937860', '#DA8BC3', '#8C8C8C', '#64B5CD', '#4E9F50',
    ]

    def beautify_axes(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.15)
        ax.grid(axis='x', visible=False)

    def fig_size_for_ne(k: int, base: float = 3.5, per_item: float = 0.12, max_w: float = 8.2, h: float = 2.6):
        w = per_item * max(k, 1) + (base - 1.5)
        w = min(max(w, base), max_w)
        return (w, h)

    ne_labels, methods_sorted, metrics = aggregate_overall_by_NE(results_root, allow_methods)
    if len(ne_labels) == 0 or len(methods_sorted) == 0:
        print('Overall NE comparison skipped: no NE combinations or methods found.')
        return

    color_map = {m: palette[i % len(palette)] for i, m in enumerate(methods_sorted)}

    import math
    step = max(1, math.ceil(len(ne_labels) / 12))
    x_ticks = list(range(len(ne_labels)))
    x_tick_labels = [lbl if (i % step == 0) else '' for i, lbl in enumerate(ne_labels)]

    ylabel_map = {
        'time': 'Solve Time (s)',
        'quality': 'Max Component Size (smaller is better)',
        'efficiency': 'Efficiency ((N - max_component)/time, higher is better)'
    }

    # 中文注释：工具函数：格式化与标注
    def format_value(v: float) -> str:
        try:
            return f"{float(v):.3g}"
        except Exception:
            return str(v)

    def add_bar_labels(ax, bars, y_vals):
        y_min, y_max = ax.get_ylim()
        y_off = max(1e-12, (y_max - y_min) * 0.015)
        for r, yv in zip(bars, y_vals):
            if yv == yv:  # 非 NaN
                h = r.get_height()
                ax.text(
                    r.get_x() + r.get_width() / 2.0,
                    h + y_off,
                    format_value(yv),
                    ha='center', va='bottom', fontsize=6
                )

    # 中文注释：按指标绘制分组条形图
    for metric, fname in [("time", 'overall_time.png'), ("quality", 'overall_quality.png'), ("efficiency", 'overall_efficiency.png')]:
        fig_w, fig_h = fig_size_for_ne(len(ne_labels))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        m_count = max(1, len(methods_sorted))
        group_w = 0.82
        bar_w = min(group_w / m_count, 0.28)
        start = - (m_count - 1) * bar_w / 2.0
        for i, m in enumerate(methods_sorted):
            y_vals = metrics.get(metric, {}).get(m, [])
            x_pos = [xt + (start + i * bar_w) for xt in x_ticks]
            # 将 NaN 替换为 0 以绘制高度，同时保留 y_vals 供标注判断
            y_plot = [0.0 if (yv != yv) else yv for yv in y_vals]
            bars = ax.bar(x_pos, y_plot, width=bar_w, label=m, color=color_map[m], alpha=0.9, edgecolor='none')
            add_bar_labels(ax, bars, y_vals)
        ax.set_xlabel('Dataset (N,E)')
        ax.set_ylabel(ylabel_map.get(metric, metric))
        ax.set_title(f'Overall {metric.capitalize()} across N,E')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels, rotation=30, ha='right')
        beautify_axes(ax)
        ax.legend(ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname))
        plt.close(fig)


def main():
    # 中文注释：命令行参数
    parser = argparse.ArgumentParser(description='Aggregate results and plot averages')
    parser.add_argument('--N', type=int, required=True, help='Number of nodes N', default=200)
    parser.add_argument('--E', type=int, required=True, help='Number of edges E', default=300)
    parser.add_argument('--results_dir', type=str, default='results', help='Results root directory')
    parser.add_argument('--figs_dir', type=str, default='figs', help='Figures root directory')
    parser.add_argument('--methods', type=str, nargs='*', help='Methods to include (e.g., greedy-s3 mealpy-ga)', default=['greedy-s3', 'mealpy-ga', 'mealpy-pso', 'mealpy-woa', 'mealpy-gwo', 'mealpy-de', 'gurobi'])
    args = parser.parse_args()

    N = args.N
    E = args.E
    results_root = os.path.abspath(args.results_dir)
    figs_root = os.path.abspath(args.figs_dir)

    # 中文注释：聚合（按方法过滤）
    allow_methods = args.methods if args.methods is not None and len(args.methods) > 0 else None
    agg = aggregate_results(results_root, N, E, allow_methods)

    # 中文注释：输出目录为 figs/Nxxx_Exxx
    out_dir = os.path.join(figs_root, f'N{N}_E{E}')
    plot_aggregates(agg, out_dir, N, E)

    # 中文注释：简要控制台输出
    non_empty = {m: v for m, v in agg.items() if len(v["time"]) > 0 or len(v["quality"]) > 0}
    if len(non_empty) == 0:
        if allow_methods:
            print(f'No results found for N={N}, E={E} with methods={allow_methods} under {results_root}')
        else:
            print(f'No results found for N={N}, E={E} under {results_root}')
    else:
        print(f'Saved figures to: {out_dir}')

    # 中文注释：生成跨 N,E 的总体对比图
    try:
        plot_overall_by_NE(results_root, figs_root, allow_methods)
    except Exception as e:
        print(f'Overall NE comparison plotting skipped due to error: {e}')


if __name__ == '__main__':
    main()


