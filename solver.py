import os
import time
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Set, Tuple, List
from utils import read_mat
from methods.greedy_method import solve_greedy
from methods.mealpy_method import solve_mealpy

class Solver:
    """图切割问题求解器"""
    
    def __init__(self, data: str, method: str = "greedy", **kwargs):
        """
        初始化求解器
        
        Args:
            data: 数据集编号
            method: 求解方法 ("greedy-s1", "gurobi", "mealpy-ga", "mealpy-pso" 等)
            **kwargs: 算法特定参数
        """
        self.algo_params = kwargs
        data_path = f"samples/PlanarNetwork_N200_{data}.mat"
        self.data_id = f"{data}"
        self.data_path = data_path
        self.method = method
        
        # 读取数据
        print(f"正在读取数据文件: {data_path}")
        self.nodes_dict, self.edges_dict, self.neighbor_nodes_dict, self.graph = read_mat(data_path)
        
        print(f"数据加载完成:")
        print(f"  - 节点数: {len(self.nodes_dict)}")
        print(f"  - 边数: {len(self.edges_dict)}")
        
        # 求解结果
        self.removed_edges = None
        self.max_connected_size = None
        self.residual_graph = None
        self.solve_time = None
        
    def solve(self, C: int):
        """
        求解图切割问题
        
        Args:
            C: 需要删除的边数量
        """
        print(f"\n开始求解 (方法: {self.method}, 删除边数: {C})")
        start_time = time.time()
        
        if "greedy" in self.method:
            self.removed_edges, self.max_connected_size, self.residual_graph = solve_greedy(
                self.nodes_dict,
                self.edges_dict,
                self.neighbor_nodes_dict,
                self.graph,
                C,
                strategy=self.method.split("-")[1]
            )
        elif self.method == "gurobi":
            # 导入gurobi求解函数
            from methods.gurobi_method import solve_gurobi
            self.removed_edges, self.max_connected_size, self.residual_graph = solve_gurobi(
                self.nodes_dict,
                self.edges_dict,
                self.neighbor_nodes_dict,
                self.graph,
                C,
                **self.algo_params
            )
        elif "mealpy" in self.method:
            # 导入mealpy求解函数
            # 使用其他算法（如 mealpy-pso, mealpy-de 等）
            algo_type = self.method.split("-")[1].upper()
            self.removed_edges, self.max_connected_size, self.residual_graph = solve_mealpy(
                self.nodes_dict,
                self.edges_dict,
                self.neighbor_nodes_dict,
                self.graph,
                C,
                algorithm=algo_type,
                **self.algo_params
            )
        else:
            raise ValueError(f"未知的求解方法: {self.method}")
        
        self.solve_time = time.time() - start_time
        
        print(f"\n求解完成!")
        print(f"  - 求解时间: {self.solve_time:.2f} 秒")
        print(f"  - 删除的边数: {len(self.removed_edges)}")
        print(f"  - 最大连通子图大小: {self.max_connected_size}")
        
    def save_cut_edges(self, output_path: str = "cut_edges.txt"):
        """
        保存切割的边到txt文件
        
        Args:
            output_path: 输出文件路径
        """
        if self.removed_edges is None:
            print("警告: 尚未进行求解，无法保存结果")
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# 求解方法: {self.method}\n")
            f.write(f"# 删除的边数: {len(self.removed_edges)}\n")
            f.write(f"# 最大连通子图大小: {self.max_connected_size}\n")
            f.write(f"# 求解时间: {self.solve_time:.2f} 秒\n")
            f.write(f"# 格式: 起点 终点\n")
            f.write("-" * 50 + "\n")
            
            for u, v in self.removed_edges:
                f.write(f"{u} {v}\n")
        
        print(f"\n切割的边已保存到: {output_path}")
        
    def visualize(self, output_path: str = "graph_visualization.png"):
        """
        可视化原始图和切割后的图
        
        Args:
            output_path: 输出图片路径
        """
        if self.residual_graph is None:
            print("警告: 尚未进行求解，无法可视化")
            return
        
        print(f"\n正在生成可视化...")
        
        # 构建原始图
        G_original = nx.Graph()
        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                if node < neighbor:  # 避免重复添加边
                    G_original.add_edge(node, neighbor)
        
        # 构建残余图
        G_residual = nx.Graph()
        for node, neighbors in self.residual_graph.items():
            for neighbor in neighbors:
                if node < neighbor:  # 避免重复添加边
                    G_residual.add_edge(node, neighbor)
        
        # 找到最大连通分量
        if G_residual.number_of_nodes() > 0:
            connected_components = list(nx.connected_components(G_residual))
            max_component = max(connected_components, key=len) if connected_components else set()
        else:
            max_component = set()
        
        # 创建节点位置（使用经纬度坐标）
        pos = {node: (lon, lat) for node, (lon, lat) in self.nodes_dict.items()}
        
        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # 左图：原始图
        ax1 = axes[0]
        nx.draw_networkx_nodes(G_original, pos, node_color='lightblue', 
                              node_size=100, ax=ax1, alpha=0.7)
        nx.draw_networkx_edges(G_original, pos, alpha=0.3, ax=ax1)
        ax1.set_title(f'Original Graph\nNodes: {G_original.number_of_nodes()}, Edges: {G_original.number_of_edges()}', 
                     fontsize=14, pad=20)
        ax1.axis('off')
        
        # 中图：被删除的边
        ax2 = axes[1]
        nx.draw_networkx_nodes(G_original, pos, node_color='lightgray', 
                              node_size=100, ax=ax2, alpha=0.5)
        nx.draw_networkx_edges(G_original, pos, alpha=0.1, ax=ax2)
        
        # 高亮显示被删除的边
        removed_edges_list = [(u, v) for u, v in self.removed_edges]
        nx.draw_networkx_edges(G_original, pos, edgelist=removed_edges_list, 
                              edge_color='red', width=2, ax=ax2, alpha=0.8)
        ax2.set_title(f'Removed Edges\nCount: {len(self.removed_edges)}', 
                     fontsize=14, pad=20)
        ax2.axis('off')
        
        # 右图：残余图和最大连通子图
        ax3 = axes[2]
        # 绘制所有残余节点
        node_colors = ['red' if node in max_component else 'lightgray' 
                      for node in G_residual.nodes()]
        node_sizes = [200 if node in max_component else 50 
                     for node in G_residual.nodes()]
        
        nx.draw_networkx_nodes(G_residual, pos, node_color=node_colors, 
                              node_size=node_sizes, ax=ax3, alpha=0.7)
        nx.draw_networkx_edges(G_residual, pos, alpha=0.3, ax=ax3)
        
        ax3.set_title(f'Residual Graph (Max Component Highlighted)\n' + 
                     f'Nodes: {G_residual.number_of_nodes()}, Edges: {G_residual.number_of_edges()}, ' +
                     f'Max Component: {len(max_component)}',
                     fontsize=14, pad=20)
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"可视化图已保存到: {output_path}")
        plt.close()
        
    def run(self, C: int, save_path: str = None, visualize_path: str = None):
        """
        完整运行求解流程：求解 -> 保存结果 -> 可视化
        
        Args:
            C: 需要删除的边数量
            save_path: 保存切割边的文件路径，默认为 "cut_edges_{method}.txt"
            visualize_path: 可视化图片路径，默认为 "visualization_{method}.png"
        """
        # 求解
        self.solve(C)

        if not os.path.exists(f"results/{self.data_id}"):
            os.makedirs(f"results/{self.data_id}")
        
        # 生成默认文件名
        if save_path is None:
            save_path = f"results/{self.data_id}/{self.method}_cut_edges.txt"
        if visualize_path is None:
            visualize_path = f"results/{self.data_id}/{self.method}_visualization.png"
        
        # 保存结果
        self.save_cut_edges(save_path)
        
        # 可视化
        self.visualize(visualize_path)
        
        print("\n" + "=" * 60)
        print("求解流程完成!")
        print("=" * 60)


if __name__ == "__main__":
    # 使用示例
    
    # 示例1: 使用贪心方法求解, s3 好一点
    solver = Solver(
        data="E250_S1",
        method="greedy-s3",
    )
    solver.run(C=10)
    
    # 示例2: 使用 Mealpy 遗传算法求解
    # solver = Solver(
    #     data="E250_S1",
    #     method="mealpy-ga",
    #     pop_size=50,
    #     n_generations=100
    # )
    # solver.run(C=10)
    
    # 示例3: 使用 Gurobi 方法求解（需要安装 Gurobi）
    # solver = Solver(
    #     data="E250_S1",
    #     method="gurobi",
    #     time_limit=300,  # 5分钟时间限制
    #     verbose=True
    # )
    # solver.run(C=10)

