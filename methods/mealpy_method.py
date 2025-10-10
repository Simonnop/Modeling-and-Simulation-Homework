import numpy as np
import copy
from typing import Dict, Set, Tuple, List
from mealpy import FloatVar
from utils import find_max_connected_graph


class GraphCutProblem:
    """图切割问题的目标函数定义"""
    
    def __init__(self, graph: Dict[int, Set[int]], 
                 edges: Dict[Tuple[int, int], float], 
                 C: int):
        """
        初始化图切割问题
        
        Args:
            graph: 无向图的邻接表
            edges: 边字典 {(起点, 终点): 长度}
            C: 需要移除的边数量
        """
        self.graph = graph
        self.edges = edges
        self.C = C
        
        # 构建边列表（无向边，统一为 (min, max) 格式）
        self.edge_list = []
        for (u, v) in edges.keys():
            edge = (min(u, v), max(u, v))
            if edge not in self.edge_list:
                self.edge_list.append(edge)
        
        self.num_edges = len(self.edge_list)
        print(f"问题初始化: 共有 {self.num_edges} 条边，需要移除 {C} 条")
    
    def decode_solution(self, solution: np.ndarray) -> List[Tuple[int, int]]:
        """
        将连续解码为离散的边选择
        
        Args:
            solution: 连续向量，每个元素在[0,1]之间
        
        Returns:
            选中的边列表
        """
        # 获取每条边的得分
        edge_scores = [(solution[i], self.edge_list[i]) for i in range(self.num_edges)]
        # 按得分排序，选择得分最高的C条边
        edge_scores.sort(reverse=True)
        selected_edges = [edge for _, edge in edge_scores[:self.C]]
        return selected_edges
    
    def fitness_function(self, solution: np.ndarray) -> float:
        """
        适应度函数：计算移除边后的最大连通子图大小
        
        Args:
            solution: 连续向量表示的解
        
        Returns:
            最大连通子图大小（需要最小化）
        """
        # 解码为边列表
        removed_edges = self.decode_solution(solution)
        
        # 构建残余图
        residual_graph = copy.deepcopy(self.graph)
        for u, v in removed_edges:
            residual_graph[u].discard(v)
            residual_graph[v].discard(u)
        
        # 计算最大连通子图大小
        max_connected_size = find_max_connected_graph(residual_graph)
        
        return max_connected_size


def solve_mealpy(
    nodes_dict: Dict[int, Tuple[float, float]],
    edges_dict: Dict[Tuple[int, int], float],
    neighbor_nodes_dict: Dict[int, Set[int]],
    graph: Dict[int, Set[int]],
    C: int,
    algorithm: str = "GA",  # GA, PSO, DE, WOA 等
    pop_size: int = 50,
    n_generations: int = 100,
    seed: int = None
) -> Tuple[List[Tuple[int, int]], int, Dict[int, Set[int]]]:
    """
    使用 mealpy 的多种算法求解边移除问题
    
    Args:
        algorithm: 算法类型 ("GA", "PSO", "DE", "WOA", "GWO" 等)
        其他参数同 solve_mealpy_ga
    
    Returns:
        (移除的边列表, 最大连通子图大小, 残余图)
    """
    print(f"\n=== 开始使用 Mealpy {algorithm} 算法求解 ===")
    print(f"参数设置:")
    print(f"  - 算法: {algorithm}")
    print(f"  - 种群大小: {pop_size}")
    print(f"  - 迭代代数: {n_generations}")
    
    # 创建问题实例
    problem = GraphCutProblem(graph, edges_dict, C)
    
    # 定义问题的边界
    bounds = FloatVar(lb=[0.0] * problem.num_edges, ub=[1.0] * problem.num_edges)
    
    # 根据算法类型创建模型
    if algorithm == "GA":
        from mealpy import GA
        model = GA.BaseGA(epoch=n_generations, pop_size=pop_size, pc=0.9, pm=0.1, seed=seed)
    elif algorithm == "PSO":
        from mealpy import PSO
        model = PSO.OriginalPSO(epoch=n_generations, pop_size=pop_size, seed=seed)
    elif algorithm == "DE":
        from mealpy import DE
        model = DE.BaseDE(epoch=n_generations, pop_size=pop_size, wf=0.8, cr=0.9, seed=seed)
    elif algorithm == "WOA":
        from mealpy import WOA
        model = WOA.OriginalWOA(epoch=n_generations, pop_size=pop_size, seed=seed)
    elif algorithm == "GWO":
        from mealpy import GWO
        model = GWO.OriginalGWO(epoch=n_generations, pop_size=pop_size, seed=seed)
    elif algorithm == "ABC":
        from mealpy import ABC
        model = ABC.OriginalABC(epoch=n_generations, pop_size=pop_size, seed=seed)
    else:
        raise ValueError(f"未知的算法类型: {algorithm}")
    
    # 求解
    print(f"\n开始优化...")
    g_best = model.solve(
        problem={"obj_func": problem.fitness_function, 
                "bounds": bounds, 
                "minmax": "min",
                "log_to": None}
    )
    best_position = g_best.solution
    best_fitness = g_best.target.fitness
    
    # 解码最优解
    removed_edges = problem.decode_solution(best_position)
    
    # 构建残余图
    residual_graph = copy.deepcopy(graph)
    for u, v in removed_edges:
        residual_graph[u].discard(v)
        residual_graph[v].discard(u)
    
    # 计算最大连通子图大小
    max_connected_size = find_max_connected_graph(residual_graph)
    
    print(f"\n优化完成!")
    print(f"  - 最优适应度: {best_fitness}")
    print(f"  - 实际最大连通子图大小: {max_connected_size}")
    
    return removed_edges, max_connected_size, residual_graph

