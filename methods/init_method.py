from typing import Dict, Set, Tuple, List, Optional
import os
import numpy as np

from methods.greedy_method import solve_greedy


def solve_init(
    nodes_dict: Dict[int, Tuple[float, float]],
    edges_dict: Dict[Tuple[int, int], float],
    neighbor_nodes_dict: Dict[int, Set[int]],
    graph: Dict[int, Set[int]],
    C: int,
    pop_size: int = 50,
    strategy: str = "s3",
    result_file: Optional[str] = None,
    algorithm: str = "GA",
    n_generations: int = 100,
    seed: Optional[int] = None,
) -> (List[Tuple[int, int]], int, Dict[int, Set[int]]):
    """
    使用贪心算法生成用于 Mealpy 的初始种群。

    返回形状为 (pop_size, num_edges) 的 numpy 数组，元素在 [0,1] 之间，
    并确保边的顺序与 `methods.mealpy_method.GraphCutProblem.edge_list` 使用的顺序兼容。
    """
    # 读取贪心结果文件（优先使用 result_file 参数，否则在 results 目录中查找匹配文件）
    removed_edges: List[Tuple[int, int]] = []

    def _read_removed_edges_from_file(path: str) -> List[Tuple[int, int]]:
        res: List[Tuple[int, int]] = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('-'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            u = int(parts[0])
                            v = int(parts[1])
                            res.append((min(u, v), max(u, v)))
                        except ValueError:
                            continue
        except FileNotFoundError:
            return []
        return res

    if result_file:
        removed_edges = _read_removed_edges_from_file(result_file)
    else:
        # 在 results 目录中查找与 strategy 匹配的文件，选择最新的一个
        results_dir = os.path.abspath('results')
        candidate_files: List[str] = []
        if os.path.isdir(results_dir):
            for root, _, files in os.walk(results_dir):
                for fname in files:
                    if fname.endswith('_cut_edges.txt') and f'greedy-{strategy}' in fname:
                        candidate_files.append(os.path.join(root, fname))

        if candidate_files:
            # 按文件修改时间选择最新的
            candidate_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            removed_edges = _read_removed_edges_from_file(candidate_files[0])
        else:
            # 回退到调用贪心（作为最后手段）
            removed_edges, _, _ = solve_greedy(
                nodes_dict, edges_dict, neighbor_nodes_dict, graph, C, strategy=strategy
            )

    # 构建无向边列表（与 mealpy_method 中的顺序一致）
    edge_list = []
    for (u, v) in edges_dict.keys():
        edge = (min(u, v), max(u, v))
        if edge not in edge_list:
            edge_list.append(edge)

    num_edges = len(edge_list)

    # 将贪心结果编码为连续向量：被删除的边置为1，其余为0
    base_solution = np.zeros(num_edges, dtype=float)
    removed_set = set((min(u, v), max(u, v)) for u, v in removed_edges)
    for i, edge in enumerate(edge_list):
        if edge in removed_set:
            base_solution[i] = 1.0

    population = np.zeros((pop_size, num_edges), dtype=float)
    population[0] = base_solution

    # 接入 mealpy 求解
    try:
        from methods.mealpy_method import GraphCutProblem
        from mealpy import FloatVar
    except Exception:
        # 无法导入 mealpy，回退返回种群
        return population, edge_list

    problem = GraphCutProblem(graph, edges_dict, C)
    bounds = FloatVar(lb=[0.0] * problem.num_edges, ub=[1.0] * problem.num_edges)

    algo = algorithm.upper()
    model = None
    if algo == "GA":
        from mealpy import GA
        model = GA.BaseGA(epoch=n_generations, pop_size=pop_size, pc=0.9, pm=0.1, seed=seed)
    elif algo == "PSO":
        from mealpy import PSO
        model = PSO.OriginalPSO(epoch=n_generations, pop_size=pop_size, seed=seed)
    elif algo == "DE":
        from mealpy import DE
        model = DE.JADE(epoch=n_generations, pop_size=pop_size, wf=0.8, cr=0.9, seed=seed)
    elif algo == "WOA":
        from mealpy import WOA
        model = WOA.OriginalWOA(epoch=n_generations, pop_size=pop_size, seed=seed)
    elif algo == "GWO":
        from mealpy import GWO
        model = GWO.OriginalGWO(epoch=n_generations, pop_size=pop_size, seed=seed)
    elif algo == "ABC":
        from mealpy import ABC
        model = ABC.OriginalABC(epoch=n_generations, pop_size=pop_size, seed=seed)
    else:
        raise ValueError(f"未知的算法类型: {algorithm}")

    # 构造初始解并传入优化器（参照示例：starting_solutions 为 numpy 数组）
    init_solution = population  # shape: (pop_size, num_edges)

    num_workers = pop_size + 1
    num_workers = num_workers if num_workers <= 7 else 7
    term_dict = {"max_early_stop": 20}

    g_best = model.solve(
            problem={"obj_func": problem.fitness_function,
                    "bounds": bounds,
                    "minmax": "min",
                    "log_to": None},
            n_workers=num_workers,
            termination=term_dict,
            mode='thread',
            starting_solutions=init_solution,
        )
    
    best_position = g_best.solution
    removed_edges_opt = problem.decode_solution(best_position)

    # 构建残余图并计算最大连通子图大小
    residual_graph = {k: set(v) for k, v in graph.items()}
    for u, v in removed_edges_opt:
        residual_graph[u].discard(v)
        residual_graph[v].discard(u)

    try:
        from utils import find_max_connected_graph
        max_connected_size = find_max_connected_graph(residual_graph)
    except Exception:
        max_connected_size = -1

    return removed_edges_opt, max_connected_size, residual_graph
