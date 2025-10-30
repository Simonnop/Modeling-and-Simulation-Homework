import gurobipy as gp
from gurobipy import GRB
from typing import Dict, Set, List, Tuple
import copy
from utils import find_max_connected_graph


def solve_gurobi(
    nodes_dict: Dict[int, Tuple[float, float]],
    edges_dict: Dict[Tuple[int, int], float],
    neighbor_nodes_dict: Dict[int, Set[int]],
    graph: Dict[int, Set[int]],
    C: int,
    time_limit: int = 300,
    verbose: bool = True
) -> Tuple[List[Tuple[int, int]], int, Dict[int, Set[int]]]:
    """
    使用Gurobi求解器求解图分割优化问题
    
    优化模型（与实现一致）：

    决策变量：
    - v_edge[(i,j)] ∈ {0,1}: 边 (i,j) 是否被删除（1 表示删除）
    - u_{i,j} ∈ {0,1}: 辅助二进制变量，表示从 i 是否能到达 j（1 表示可达）
    - u_node[i] ≥ 0: 节点 i 所在连通分量大小（连续变量，0 ≤ u_node[i] ≤ n）
    - max_u ≥ 0: 最大连通分量大小（目标变量）

    目标：
    minimize max_u

    主要约束：
    (1) 预算约束: 删除的边数量不超过 C，即 sum_{(i,j)∈E} v_edge[(i,j)] ≤ C
    (2) 边连通性: 对任意边 (i,j)∈E, u_{i,j} ≥ 1 - v_edge[(i,j)]（若边未删除则可达）
    (3) 传播/传递约束: 对任意 i≠j, 对任意 k∈N(i), k≠j, 有 u_{i,j} ≥ u_{k,j} - v_edge[(min(i,k),max(i,k))]
    (4) 上界传播（邻边）: 对 (i,j)∈E, u_{i,j} ≤ sum_{k∈N(i)\{j}} u_{k,j} + v_edge[(i,j)]
    (5) 上界传播（非边）: 对 (i,j)∉E, u_{i,j} ≤ sum_{k∈N(i)\{j}} u_{k,j}
    (6) 自达: u_{i,i} = 1
    (7) 对称性: u_{i,j} = u_{j,i}
    (8) 连通分量大小定义: u_node[i] = sum_{j∈V} u_{i,j}
    (9) max_u ≥ u_node[i], ∀i

    变量类型：
    - v_edge 为二进制变量
    - u 为二进制辅助变量
    - u_node 与 max_u 为连续变量

    Args:
        nodes_dict: 节点字典
        edges_dict: 边字典
        neighbor_nodes_dict: 邻居节点字典
        graph: 无向图邻接表
        C: 最多可删除的边数量（预算）
        time_limit: 求解时间限制（秒）
        verbose: 是否打印详细信息
    
    Returns:
        (移除的边列表, 最大连通子图大小, 残余图)
    """
    
    # 创建模型
    model = gp.Model("GraphPartitioning")
    
    # 设置参数
    model.Params.TimeLimit = time_limit
    if not verbose:
        model.Params.OutputFlag = 0
    
    # 获取节点和边集合
    V = list(nodes_dict.keys())
    n = len(V)
    
    # 构建无向边集合
    E = set()
    for (u, v) in edges_dict.keys():
        E.add((min(u, v), max(u, v)))
    E = list(E)
    
    # 决策变量
    # v_edge[(i,j)]: 边(i,j)是否被删除 (二进制)，1 表示删除
    v_edge = model.addVars(E, vtype=GRB.BINARY, name="v_edge")
    
    # u[(i,j)]: 辅助变量，表示j是否可以从i到达 (二进制)
    u = {}
    for i in V:
        for j in V:
            u[i, j] = model.addVar(vtype=GRB.BINARY, name=f"u_{i}_{j}")
    
    # u_node[i]: 节点i所在连通分量的大小
    u_node = model.addVars(V, vtype=GRB.CONTINUOUS, name="u_node", lb=0, ub=n)
    
    # 目标变量：最大连通分量大小
    max_u = model.addVar(vtype=GRB.CONTINUOUS, name="max_u", lb=0, ub=n)
    
    # 目标函数：最小化最大连通分量大小
    model.setObjective(max_u, GRB.MINIMIZE)
    
    # 约束条件
    
    # (1) 预算约束：删除的边数量不超过C
    # 这里简化为：删除边的数量 <= C
    model.addConstr(gp.quicksum(v_edge[e] for e in E) <= C, name="budget")
    
    # (2) 边连通性约束：若(i,j)为边且未被删除，则 u_{i,j} 必须为1
    # 形式化：u_{i,j} >= 1 - v_{i,j}
    for (i, j) in E:
        model.addConstr(
            u[i, j] >= 1 - v_edge[i, j],
            name=f"edge_conn_{i}_{j}"
        )
    
    # (3) 传递约束：通过邻居k到达j时，若(i,k)边未被删除，则 i 可以到达 j
    # 形式化：u_{i,j} >= u_{k,j} - v_{i,k}, ∀k∈N(i), k≠j
    for i in V:
        neighbors_i = graph.get(i, set())
        for j in V:
            if i == j:
                continue
            for k in neighbors_i:
                if k == j:
                    continue
                e_ik = (min(i, k), max(i, k))
                model.addConstr(
                    u[i, j] >= u[k, j] - v_edge[e_ik],
                    name=f"non_neighbor_{i}_{j}_{k}"
                )
    
    # (4) (5) 邻接/非邻接传播的上界约束
    # 若 (i,j) ∈ E： u_{i,j} <= sum_{k∈N(i)\{j}} u_{k,j} + v_{i,j}
    for (i, j) in E:
        neighbors_i = [k for k in graph.get(i, set()) if k != j]
        if neighbors_i:
            model.addConstr(
                u[i, j] <= gp.quicksum(u[k, j] for k in neighbors_i) + v_edge[i, j],
                name=f"neighbor_edge_{i}_{j}"
            )

    # 若 (i,j) ∉ E： u_{i,j} <= sum_{k∈N(i)\{j}} u_{k,j}
    for i in V:
        neighbors_i = graph.get(i, set())
        for j in V:
            if i == j or j in neighbors_i:
                continue
            neighbors_i_except_j = [k for k in neighbors_i if k != j]
            if neighbors_i_except_j:
                model.addConstr(
                    u[i, j] <= gp.quicksum(u[k, j] for k in neighbors_i_except_j),
                    name=f"neighbor_nonedge_{i}_{j}"
                )
    
    # (6)(7) 节点变量 v_i 在本问题中不使用（仅删边），因此不设置对应上界约束
    
    # (10) 对称性约束
    for i in V:
        for j in V:
            if i < j:
                model.addConstr(u[i, j] == u[j, i], name=f"symmetry_{i}_{j}")
    
    # 自达约束：每个节点能到达自身
    for i in V:
        model.addConstr(u[i, i] == 1, name=f"self_reach_{i}")

    # (11) 连通分量大小定义
    for i in V:
        model.addConstr(
            u_node[i] == gp.quicksum(u[i, j] for j in V),
            name=f"component_size_{i}"
        )
    
    # max_u >= u_node[i] for all i
    for i in V:
        model.addConstr(max_u >= u_node[i], name=f"max_u_{i}")
    
    # 求解
    if verbose:
        print("\n开始Gurobi优化求解...")
    
    model.optimize()
    
    # 提取结果
    removed_edges = []
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if verbose:
            if model.status == GRB.OPTIMAL:
                print("找到最优解！")
            else:
                print("达到时间限制，返回当前最优解")
            print(f"目标值（最大连通分量大小）: {max_u.X:.2f}")
        
        # 提取被删除的边
        for e in E:
            if v_edge[e].X > 0.5:  # 二进制变量，>0.5表示为1
                removed_edges.append(e)
        
        if verbose:
            print(f"删除的边数量: {len(removed_edges)}")
    
    else:
        print(f"求解失败，状态码: {model.status}")
        # 如果求解失败，返回空结果
        removed_edges = []
    
    # 构建残余图
    residual_graph = copy.deepcopy(graph)
    for u, v in removed_edges:
        residual_graph[u].discard(v)
        residual_graph[v].discard(u)
    
    # 计算最大连通子图大小
    max_connected_size = find_max_connected_graph(residual_graph)
    
    return removed_edges, max_connected_size, residual_graph
