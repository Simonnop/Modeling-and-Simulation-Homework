import gurobipy as gp
import copy
from typing import Dict, Set, Tuple, List
from utils import find_max_connected_graph


def solve_gurobi(
    nodes_dict: Dict[int, Tuple[float, float]],
    edges_dict: Dict[Tuple[int, int], float],
    neighbor_nodes_dict: Dict[int, Set[int]],
    graph: Dict[int, Set[int]],
    C: int,
    time_limit: int = None,
    verbose: bool = True
) -> Tuple[List[Tuple[int, int]], int, Dict[int, Set[int]]]:
    """
    使用 Gurobi 求解图切割问题
    
    Args:
        nodes_dict: 节点字典 {节点ID: (经度, 纬度)}
        edges_dict: 边字典 {(起点, 终点): 长度}
        neighbor_nodes_dict: 邻居节点字典 {节点: {邻居节点集合}}
        graph: 无向图的邻接表
        C: 需要删除的边数量
        time_limit: 求解时间限制（秒），默认为 None（无限制）
        verbose: 是否显示求解过程
    
    Returns:
        (移除的边列表, 最大连通子图大小, 残余图)
    """
    print(f"\n=== 开始使用 Gurobi 求解 ===")
    print(f"参数设置:")
    print(f"  - 删除边数: {C}")
    print(f"  - 时间限制: {time_limit if time_limit else '无限制'}")
    
    # 模型所需参数
    node_num = len(nodes_dict)
    edge_num = len(edges_dict)
    
    # 构建模型
    model = gp.Model("GraphCutProblem")
    
    # 添加决策变量
    # v[(i,j)]: 边(i,j)是否被删除
    v = {}
    for (i, j) in edges_dict.keys():
        v[(i, j)] = model.addVar(vtype=gp.GRB.BINARY, name=f'v_{i}_{j}')
    
    # u[i][j]: 节点i和节点j是否连通
    u = []
    for i in range(node_num):
        u_i = []
        for j in range(node_num):
            u_ij = model.addVar(vtype=gp.GRB.BINARY, name=f'u_{i}_{j}')
            u_i.append(u_ij)
        u.append(u_i)
    
    # L: 最大连通子图的节点数
    L = model.addVar(lb=0, ub=node_num, vtype=gp.GRB.INTEGER, name='L')
    model.update()
    
    print(f"\n添加约束...")
    # 约束1：删除的边等于特定数目
    model.addConstr(gp.quicksum(list(v.values())) == C, name="edge_count")
    
    # 约束2：邻居节点连通性约束下界
    for (i, j) in edges_dict.keys():
        model.addConstr(u[i][j] >= 1 - v[(i, j)], name=f"neighbor_lb_{i}_{j}")
    
    # 约束3：邻居节点连通性约束上界
    for (i, j) in edges_dict.keys():
        model.addConstr(
            u[i][j] <= gp.quicksum([u[k][j] for k in neighbor_nodes_dict[i]]) + v[(i, j)],
            name=f"neighbor_ub_{i}_{j}"
        )
    
    # 约束4：非邻居节点连通性约束下界
    for i in range(node_num - 1):
        for j in range(i + 1, node_num):
            for k in neighbor_nodes_dict[i]:
                if k == j:
                    continue
                # 确保使用正确的边方向
                edge_key = (i, k) if (i, k) in edges_dict else (k, i)
                if edge_key in v:
                    model.addConstr(u[i][j] >= u[k][j] - v[edge_key], name=f"non_neighbor_lb_{i}_{j}_{k}")
    
    # 约束5：非邻居节点连通性约束上界
    for i in range(node_num - 1):
        for j in range(i + 1, node_num):
            # 检查 (i,j) 是否为边（考虑两个方向）
            is_edge = (i, j) in edges_dict or (j, i) in edges_dict
            if not is_edge:
                neighbors_excluding_j = [k for k in neighbor_nodes_dict[i] if k != j]
                if neighbors_excluding_j:
                    model.addConstr(
                        u[i][j] <= gp.quicksum([u[k][j] for k in neighbors_excluding_j]),
                        name=f"non_neighbor_ub_{i}_{j}"
                    )
    
    # 约束6：连通性的对称性
    for i in range(node_num):
        for j in range(node_num):
            if i != j:
                model.addConstr(u[i][j] == u[j][i], name=f"symmetry_{i}_{j}")
    
    # 约束7：最大连通子图的节点数
    for i in range(node_num):
        model.addConstr(L >= gp.quicksum(u[i]), name=f"max_component_{i}")
    
    # 约束8：节点自己的连通性为1
    for i in range(node_num):
        model.addConstr(u[i][i] == 1, name=f"self_connected_{i}")
    
    # 添加目标函数：最小化最大连通子图大小
    model.setObjective(L, sense=gp.GRB.MINIMIZE)
    
    # 设置求解参数
    if not verbose:
        model.Params.LogToConsole = 0  # 不显示求解过程
    else:
        model.Params.LogToConsole = 1  # 显示求解过程
    
    if time_limit:
        model.Params.TimeLimit = time_limit  # 设置时间限制
    
    # 求解
    print(f"\n开始优化...")
    model.optimize()
    
    # 提取结果
    if model.status == gp.GRB.OPTIMAL:
        print(f"\n找到最优解!")
    elif model.status == gp.GRB.TIME_LIMIT:
        print(f"\n达到时间限制，返回当前最优解")
    else:
        print(f"\n求解状态: {model.status}")
    
    # 提取删除的边
    removed_edges = []
    for (i, j), var in v.items():
        if var.X > 0.5:  # 二进制变量
            removed_edges.append((i, j))
    
    # 构建残余图
    residual_graph = copy.deepcopy(graph)
    for u_node, v_node in removed_edges:
        # 处理有向边和无向边
        residual_graph[u_node].discard(v_node)
        residual_graph[v_node].discard(u_node)
    
    # 计算最大连通子图大小
    max_connected_size = find_max_connected_graph(residual_graph)
    
    print(f"\n优化完成!")
    print(f"  - 目标函数值: {L.X}")
    print(f"  - 实际最大连通子图大小: {max_connected_size}")
    print(f"  - 删除的边数: {len(removed_edges)}")
    
    return removed_edges, max_connected_size, residual_graph