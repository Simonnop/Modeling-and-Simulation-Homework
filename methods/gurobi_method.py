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
    # v_{i,j}: 边(i,j)是否被删除，1表示删除，0表示保留
    v = {}
    for (i, j) in edges_dict.keys():
        v[(i, j)] = model.addVar(vtype=gp.GRB.BINARY, name=f'v_{i}_{j}')
    
    # u_{i,j}: 节点i和节点j是否连通，1表示连通，0表示不连通
    u = []
    for i in range(node_num):
        u_i = []
        for j in range(node_num):
            u_ij = model.addVar(vtype=gp.GRB.BINARY, name=f'u_{i}_{j}')
            u_i.append(u_ij)
        u.append(u_i)
    
    # u_i: 节点i所在连通分量的大小（即与节点i连通的节点总数）
    u_sum = []
    for i in range(node_num):
        u_sum.append(model.addVar(lb=0, ub=node_num, vtype=gp.GRB.INTEGER, name=f'u_sum_{i}'))
    
    model.update()
    
    print(f"\n添加约束...")
    
    # 约束(1)：删除边数量约束
    # Σ v_{i,j} = C（删除C条边）
    model.addConstr(gp.quicksum(list(v.values())) == C, name="edge_count")
    
    # 约束(2)：邻居节点连通性约束下界
    # u_{i,j} >= 1 - v_{i,j}, ∀(i,j) ∈ E
    # 如果边(i,j)保留（v_{i,j}=0），则节点i和j必须连通（u_{i,j}>=1）
    for (i, j) in edges_dict.keys():
        model.addConstr(u[i][j] >= 1 - v[(i, j)], name=f"neighbor_lb_{i}_{j}")
    
    # 约束(3)：非邻居节点连通性约束下界
    # u_{i,j} >= u_{k,j} - v_{i,k}, ∀i,j ∈ V, i≠j, ∀k ∈ N_G(i), k≠j
    # 如果k和j连通，且边(i,k)保留，则i和j也连通
    for i in range(node_num):
        for j in range(node_num):
            if i == j:
                continue
            for k in neighbor_nodes_dict[i]:
                if k == j:
                    continue
                # 找到边(i,k)或(k,i)
                edge_key = None
                if (i, k) in edges_dict:
                    edge_key = (i, k)
                elif (k, i) in edges_dict:
                    edge_key = (k, i)
                
                if edge_key:
                    model.addConstr(
                        u[i][j] >= u[k][j] - v[edge_key],
                        name=f"non_neighbor_lb_{i}_{j}_{k}"
                    )
    
    # 约束(4)：非邻居节点连通性约束上界（邻居）
    # u_{i,j} <= Σ_{k∈N_G(i),k≠j} u_{k,j} + v_{i,j}, ∀(i,j) ∈ E
    for (i, j) in edges_dict.keys():
        neighbors_excluding_j = [k for k in neighbor_nodes_dict[i] if k != j]
        if neighbors_excluding_j:
            model.addConstr(
                u[i][j] <= gp.quicksum([u[k][j] for k in neighbors_excluding_j]) + v[(i, j)],
                name=f"neighbor_ub_{i}_{j}"
            )
    
    # 约束(5)：非邻居节点连通性约束上界（非邻居）
    # u_{i,j} <= Σ_{k∈N_G(i),k≠j} u_{k,j}, ∀(i,j) ∉ E
    for i in range(node_num):
        for j in range(node_num):
            if i == j:
                continue
            # 检查(i,j)是否为边
            is_edge = (i, j) in edges_dict or (j, i) in edges_dict
            if not is_edge:
                neighbors_excluding_j = [k for k in neighbor_nodes_dict[i] if k != j]
                if neighbors_excluding_j:
                    model.addConstr(
                        u[i][j] <= gp.quicksum([u[k][j] for k in neighbors_excluding_j]),
                        name=f"non_neighbor_ub_{i}_{j}"
                    )
    
    # 约束(10)：连通性对称性
    # u_{i,j} = u_{j,i}, ∀i,j ∈ V
    for i in range(node_num):
        for j in range(i + 1, node_num):
            model.addConstr(u[i][j] == u[j][i], name=f"symmetry_{i}_{j}")
    
    # 约束：节点与自己连通
    # u_{i,i} = 1, ∀i ∈ V
    for i in range(node_num):
        model.addConstr(u[i][i] == 1, name=f"self_connected_{i}")
    
    # 约束(11)：计算每个节点所在连通分量大小
    # u_i = Σ_{j∈V} u_{i,j}, ∀i ∈ V
    for i in range(node_num):
        model.addConstr(
            u_sum[i] == gp.quicksum([u[i][j] for j in range(node_num)]),
            name=f"component_size_{i}"
        )
    
    # 目标函数：最小化最大连通子图大小
    # min max_i u_i
    max_component_size = model.addVar(lb=0, ub=node_num, vtype=gp.GRB.INTEGER, name='max_component')
    for i in range(node_num):
        model.addConstr(max_component_size >= u_sum[i], name=f"max_component_{i}")
    
    model.setObjective(max_component_size, sense=gp.GRB.MINIMIZE)
    
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
        if var.X > 0.5:  # 二进制变量，如果v_{i,j}=1表示边被删除
            removed_edges.append((i, j))
    
    # 构建残余图
    residual_graph = copy.deepcopy(graph)
    
    # 删除边
    for u_node, v_node in removed_edges:
        residual_graph[u_node].discard(v_node)
        residual_graph[v_node].discard(u_node)
    
    # 计算最大连通子图大小
    max_connected_size = find_max_connected_graph(residual_graph)
    
    print(f"\n优化完成!")
    print(f"  - 目标函数值: {max_component_size.X}")
    print(f"  - 实际最大连通子图大小: {max_connected_size}")
    print(f"  - 删除的边数: {len(removed_edges)}")
    
    return removed_edges, max_connected_size, residual_graph