from collections import defaultdict
from typing import Dict, Set, List, Tuple
import copy
from utils import find_max_connected_graph


def greedy_edge_removal_strategy1(
    graph: Dict[int, Set[int]], 
    edges: Dict[Tuple[int, int], float],
    C: int
) -> List[Tuple[int, int]]:
    """
    贪心策略1: 每次选择移除后能最大减少最大连通子图大小的边
    
    Args:
        graph: 无向图的邻接表
        edges: 边字典 {(起点, 终点): 长度}
        C: 需要移除的边数量
    
    Returns:
        移除的边列表
    """
    graph_copy = copy.deepcopy(graph)
    edges_copy = copy.deepcopy(edges)
    removed_edges = []
    
    # 构建无向边列表
    all_edges = set()
    for (u, v) in edges_copy.keys():
        all_edges.add((min(u, v), max(u, v)))
    
    for _ in range(C):
        if not all_edges:
            break
            
        best_edge = None
        min_connected_size = float('inf')
        
        # 尝试每条边，选择移除后最大连通子图最小的
        for edge in all_edges:
            u, v = edge
            
            # 临时移除边
            graph_copy[u].discard(v)
            graph_copy[v].discard(u)
            
            # 计算最大连通子图大小
            current_connected_size = find_max_connected_graph(graph_copy)
            
            # 恢复边
            graph_copy[u].add(v)
            graph_copy[v].add(u)
            
            # 更新最佳边
            if current_connected_size < min_connected_size:
                min_connected_size = current_connected_size
                best_edge = edge
        
        if best_edge:
            u, v = best_edge
            # 永久移除最佳边
            graph_copy[u].discard(v)
            graph_copy[v].discard(u)
            all_edges.remove(best_edge)
            removed_edges.append(best_edge)
    
    return removed_edges


def greedy_edge_removal_strategy2(
    graph: Dict[int, Set[int]], 
    edges: Dict[Tuple[int, int], float],
    C: int
) -> List[Tuple[int, int]]:
    """
    贪心策略2: 基于边的重要性（连接高度数节点的边优先）
    
    Args:
        graph: 无向图的邻接表
        edges: 边字典 {(起点, 终点): 长度}
        C: 需要移除的边数量
    
    Returns:
        移除的边列表
    """
    graph_copy = copy.deepcopy(graph)
    removed_edges = []
    
    # 构建无向边列表
    all_edges = set()
    for (u, v) in edges.keys():
        all_edges.add((min(u, v), max(u, v)))
    
    for _ in range(C):
        if not all_edges:
            break
        
        # 计算每条边的得分（两端节点度数之和）
        edge_scores = []
        for edge in all_edges:
            u, v = edge
            degree_u = len(graph_copy.get(u, set()))
            degree_v = len(graph_copy.get(v, set()))
            score = degree_u + degree_v
            edge_scores.append((score, edge))
        
        # 选择得分最高的边（连接高度数节点）
        edge_scores.sort(reverse=True)
        best_edge = edge_scores[0][1]
        
        u, v = best_edge
        # 移除边
        graph_copy[u].discard(v)
        graph_copy[v].discard(u)
        all_edges.remove(best_edge)
        removed_edges.append(best_edge)
    
    return removed_edges


def greedy_edge_removal_strategy3(
    graph: Dict[int, Set[int]], 
    edges: Dict[Tuple[int, int], float],
    C: int
) -> List[Tuple[int, int]]:
    """
    贪心策略3: 基于边介数中心性（在最短路径中出现频率高的边优先）
    
    Args:
        graph: 无向图的邻接表
        edges: 边字典 {(起点, 终点): 长度}
        C: 需要移除的边数量
    
    Returns:
        移除的边列表
    """
    graph_copy = copy.deepcopy(graph)
    removed_edges = []
    
    # 构建无向边列表
    all_edges = set()
    for (u, v) in edges.keys():
        all_edges.add((min(u, v), max(u, v)))
    
    def calculate_edge_betweenness(g: Dict[int, Set[int]]) -> Dict[Tuple[int, int], float]:
        """计算边介数"""
        edge_betweenness = defaultdict(float)
        nodes = list(g.keys())
        
        from tqdm import tqdm
        for source in tqdm(nodes, desc="Calculating edge betweenness", leave=True):
            # BFS计算从source到其他节点的最短路径
            from collections import deque
            queue = deque([source])
            visited = {source}
            pred = defaultdict(list)  # 前驱节点
            dist = {source: 0}
            sigma = defaultdict(int)  # 最短路径数量
            sigma[source] = 1
            
            while queue:
                v = queue.popleft()
                for w in g.get(v, set()):
                    if w not in visited:
                        visited.add(w)
                        queue.append(w)
                        dist[w] = dist[v] + 1
                    
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)
            
            # 累积边介数
            delta = defaultdict(float)
            sorted_nodes = sorted(visited, key=lambda x: dist[x], reverse=True)
            
            for w in sorted_nodes:
                for v in pred[w]:
                    c = (sigma[v] / sigma[w]) * (1 + delta[w])
                    edge = (min(v, w), max(v, w))
                    edge_betweenness[edge] += c
                    delta[v] += c
        
        return edge_betweenness
    
    for i in range(C):

        print(f"Calculating edge betweenness {i+1}/{C}")
        
        if not all_edges:
            break
        
        # 计算边介数
        betweenness = calculate_edge_betweenness(graph_copy)
        
        # 选择介数最高的边
        valid_betweenness = {e: betweenness.get(e, 0) for e in all_edges}
        best_edge = max(valid_betweenness.keys(), key=lambda e: valid_betweenness[e])
        
        u, v = best_edge
        # 移除边
        graph_copy[u].discard(v)
        graph_copy[v].discard(u)
        all_edges.remove(best_edge)
        removed_edges.append(best_edge)
    
    return removed_edges


def greedy_edge_removal_strategy4(
    graph: Dict[int, Set[int]], 
    edges: Dict[Tuple[int, int], float],
    C: int
) -> List[Tuple[int, int]]:
    """
    贪心策略4: 混合策略 - 优先移除高度数节点间的边，同时考虑三角形数量
    
    Args:
        graph: 无向图的邻接表
        edges: 边字典 {(起点, 终点): 长度}
        C: 需要移除的边数量
    
    Returns:
        移除的边列表
    """
    graph_copy = copy.deepcopy(graph)
    removed_edges = []
    
    # 构建无向边列表
    all_edges = set()
    for (u, v) in edges.keys():
        all_edges.add((min(u, v), max(u, v)))
    
    def count_triangles_with_edge(g: Dict[int, Set[int]], u: int, v: int) -> int:
        """计算包含边(u,v)的三角形数量"""
        neighbors_u = g.get(u, set())
        neighbors_v = g.get(v, set())
        return len(neighbors_u.intersection(neighbors_v))
    
    for _ in range(C):
        if not all_edges:
            break
        
        # 计算每条边的得分（度数乘积 + 三角形数量）
        edge_scores = []
        for edge in all_edges:
            u, v = edge
            degree_u = len(graph_copy.get(u, set()))
            degree_v = len(graph_copy.get(v, set()))
            triangles = count_triangles_with_edge(graph_copy, u, v)
            score = degree_u * degree_v - 10 * triangles  # 三角形权重更高
            edge_scores.append((score, edge))
        
        # 选择得分最高的边
        edge_scores.sort(reverse=True)
        best_edge = edge_scores[0][1]
        
        u, v = best_edge
        # 移除边
        graph_copy[u].discard(v)
        graph_copy[v].discard(u)
        all_edges.remove(best_edge)
        removed_edges.append(best_edge)
    
    return removed_edges


def solve_greedy(
    nodes_dict: Dict[int, Tuple[float, float]],
    edges_dict: Dict[Tuple[int, int], float],
    neighbor_nodes_dict: Dict[int, Set[int]],
    graph: Dict[int, Set[int]],
    C: int,
    strategy: str = "s1"
) -> Tuple[List[Tuple[int, int]], int, Dict[int, Set[int]]]:
    """
    使用贪心算法求解边移除问题
    
    Args:
        nodes_dict: 节点字典
        edges_dict: 边字典
        neighbor_nodes_dict: 邻居节点字典
        graph: 无向图邻接表
        C: 需要移除的边数量
        strategy: 策略选择 ("s1", "s2", "s3", "s4")，默认为s1
    
    Returns:
        (移除的边列表, 最大连通子图大小, 残余图)
    """
    # 选择策略
    if strategy == "s1":
        removed_edges = greedy_edge_removal_strategy1(graph, edges_dict, C)
    elif strategy == "s2":
        removed_edges = greedy_edge_removal_strategy2(graph, edges_dict, C)
    elif strategy == "s3":
        removed_edges = greedy_edge_removal_strategy3(graph, edges_dict, C)
    else:  # s4
        removed_edges = greedy_edge_removal_strategy4(graph, edges_dict, C)
    
    # 构建残余图
    residual_graph = copy.deepcopy(graph)
    for u, v in removed_edges:
        residual_graph[u].discard(v)
        residual_graph[v].discard(u)
    
    # 计算最大连通子图大小
    max_connected_size = find_max_connected_graph(residual_graph)
    
    return removed_edges, max_connected_size, residual_graph

