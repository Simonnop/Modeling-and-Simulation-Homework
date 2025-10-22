import scipy.io as sio
from collections import defaultdict
from typing import Dict, Set

def read_mat(filepath:str) :
    """读取.mat网络文件
    用python读取mat文件时，所有数据元素都会变成二维数组

    Args:
        filepath (str): 文件完整路径

    Returns:
        node_num,nodes_dict,edges_dict,neighbor_nodes_dict,graph: 节点属性字典,边属性字典,邻居节点字典,无向图邻接表
    """
    
    mat_data = sio.loadmat(file_name=filepath)
    node_data = mat_data['PlanarNetwork']['Node'][0,0]
    edge_data = mat_data['PlanarNetwork']['Edge'][0,0]
    
    # 保存node节点中数据,{节点:(经度,维度)}
    nodes_dict = {}
    node_num = node_data.shape[1]
    for i in range(node_num):
        nodeID = node_data[0,i][0][0,0]-1
        longitude = node_data[0,i][1][0,0]
        latitude = node_data[0,i][2][0,0]
        nodes_dict[nodeID] = (longitude,latitude)
    
    # 保存edge节点中的数据,{(起点,终点):长度},以及邻居节点{节点:{邻居节点}}
    edges_dict = {}
    neighbor_nodes_dict = defaultdict(set)
    edge_num = edge_data.shape[1]
    for i in range(edge_num):
        fromNodeID = edge_data[0,i][1][0,0]-1
        endNodeID = edge_data[0,i][2][0,0]-1
        edge_len = edge_data[0,i][3][0,0]
        edges_dict[(fromNodeID,endNodeID)] = edge_len
        neighbor_nodes_dict[fromNodeID].add(endNodeID)

    graph = defaultdict(set)
    for node, neighbors in neighbor_nodes_dict.items():
        for neighbor in neighbors:
            graph[node].add(neighbor)
            graph[neighbor].add(node)  # 添加反向边
    
    return nodes_dict, edges_dict, neighbor_nodes_dict, graph


def find_max_connected_graph(graph: Dict[int, Set[int]]) -> int:
    """
    查找最大连通子图的节点数量（使用广度优先遍历BFS）
    Args:
        graph: 无向图的邻接表表示，格式为 {节点: {邻居节点集合}}
    Returns:
        int: 最大连通子图的节点数量
    """
    from collections import deque

    visited = set()
    max_size = 0

    for node in graph:
        if node not in visited:
            queue = deque([node])
            component = set([node])
            visited.add(node)
            while queue:
                current = queue.popleft()
                for neighbor in graph.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
            max_size = max(max_size, len(component))
    return max_size
