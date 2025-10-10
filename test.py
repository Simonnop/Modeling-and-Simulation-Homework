from utils import read_mat, find_max_connected_graph

_,_,_,graph = read_mat('samples/PlanarNetwork_N200_E250_S100.mat')

print("initial state: ",find_max_connected_graph(graph))