import matplotlib
matplotlib.use('TkAgg')

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
import copy
import powerlaw
import random
from scipy.fft import fft, fftfreq
from scipy.fft import fft2, fftfreq

# 设置随机种子
np.random.seed(21)

# 设置二维平面范围
longitude_range = [-90, 90]
latitude_range = [-180, 180]

# 生成符合幂率分布的节点度数
def power_law_degree_sequence(n, exponent=2):
    return np.random.zipf(exponent, n)

# 生成节点坐标
def generate_coordinates(num_nodes):
    longitude = np.random.uniform(low=longitude_range[0], high=longitude_range[1], size=num_nodes)
    latitude = np.random.uniform(low=latitude_range[0], high=latitude_range[1], size=num_nodes)
    return list(zip(longitude, latitude))

# 生成网络
def generate_network(num_nodes, average_degree):
    G = nx.Graph()
    degrees = power_law_degree_sequence(num_nodes)
    coordinates = generate_coordinates(num_nodes)

    # 添加节点
    for i in range(num_nodes):
        G.add_node(i, coordinates=coordinates[i])

    # 添加边，优先连接距离近的节点
    for i in range(num_nodes):
        node_i = G.nodes[i]['coordinates']
        distances = [np.linalg.norm(np.array(node_i) - np.array(G.nodes[j]['coordinates'])) for j in range(i)]
        closest_nodes = np.argsort(distances)[:min(int(average_degree), i)]
        for j in closest_nodes:
            G.add_edge(i, j)

    return G

# 绘制网络
def plot_network(G, node_size_factor=20, font_size=8):
    pos = {i: G.nodes[i]['coordinates'] for i in G.nodes}

    # 获取节点度数
    node_degrees = [G.degree(node) for node in G.nodes]

    # 找到节点度数最高的节点
    max_degree_node = np.argmax(node_degrees)

    # 将节点度数最高的节点标记为红色
    node_colors = ['yellow' if i == start_node else 'red' if i == max_degree_node else 'skyblue' for i in G.nodes]

    nx.draw(G, pos, with_labels=False, node_size=[deg * node_size_factor for deg in node_degrees],
            node_color=node_colors, cmap=plt.cm.Blues)

    # 添加节点标签
    nx.draw_networkx_labels(G, pos, labels={i: str(i) for i in G.nodes}, font_size=font_size)

    # 添加带有曲度的边
    for edge in G.edges:
        start = pos[edge[0]]
        end = pos[edge[1]]
        line = plt.Line2D([start[0], end[0]], [start[1], end[1]], color='grey', linestyle='-', linewidth=0.8,
                          alpha=0.7)
        plt.gca().add_line(line)

    # plt.show()

# 计算网络的平均度数
def calculate_average_degree(G):
    degrees = [G.degree(node) for node in G.nodes]
    return np.mean(degrees)

# 最远点采样算法
def farthest_point_sampling(G, start_node, num_samples):
    sampled_nodes = {start_node}
    remaining_nodes = set(G.nodes) - sampled_nodes

    while len(sampled_nodes) < num_samples:
        distances = {node: nx.shortest_path_length(G, source=start_node, target=node) for node in remaining_nodes}
        farthest_node = max(distances, key=distances.get)
        sampled_nodes.add(farthest_node)
        remaining_nodes.remove(farthest_node)

    return sampled_nodes

# 生成并绘制网络
num_nodes = 1000
average_degree = np.random.uniform(1, 3)
network = generate_network(num_nodes, average_degree)

# 输出每个节点的坐标、度数和序号
#for node in network.nodes:
    #print(f"Node {node}: Coordinates = {network.nodes[node]['coordinates']}, Degree = {network.degree[node]}")


# Find the node with the highest degree in the original graph
highest_degree_node = max(network.nodes, key=lambda node: network.degree(node))

# Randomly select one of its neighbors as the new starting node
start_node_neighbors = list(network.neighbors(highest_degree_node))
start_node = random.choice(start_node_neighbors)

# 运行最远点采样算法，选出100个节点
num_samples = 300
sampled_nodes = farthest_point_sampling(network, start_node, num_samples)

# 复制原始图，保持原有连接属性
subgraph = copy.deepcopy(network)
subgraph.remove_nodes_from(set(network.nodes) - sampled_nodes)

# removed_nodes = [node for node in subgraph.nodes if subgraph.degree(node) == 0]
# num_removed_nodes = len(removed_nodes)
subgraph.remove_nodes_from([node for node in subgraph.nodes if subgraph.degree(node) == 0])
# subgraph.remove_nodes_from(set(network.nodes) - sampled_nodes-set(network.neighbors(start_node)))


# nonzero_degree_nodes = [node for node in network.nodes if network.degree(node) > 0]
# random_selected_nodes = random.sample(nonzero_degree_nodes, num_removed_nodes)
# subgraph.add_nodes_from(random_selected_nodes)


# # 绘制原图
# plt.figure(1)
# plot_network(network, node_size_factor=10, font_size=4)
# plt.savefig('1.png',dpi=300)
# plt.figure(2)
# # 绘制采样后的子图
# plot_network(subgraph, node_size_factor=10, font_size=4)
#
# plt.savefig('2.png',dpi=300)
#
# plt.show()

# 计算原图和采样后子图的度分布
network_degrees = [network.degree(node) for node in network.nodes]
subgraph_degrees = [subgraph.degree(node) for node in subgraph.nodes]

# 计算傅里叶变换
network_fft = np.fft.fft(network_degrees)
network_fft = np.fft.fftshift(network_fft)
subgraph_fft = np.fft.fft(subgraph_degrees)
subgraph_fft = np.fft.fftshift(subgraph_fft)


# 计算频率轴
freq_network = np.fft.fftfreq(len(network_degrees))
freq_network = np.fft.fftshift(freq_network)  # 平移频率以匹配 fftshift
freq_subgraph = np.fft.fftfreq(len(subgraph_degrees))
freq_subgraph = np.fft.fftshift(freq_subgraph)

# 绘制功率谱
plt.figure(5)
plt.plot(freq_network, np.abs(network_fft), label='Original Graph')
plt.plot(freq_subgraph, np.abs(subgraph_fft), label='Sampled Subgraph')
plt.yscale('log')  # 使用对数坐标以便更好地可视化
plt.xlabel('Frequency')
plt.ylabel('Power Spectrum')
plt.legend()
# plt.savefig('5.png', dpi=300)
plt.show()


# Function to fit power-law distribution and get the exponent
def fit_power_law(data):
    fit = powerlaw.Fit(data, discrete=True)
    return fit.alpha

# Get degrees of the subgraph nodes
subgraph_degrees = [subgraph.degree(node) for node in subgraph.nodes]
network_degrees = [network.degree(node) for node in network.nodes]
# Calculate the power-law exponent of the subgraph
exponent_subgraph = fit_power_law(subgraph_degrees)
exponent_network = fit_power_law(network_degrees)

# Calculate the average degree of the subgraph
average_degree_subgraph = calculate_average_degree(subgraph)

clustering_coefficient_original = nx.average_clustering(network)

clustering_coefficient_subgraph = nx.average_clustering(subgraph)


# Print the results
print(f"Node with the highest degree: {start_node}, Degree: {network.degree[start_node]}")
print(f"Average Degree of the Network: {calculate_average_degree(network)}")
print(f"Average Degree of the subgraph: {average_degree_subgraph}")
print(f"Power-law exponent of the Network: {exponent_network}")
print(f"Power-law exponent of the subgraph: {exponent_subgraph}")

print(f"Clustering Coefficient of the Original Graph: {clustering_coefficient_original}")
print(f"Clustering Coefficient of the Subgraph: {clustering_coefficient_subgraph}")





