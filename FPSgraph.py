import matplotlib
matplotlib.use('TkAgg')


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 创建无向图（使用Barabási–Albert模型）
def create_power_law_graph(n, m, gamma, average_degree):
    G = nx.barabasi_albert_graph(n, m)
    return G

# 固定节点在平面内的随机分布
def fixed_layout(G, seed=42):
    pos = nx.random_layout(G, seed=seed)
    return pos

# 获取节点编号和对应的度数，并计算平均度数
def get_node_degrees(G):
    degrees = dict(G.degree())
    average_degree = np.mean(list(degrees.values()))
    return degrees, average_degree

# 绘制度数的概率密度图
def plot_degree_distribution_density(G):
    degrees = dict(G.degree())
    degree_values = list(degrees.values())

    plt.hist(degree_values, bins=50, density=True, alpha=0.75, edgecolor='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Degree Distribution (Density Plot)")
    plt.xlabel("Degree")
    plt.ylabel("Probability Density")
    plt.show()

# 主程序
if __name__ == "__main__":
    num_nodes = 10000
    m = 5  # 每个新节点连接到m个现有节点
    gamma = 2
    average_degree = 2 * m  # Barabási–Albert 模型的平均度数为2m

    # 创建幂率分布的无向图
    graph = create_power_law_graph(num_nodes, m, gamma, average_degree)

    # 固定节点在平面内的随机分布
    pos = fixed_layout(graph)

    # 获取节点编号和对应的度数，并计算平均度数
    node_degrees, avg_degree = get_node_degrees(graph)
    print("Node ID \t Degree")
    for node, degree in node_degrees.items():
        print(f"{node} \t\t {degree}")

    print(f"\nAverage Degree: {avg_degree}")

    # 绘制度数的概率密度图
    plot_degree_distribution_density(graph)
