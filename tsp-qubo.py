import numpy as np
import dimod
import matplotlib.pyplot as plt
import networkx as nx

# 1. create 5 random cities
np.random.seed(42)
n_cities = 12
coords = np.random.rand(n_cities, 2) * 10

# 2. distances
distances = np.zeros((n_cities, n_cities))
for i in range(n_cities):
    for j in range(n_cities):
        distances[i,j] = np.linalg.norm(coords[i] - coords[j])


bqm = dimod.BinaryQuadraticModel('BINARY')

# penalty
penalty = 20 * np.max(distances) 

# Constraint 1:
for i in range(n_cities):
    for t in range(n_cities):
    
        bqm.add_variable(f'x_{i}_{t}', -penalty)
    for t1 in range(n_cities):
        for t2 in range(t1+1, n_cities):
            bqm.add_interaction(f'x_{i}_{t1}', f'x_{i}_{t2}', 2 * penalty)

# Constraint 2: 
for t in range(n_cities):
    for i in range(n_cities):
        
        bqm.add_linear(f'x_{i}_{t}', -penalty)
    for i1 in range(n_cities):
        for i2 in range(i1+1, n_cities):
            bqm.add_interaction(f'x_{i1}_{t}', f'x_{i2}_{t}', 2 * penalty)

# Objective Function: 
for i in range(n_cities):
    for j in range(n_cities):
        if i != j:
            for t in range(n_cities):
                bqm.add_interaction(f'x_{i}_{t}', f'x_{j}_{(t+1)%n_cities}', distances[i,j])


sampler = dimod.SimulatedAnnealingSampler() 
sampleset = sampler.sample(bqm, num_reads=500) 
best = sampleset.first.sample

# 5. result
route = [-1] * n_cities
for t in range(n_cities):
    for i in range(n_cities):
        if best[f'x_{i}_{t}'] > 0.5:
            route[t] = i

#y
if -1 in route or len(set(route)) < n_cities:
    print("No 100%/t solution found")
    print("النتيجة الخام الحالية:", route)
else:
    route.append(route[0]) 
    total_dist = sum(distances[route[k], route[k+1]] for k in range(n_cities))
    print("Best route found:", route)
    print(f"Total Distance: {total_dist:.2f}")

    # 6. 
    G = nx.Graph()
    G.add_nodes_from(range(n_cities))
    pos = {k: coords[k] for k in range(n_cities)}
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, font_weight='bold')
    edges = [(route[k], route[k+1]) for k in range(n_cities)]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2)
    
    plt.title(f"TSP Solution via Simulated Annealing - Distance: {total_dist:.2f}")
    plt.savefig('tsp_solution.png')
    plt.show()