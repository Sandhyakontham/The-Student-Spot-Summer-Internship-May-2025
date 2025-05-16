from collections import deque

# Graph represented as an adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B#pragma once'],
    'E': ['B'],
    'F': ['C']
}

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    traversal = []
    
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            traversal.append(node)
            # Add unvisited neighbors to queue
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
    
    return traversal

def dfs(graph, start, visited=None, traversal=None):
    if visited is None:
        visited = set()
    if traversal is None:
        traversal = []
    
    if start not in visited:
        visited.add(start)
        traversal.append(start)
        # Recursively visit each neighbor
        for neighbor in graph[start]:
            dfs(graph, neighbor, visited, traversal)
    
    return traversal

# Test the algorithms
if __name__ == "__main__":
    print("BFS Traversal:", bfs(graph, 'A'))  # Expected: ['A', 'B', 'C', 'D', 'E', 'F']
    print("DFS Traversal:", dfs(graph, 'A'))  # Expected: ['A', 'B', 'D', 'E', 'C', 'F']