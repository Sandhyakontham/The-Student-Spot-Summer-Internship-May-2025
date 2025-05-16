# The Student Spot Summer Internship May 2025

## Search Algorithms Implementation

This repository contains implementations of two fundamental graph traversal algorithms:
- Breadth-First Search (BFS)
- Depth-First Search (DFS)

### Folder Structure
```
The-Student-Spot-Summer-Internship-May-2025/
├── src/
│   └── search_algorithms.py
├── README.md
```

### Files Description
- `src/search_algorithms.py`: Contains the Python implementation of both BFS and DFS algorithms with a Graph class for testing.
- `README.md`: This file, providing an overview of the project and instructions.

### Implementation Details
- **Graph Class**: Represents an undirected graph using an adjacency list.
- **BFS**: Uses a queue (via `collections.deque`) to explore nodes level by level.
- **DFS**: Uses a stack (via list) for iterative depth-first traversal.
- Both algorithms return a list of nodes in the order they were visited.
- The code includes type hints and docstrings for better readability.
- Example usage is provided at the bottom of `search_algorithms.py`.

### How to Run
1. Ensure Python 3.8+ is installed.
2. Clone this repository:
   ```
   git clone <repository-url>
   cd The-Student-Spot-Summer-Internship-May-2025
   ```
3. Run the script:
   ```
   python src/search_algorithms.py
   ```
4. Sample output will show BFS and DFS traversal results for a test graph.

### Sample Graph
The test graph has the following structure:
```
    A
   / \
  B   C
  |   |
  D---E
```

### Output
Running the script will produce:
```
BFS traversal starting from node A: ['A', 'B', 'C', 'D', 'E']
DFS traversal starting from node A: ['A', 'C', 'E', 'D', 'B']
```

### Dependencies
- Python standard library (no external dependencies required)

### Notes
- The implementation uses iterative approaches for both algorithms to avoid recursion stack issues with large graphs.
- The code is modular and can be easily extended for directed graphs or weighted edges.
- Proper error handling and input validation can be added for production use.