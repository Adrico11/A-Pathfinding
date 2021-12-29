### Introduction to the project : 

#### A* pathfinding algorithm 
Informal definition :  informed search algorithm which finds the shortest path between a start and an end node using a heuristic function

#### This project is an interactive visualisation tool of the A* pathfinding algorithm : 
1. Code the actual algorithm using PriorityQueue as our main datastructures (see below for more details on the algorithm)
2. Integrate it into an interactive tool (Pygame) : possibility to choose the start and end positions in a grid as well as draw barriers on the grid to make it more interesting!

### Algorithm details and main implementation steps : 

#### Fundamentals of the algorithm :

For a node N : F(N) = G(N) + H(N) with 
F(N) : total F score for this node
G(N) : current shortest distance from start node to node N
H(N) : heuristic = estimte of distance from node N to end node (euclidian, manhattan...)
We want to look first for the nodes which have the lowest F score

We continuously update the following table :
NODES    F SCORES   G SCORES    H SCORES    PREVIOUS NODE

Update : If new score is lower than current score : update scores
         If G score if lower than current G score : update previous node (we have found a shorter way to reach that node)

#### Algorithm main steps :         
0. Remove (0,Start node) from Open set
1. Compute F(N) for all the neighoburs of start node and update table
2. Add (F(N),N) to the open set for these nodes
3. Remove tuple with the lowest F score from the set
4. Repeat 1-2-3 until removed tuple is end node
5. Backtrack from end node to start node to get the shortest path


### Evolved version of the project : 
1. Compare A* with other shortest path finding algorithms (Djikstra, BFS, Ford-Bellman single-pair version)
2. Taveling salesman problem (= multiple shortest paths ?)
