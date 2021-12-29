"""
A* star pathfinding algorithm = informed search algorithm :
finds the shortest path between a start and an end node using a heuristic function

Open set : queue of the nodes we will look at next = neighbours

For a node N : F(N) = G(N) + H(N)
G(N) : current shortest distance from start node to node N
H(N) : heuristic = estimte of distance from node N to end node (euclidian, manhattan...)
We want to look first for the nodes which have the lowest F score

We continuously update the following table :
NODES    F SCORES   G SCORES    H SCORES    PREVIOUS NODE
Update : If new score is lower than current score : update scores
         If G score if lower than current G score : update previous node (we have found a shorter way to reach that node)

Algo main steps :         
0. Remove (0,Start node) from Open set
1. Compute F(N) for all the neighoburs of start node and update table
2. Add (F(N),N) to the open set for these nodes
3. Remove tuple with the lowest F score from the set
4. Repeat 1-2-3 until removed tuple is end node
5. Backtrack from end node to start node to get the shortest path
"""


"""
Evolved version of the program : 
1. Compare A* with other shortest path finding algorithms (Djikstra, BFS, Ford-Bellman single-pair version)
2. Taveling salesman problem (= multiple shortest paths ?)
"""

import pygame
import math 
from queue import PriorityQueue

#Set up the display window
WIDTH = 800
WIN = pygame.display.set_mode ((WIDTH,WIDTH))
pygame.display.set_caption("A* Pathfinding Algorithm")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

class Spot:
    def __init__(self,row,col,width,total_rows):
        self.row = row
        self.col = col
        self.x = row*width
        self.y = col*width
        self.color = WHITE
        self.neighbours = []
        self.width = width
        self.total_rows = total_rows
        
    def get_pos(self):
        return self.row,self.col
        
    #Dtermine if a spot has already been checked
    def is_closed(self):
        return self.color == RED
    #Mark the spot as checked
    def make_closed(self):
        self.color = RED

    #Is the spot in the open set ?
    def is_open(self):
        return self.color == GREEN
    def make_open(self):
        self.color = GREEN

    
    #Is the spot a barrier ?
    def is_barrier(self):
        return self.color == BLACK
    def make_barrier(self):
        self.color = BLACK

    
    #Is the spot the start node ?
    def is_start(self):
        return self.color == ORANGE
    def make_start(self):
        self.color = ORANGE


    #Is the spot the end node ?    
    def is_end(self):
        return self.color == TURQUOISE
    def make_end(self):
        self.color = TURQUOISE
    
    #Is the spot nothing ?    
    def reset(self):
        self.color = WHITE

    #Mark the spot as part of the path
    def make_path(self):
        self.color = PURPLE
  
    #Draw the spot = rectangle
    def draw(self,win):
        pygame.draw.rect(win,self.color,(self.x,self.y,self.width,self.width))
        
    def update_neighbours(self,grid):
        "Update the neighbours of a spot (4 max but not if it's a barrier"
        self.neighbours = []
        if self.row < self.total_rows-1 and not grid[self.row+1][self.col].is_barrier(): #DOWN
            self.neighbours.append(grid[self.row+1][self.col])
        if self.row > 0 and not grid[self.row-1][self.col].is_barrier(): #UP
            self.neighbours.append(grid[self.row-1][self.col])    
        if self.col < self.total_rows-1 and not grid[self.row][self.col+1].is_barrier(): #RIGHT
            self.neighbours.append(grid[self.row][self.col+1])
        if self.col > 0 and not grid[self.row][self.col-1].is_barrier(): #LEFT
            self.neighbours.append(grid[self.row][self.col-1])
    
    
    #Compare a spot to another spot 
    #and always return that the other spot is greater than the current spot
    def __lt__(self,other):
        return False
    
    
def h(p1,p2):
   "Returns the manhattan distance between points p1 and p2"
   x1, y1 = p1
   x2, y2 = p2
   return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(came_from,current,draw):
    "Reconstructs the shrtest path from the end"
    path=[]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.pop()    
    shortest_path = path[::-1] #we construct the path from the start
    for x in shortest_path:   
        x.make_path()
        draw()

def algorithm(draw,grid,start,end):
    count = 0
    open_set = PriorityQueue() #data structure that allows us to get the smallest element out directly
    open_set.put((0,count,start)) #add the count variable to break ties in case two nodes have the same F score
    came_from = {} #dico of paris of nodes (to reconstruct the path later)
    g_score = {spot: float("inf") for row in grid for spot in row} #initialise all g-scores to infinity for all spots
    g_score[start]=0
    f_score = {spot: float("inf") for row in grid for spot in row} #initialise all g-scores to infinity for all spots
    f_score[start]=h(start.get_pos(),end.get_pos())
    
    open_set_hash = {start} #keep track of the spots in the PiorityQueue
    #we can only add or remove objects from the PQ but we cannot know which objects it contains
    
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit() #if we want to quit during the algo is running
        
        current = open_set.get()[2] #get the spot on the top of the PQ
        open_set_hash.remove(current) #syncronise the hash
        
        if current == end: #we are done
            reconstruct_path(came_from,end,draw)
            return True 
        
        for neighbour in current.neighbours:
            temp_g_score = g_score[current]+1
            if temp_g_score < g_score[neighbour]: #if we have found a shorter path to this neighbour
                #update the scores and the came_from dico
                g_score[neighbour] = temp_g_score
                f_score[neighbour] = temp_g_score + h(neighbour.get_pos(),end.get_pos())
                came_from[neighbour] = current
                if neighbour not in open_set_hash: #add the neighbour to the open set
                    count += 1
                    open_set.put((f_score[neighbour],count,neighbour))
                    open_set_hash.add(neighbour)
                    if neighbour != end:
                        neighbour.make_open()
                    
        draw()
        
        if current != start: #we will not check this node again
            current.make_closed()
                
    return False #we cannot find a path from start to end nodes
                
def make_grid(rows,width):
    "The grid is a list of rows, each containing spot objects in that row"
    grid = []
    gap = width//rows #width of each cube
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i,j,gap,rows)
            grid[i].append(spot)
    return grid

def draw_grid(win,rows,width):
    "Draw the grid lines on the window"
    gap = width // rows
    for i in range (rows):
        pygame.draw.line(win,GREY,(0,i*gap),(width,i*gap))
        pygame.draw.line(win,GREY,(i*gap,0),(i*gap,width))
        
# for i in range(rows):
# 		pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
# 		for j in range(rows):
# 			pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))
    
def draw(win,grid,rows,width):
    "Draw everything"
    win.fill(WHITE) #for every new frame : fill the window with white
    #draw all the spots 
    for row in grid:
        for spot in row:
            spot.draw(win)
    #draw the grid lines
    draw_grid(win,rows,width)
    #update the window
    pygame.display.update()
    

def get_clicked_pos (pos,rows,width):
    "Returns the row and col of where the mouse clicked"
    gap = width// rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col

def main(win,width):
    ROWS = 50
    grid = make_grid(ROWS,width)
    start = None
    end = None
    
    run = True #window is open
    
        
    while run:
        draw(win,grid,ROWS,width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                
            if pygame.mouse.get_pressed()[0]: #if left-click
                pos = pygame.mouse.get_pos()
                row,col = get_clicked_pos(pos,ROWS,width)
                spot = grid[row][col]
                if not start and spot != end: #if start is not defined yet
                    start = spot
                    start.make_start()
                elif not end and spot != start: #else if end not defined
                    end = spot
                    end.make_end()
                elif spot != start and spot != end:
                    spot.make_barrier()
                
            elif pygame.mouse.get_pressed()[2]: #if right-click    
                pos = pygame.mouse.get_pos()
                row,col = get_clicked_pos(pos,ROWS,width)
                spot = grid[row][col]                
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None
                    
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    #update all neighbours just before we start the algorithm
                    for row in grid:
                        for spot in row:
                            spot.update_neighbours(grid)
                            
                    algorithm(lambda: draw(win,grid,ROWS,width),grid,start,end)        
                 
                if event.key == pygame.K_c:#restart if we press clear
                    start = None
                    end = None
                    grid = make_grid(ROWS,width)
                        
    pygame.quit()
    
    
main(WIN,WIDTH)