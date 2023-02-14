from random import randint
import heapq

class node:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.f = 0
        self.g = 0
        self.h = 0
        self.prev = None
        self.obstacle = False

def A_star(maze, start, goal):

    open_list = []
    heapq.heappush()
    closed_list = {}
    cost_so_far = {}
    closed_list[start] = None
    cost_so_far[start] = 0

    open_list[start_node] = None
    current_node = start_node
    while len(open_list) > 0:
        for key in open_list:
            if key.f < 
    return None

def generate_maze(rows, cols):
    maze = []

    for r in range(rows):
        maze.append([])
        for c in range(cols):
            maze[-1].append(0)
    
    for i in range(rows):
        for j in range(cols):
            maze[i][j] = node(i,j)
            rand = randint(0,100)
            if rand < 20:
                maze[i][j].obstacle = True
            
    return maze

def display_maze(maze):
    for r in maze:
        for c in r:
            if c.obstacle:
                print('*', end=' ')
            else:
                print('^', end=' ')
        print()
maze = generate_maze(10,10)
display_maze(maze)
temp = []
heapq.heappush(temp, (0,maze[0][0]))
print(maze[0][0].x)
print(maze[0][0].y)
heapq.heappush(temp,(1,maze[0][1]))
print(list(temp))
(_,a)=heapq.heappop(temp)
print(a.x)
print(a.y)