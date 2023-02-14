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

def A_star(maze, start: node, goal: node):

    open_list = []
    heapq.heappush(open_list, (0, start))
    closed_list = {}
    cost_so_far = {}
    closed_list[start] = None
    cost_so_far[start] = 0

    while not open_list:
        (_, curr) = heapq.heappop(open_list)
        if curr == goal:
            break
    return None

def get_neighbors(maze, agent: node):

    neighbors = []
    if agent.x < len(maze)-1:
        neighbors.append(maze[agent.x+1][agent.y])
    if agent.x > 0:
        neighbors.append(maze[agent.x-1][agent.y])

def generate_maze(rows: int, cols: int):

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
heapq.heappush(temp,(1,maze[0][1]))
(_,a)=heapq.heappop(temp)
print(len(temp))
(_,a)=heapq.heappop(temp)