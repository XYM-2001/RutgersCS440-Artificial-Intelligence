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
    #return a path found by A* and the resulting node
    
    open_list = []
    heapq.heappush(open_list, (0, start))
    closed_list = {}
    closed_list[start] = None
    cost_so_far = {}
    cost_so_far[start] = 0

    while open_list:
        (priority, curr) = heapq.heappop(open_list)
        display_maze(maze, curr, goal)
        print(len(get_neighbors(maze, curr)))
        if curr == goal:
            #the goal is found
            break
        
        elif not get_neighbors(maze, curr): 
            #run into obstacles
            break

        for next in get_neighbors(maze, curr):
            new_g = cost_so_far[curr] + 1
            if next not in cost_so_far or new_g < cost_so_far[next]:
                cost_so_far[next] = new_g
                priority = new_g + h(curr,goal)
                heapq.heappush(open_list, (priority, next))
                closed_list[next] = curr
    print('?')
    end = curr
    path = []
    curr = goal
    while curr != start:
        path.append(curr)
        curr = closed_list[curr]
    path.append(start)
    path.reverse
    return path, end

def h(start: node, goal: node):

    #get heuristic by calculating manhattan distances
    return abs(start.x - goal.x) + abs(start.y - goal.y)

def get_neighbors(maze, agent: node):

    #return valid neighbors of current agent in four directions
    neighbors = []
    if agent.x < len(maze)-1:
        if not maze[agent.x+1][agent.y].obstacle :
            neighbors.append(maze[agent.x+1][agent.y])
    if agent.x > 0:
        if not maze[agent.x-1][agent.y]: 
            neighbors.append(maze[agent.x-1][agent.y])
    if agent.y < len(maze[0])-1:
        if not maze[agent.x][agent.y+1]: 
            neighbors.append(maze[agent.x][agent.y+1])
    if agent.y > 0: 
        if not maze[agent.x][agent.y-1]: 
            neighbors.append(maze[agent.x][agent.y-1])
    return neighbors

def generate_maze(rows: int, cols: int):
    
    #generate a maze by randomly choose 20% of the grid to be obstacles using randint
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

def display_maze(maze, current, goal):
    for r in maze:
        for c in r:
            if c == current:
                print('#', end=' ')
            elif c == goal:
                print('@', end=' ')
            elif c.obstacle:
                print('*', end=' ')
            else:
                print('^', end=' ')
        print()
    print()
maze = generate_maze(10,10)
display_maze(maze, maze[0][0], maze[9][9])
A_star(maze, maze[0][0], maze[9][9])