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
    cost_so_far = {}
    closed_list[start] = None
    cost_so_far[start] = 0

    while not open_list:
        (_, curr) = heapq.heappop(open_list)

        if curr == goal:
            #the goal is found
            break
        
        elif not get_neighbors(curr): 
            #run into obstacles
            break

        for next in get_neighbors(curr):
            new_cost = cost_so_far[curr] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + h(curr,goal)
                heapq.heappush(open_list, (priority, next))
                closed_list[next] = curr
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