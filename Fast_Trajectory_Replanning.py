from random import randint
import heapq
import sys

class Heap:
    
    def __init__(self, maxsize) -> None:
        self.maxsize = maxsize
        self.size = 0
        self.Heap = [0]*(self.maxsize + 1)
        self.Heap[0] = (-1 * sys.maxsize, node(0,0))
        self.FRONT = 1
    
    def parent(self, pos):
        return pos//2

    def leftchild(self, pos):
        return 2*pos

    def rightchild(self, pos):
        return (2*pos)+1
    
    def isLeaf(self, pos):
        return pos*2 > self.size

    def swap(self, fpos, spos):
        self.Heap[fpos], self.Heap[spos] = self.Heap[spos], self.Heap[fpos]

    def push(self, element):

        if self.size >= self.maxsize:
            return 
        self.size += 1
        self.Heap[self.size] = element
        
        current = self.size

        while self.Heap[current][0] < self.Heap[self.parent(current)][0]:
            self.swap(current, self.parent(current))
            current = self.parent(current)
    
    def minHeapify(self, pos):
        if not self.isLeaf(pos):
            if (self.Heap[pos][0] > self.Heap[self.leftchild(pos)][0] or 
                self.Heap[pos][0] > self.Heap[self.rightchild(pos)][0]):
                
                if self.Heap[self.leftchild(pos)][0] < self.Heap[self.rightchild(pos)][0]:
                    self.swap(pos, self.leftchild(pos))
                    self.minHeapify(self.leftchild(pos))

                else:
                    self.swap(pos, self.rightchild(pos))
                    self.minHeapify(self.rightchild(pos))

    def isEmpty(self):
        if self.size == 0:
            return True
        return False

    def pop(self):

        popped = self.Heap[self.FRONT]
        self.Heap[self.FRONT] = self.Heap[self.size]
        self.size -= 1
        self.minHeapify(self.FRONT)
        return popped

    def Print(self):
        for i in range(1, (self.size//2)+1):
            print(' parent : ' + str(self.Heap[i][0]) + ' left child : ' + 
                                str(self.Heap[2*i][0]) + ' right child : '+ 
                                str(self.Heap[2*i+i][0]))
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
    
    # open_list = []
    open_list = Heap(1000)

    # heapq.heappush(open_list, (0, start))
    open_list.push((h(start, goal), start))
    
    closed_list = {}
    closed_list[start] = None
    cost_so_far = {}
    cost_so_far[start] = 0

    while not open_list.isEmpty():
        # (priority, curr) = heapq.heappop(open_list)
        (priority, curr) = open_list.pop()

        # display_maze(maze, curr, goal)
        #print(len(get_neighbors(maze, curr)))
        if curr == goal:
            #the goal is found
            break
        
        # elif not get_neighbors(maze, curr): 
        #     #run into obstacles
        #     break

        for next in get_neighbors(maze, curr):
            new_g = cost_so_far[curr] + 1
            if next not in cost_so_far or new_g < cost_so_far[next]:
                cost_so_far[next] = new_g
                priority = new_g + h(curr,goal)
                # heapq.heappush(open_list, (priority, next))
                open_list.push((priority, next))
                closed_list[next] = curr
    end = curr
    path = []
    curr = goal
    while curr != start:
        path.append(curr)
        curr = closed_list[curr]
    path.append(start)
    return path

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
        if not maze[agent.x-1][agent.y].obstacle: 
            neighbors.append(maze[agent.x-1][agent.y])
    if agent.y < len(maze[0])-1:
        if not maze[agent.x][agent.y+1].obstacle: 
            neighbors.append(maze[agent.x][agent.y+1])
    if agent.y > 0: 
        if not maze[agent.x][agent.y-1].obstacle: 
            neighbors.append(maze[agent.x][agent.y-1])
    return neighbors

def generate_maze(rows: int, cols: int):
    
    #generate a maze by randomly choose 20% of the grid to be obstacles using randint
    agent_maze = []
    maze = []

    for r in range(rows):
        agent_maze.append([])
        maze.append([])
        for c in range(cols):
            maze[-1].append(0)
            agent_maze[-1].append(0)
    
    for i in range(rows):
        for j in range(cols):
            agent_maze[i][j] = node(i,j)
            maze[i][j] = node(i,j)
            rand = randint(0,100)
            if rand < 20:
                maze[i][j].obstacle = True
            
    return maze, agent_maze

def display_maze(maze, path, goal):
    for r in maze:
        for c in r:
            if c in path:
                print('#', end=' ')
            elif c == goal:
                print('@', end=' ')
            elif c.obstacle:
                print('*', end=' ')
            else:
                print('^', end=' ')
        print()
    print()

# def main():
#     args = sys.argv[1:]
#     maze_size = args[0]

# if __name__=="__main__":
#     main()

orig_stdout = sys.stdout
f = open('output.txt', 'w')
sys.stdout = f
maze,agent_maze = generate_maze(10,10)

#forward A* execution
curr_start = agent_maze[0][0]
goal = agent_maze[9][9]

if maze[curr_start.x][curr_start.y].obstacle:
    sys.exit('Starting from an obstacle')

path = [curr_start]
final_path = []
print('original maze:')
display_maze(maze, [None], None)
print()
print('forward A*: ')
while True:
    while path:

        temp = path.pop(-1)
        if maze[temp.x][temp.y].obstacle:
            if temp == goal:
                curr_start = temp
                break
            else:
                agent_maze[temp.x][temp.y].obstacle = maze[temp.x][temp.y].obstacle
                break

        curr_start = temp
        final_path.append(curr_start)
        if curr_start.x < len(agent_maze)-1:
            #add right block
            agent_maze[curr_start.x+1][curr_start.y].obstacle = maze[curr_start.x+1][curr_start.y].obstacle
        if curr_start.x > 0:
            #add left block
            agent_maze[curr_start.x-1][curr_start.y].obstacle = maze[curr_start.x-1][curr_start.y].obstacle
        if curr_start.y < len(agent_maze[0])-1:
            #add down block
            agent_maze[curr_start.x][curr_start.y+1].obstace = maze[curr_start.x][curr_start.y+1].obstacle
        if curr_start.y > 0: 
            #add up block
            agent_maze[curr_start.x][curr_start.y-1].obstacle = maze[curr_start.x][curr_start.y-1].obstacle
    path = A_star(agent_maze, curr_start, goal)
    print('current state:')
    display_maze(agent_maze, path, goal)
    print()
    if curr_start == goal:
        print('found!')
        break
print('final path:')
final_path = list(set(final_path))
display_maze(agent_maze, final_path, goal)
print('path length: ' + str(len(final_path)))

#Backward A* execution
_,agent_maze = generate_maze(10,10)
curr_start = agent_maze[9][9]
goal = agent_maze[0][0]

if maze[curr_start.x][curr_start.y].obstacle:
    sys.exit('Starting from an obstacle')

path = [curr_start]
final_path = []
print('original maze:')
display_maze(maze, [None], None)
print()
print('backward A*: ')
while True:
    while path:

        temp = path.pop(-1)
        if maze[temp.x][temp.y].obstacle:
            if temp == goal:
                curr_start = temp
                break
            else:
                agent_maze[temp.x][temp.y].obstacle = maze[temp.x][temp.y].obstacle
                break

        curr_start = temp
        final_path.append(curr_start)
        if curr_start.x < len(agent_maze)-1:
            #add right block
            agent_maze[curr_start.x+1][curr_start.y].obstacle = maze[curr_start.x+1][curr_start.y].obstacle
        if curr_start.x > 0:
            #add left block
            agent_maze[curr_start.x-1][curr_start.y].obstacle = maze[curr_start.x-1][curr_start.y].obstacle
        if curr_start.y < len(agent_maze[0])-1:
            #add down block
            agent_maze[curr_start.x][curr_start.y+1].obstace = maze[curr_start.x][curr_start.y+1].obstacle
        if curr_start.y > 0: 
            #add up block
            agent_maze[curr_start.x][curr_start.y-1].obstacle = maze[curr_start.x][curr_start.y-1].obstacle
    path = A_star(agent_maze, curr_start, goal)
    print('current state:')
    display_maze(agent_maze, path, goal)
    print()
    if curr_start == goal:
        print('found!')
        break
print('final path:')
final_path = list(set(final_path))
display_maze(agent_maze, final_path, goal)
print('path length: ' + str(len(final_path)))

sys.stdout = orig_stdout
f.close()