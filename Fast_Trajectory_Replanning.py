from random import randint
import heapq
import sys
import time


class Heap:
    
    def __init__(self, maxsize) -> None:
        self.maxsize = maxsize
        self.size = 0
        self.Heap = [0]*(self.maxsize + 1)
        self.Heap[0] = (-1 * sys.maxsize, 0, node(0,0))
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
        
        if self.Heap[current][0] == self.Heap[self.parent(current)][0]:
            while self.Heap[current][1] < self.Heap[self.parent(current)][1]:
                self.swap(current, self.parent(current))
                current = self.parent(current)
    
    def minHeapify(self, pos):
        if not self.isLeaf(pos):
            if (self.Heap[pos][0] > self.Heap[self.leftchild(pos)][0] or 
                self.Heap[pos][0] > self.Heap[self.rightchild(pos)][0]):
                
                if self.Heap[self.leftchild(pos)][0] == self.Heap[self.rightchild(pos)][0]:
                    if self.Heap[self.leftchild(pos)][1] > self.Heap[self.rightchild(pos)][1]:
                        self.swap(pos, self.leftchild(pos))
                        self.minHeapify(self.leftchild(pos))

                    else:
                        self.swap(pos, self.rightchild(pos))
                        self.minHeapify(self.rightchild(pos))

                elif self.Heap[self.leftchild(pos)][0] < self.Heap[self.rightchild(pos)][0]:
                    self.swap(pos, self.leftchild(pos))
                    self.minHeapify(self.leftchild(pos))

                else:
                    self.swap(pos, self.rightchild(pos))
                    self.minHeapify(self.rightchild(pos))

            elif (self.Heap[pos][0] == self.Heap[self.leftchild(pos)][0] or 
                self.Heap[pos][0] == self.Heap[self.rightchild(pos)][0]):

                if self.Heap[self.leftchild(pos)][0] == self.Heap[self.rightchild(pos)][0]:
                    if self.Heap[self.leftchild(pos)][1] > self.Heap[self.rightchild(pos)][1] and self.Heap[self.leftchild(pos)][1] > self.Heap[pos][1]:
                        self.swap(pos, self.leftchild(pos))
                        self.minHeapify(self.leftchild(pos))

                    elif self.Heap[self.rightchild(pos)][1] > self.Heap[self.rightchild(pos)][1] and self.Heap[self.rightchild(pos)][1] > self.Heap[pos][1]:
                        self.swap(pos, self.rightchild(pos))
                        self.minHeapify(self.rightchild(pos))

                elif self.Heap[pos][0] == self.Heap[self.rightchild(pos)][0]:
                    if self.Heap[self.rightchild(pos)][1] > self.Heap[pos][1]:
                        self.swap(pos, self.rightchild(pos))
                        self.minHeapify(self.rightchild(pos))
                
                else:
                    if self.Heap[self.leftchild(pos)][1] > self.Heap[pos][1]:
                        self.swap(pos, self.leftchild(pos))
                        self.minHeapify(self.leftchild(pos))

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
        self.h = 0
        # self.f = 0
        # self.g = 0
        # self.prev = None
        self.obstacle = False

def A_star(maze, start: node, goal: node):
    #return a path found by A* and the resulting node
    
    # open_list = []
    open_list = Heap(1000)

    # heapq.heappush(open_list, (0, start))
    open_list.push((start.h, 0, start))
    
    closed_list = {}
    closed_list[start] = None
    cost_so_far = {}
    cost_so_far[start] = 0

    while not open_list.isEmpty():
        # (priority, curr) = heapq.heappop(open_list)
        (priority, _, curr) = open_list.pop()

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
                priority = new_g + curr.h
                # heapq.heappush(open_list, (priority, next))
                open_list.push((priority, new_g, next))
                closed_list[next] = curr
    path = []
    curr = goal
    if goal not in closed_list:
        sys.exit("Path doesn't exist!")
    while curr != start:
        path.append(curr)
        curr = closed_list[curr]
    path.append(start)
    return path

def A_star_tie(maze, start: node, goal: node):
    #return a path found by A* and the resulting node
    
    # open_list = []
    open_list = Heap(1000)

    # heapq.heappush(open_list, (0, start))
    open_list.push((start.h, 0, start))
    
    closed_list = {}
    closed_list[start] = None
    cost_so_far = {}
    cost_so_far[start] = 0

    while not open_list.isEmpty():
        # (priority, curr) = heapq.heappop(open_list)
        (priority, _, curr) = open_list.pop()

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
                priority = 203*(new_g + curr.h) - new_g
                # heapq.heappush(open_list, (priority, next))
                open_list.push((priority, new_g, next))
                closed_list[next] = curr
    path = []
    curr = goal
    if goal not in closed_list:
        sys.exit("Path doesn't exist!")
    while curr != start:
        path.append(curr)
        curr = closed_list[curr]
    path.append(start)
    return path

def adaptive_A_star(maze, start: node, goal: node):
    #return a path found by A* and the resulting node
    
    # open_list = []
    open_list = Heap(1000)

    # heapq.heappush(open_list, (0, start))
    open_list.push((start.h, 0, start))
    
    closed_list = {}
    closed_list[start] = None
    cost_so_far = {}
    cost_so_far[start] = 0

    while not open_list.isEmpty():
        # (priority, curr) = heapq.heappop(open_list)
        (priority, _, curr) = open_list.pop()

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
                priority = 203*(new_g + curr.h) - new_g
                # priority = new_g + curr.h

                # heapq.heappush(open_list, (priority, next))
                open_list.push((priority, new_g, next))
                closed_list[next] = curr
    path = []
    curr = goal
    if goal not in closed_list:
        sys.exit("Path doesn't exist!")
    while curr != start:
        path.append(curr)
        curr = closed_list[curr]
    path.append(start)
    return cost_so_far, path

# def h(start: node, goal: node):

#     #get heuristic by calculating manhattan distances
#     return abs(start.x - goal.x) + abs(start.y - goal.y)

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

def generate_maze(size):
    
    #generate a maze by randomly choose 20% of the grid to be obstacles using randint
    agent_maze = []
    maze = []

    for r in range(size):
        agent_maze.append([])
        maze.append([])
        for c in range(size):
            maze[-1].append(0)
            agent_maze[-1].append(0)
    
    for i in range(size):
        for j in range(size):
            agent_maze[i][j] = node(i,j)
            maze[i][j] = node(i,j)
            rand = randint(0,100)
            if rand < 20:
                maze[i][j].obstacle = True
            
    return maze, agent_maze

def display_maze(maze, size, path, goal):
    for i in range(size):
        for j in range(size):
            if maze[i][j] in path:
                print('#', end=' ')
            elif maze[i][j] == goal:
                print('@', end=' ')
            elif maze[i][j].obstacle:
                print('*', end=' ')
            else:
                print('^', end=' ')
        print()
    print()

def main():
    size = 101
    orig_stdout = sys.stdout
    f = open('output.txt', 'w')
    sys.stdout = f
    maze,agent_maze = generate_maze(size)

    #forward A* execution
    curr_start = agent_maze[0][0]
    goal = agent_maze[size-1][size-1]

    if maze[curr_start.x][curr_start.y].obstacle:
        sys.exit('Starting from an obstacle')

    for i in range(size):
        for j in range(size):
            agent_maze[i][j].h = abs(i - goal.x) + abs(j - goal.y)

    path = [curr_start]
    final_path = []
    start_time = time.time()
    print('original maze:')
    display_maze(maze, size, [None], None)
    print()
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
        display_maze(agent_maze, size, path, goal)
        print()
        if curr_start == goal:
            print('found!')
            break
    print('final path:')
    final_path = list(set(final_path))
    display_maze(agent_maze, size, final_path, goal)
    print('excuted forward A*: ')
    print('path length: ' + str(len(final_path)))
    print('Execution time: %s' % (time.time() - start_time))


    # break tie
    curr_start = agent_maze[0][0]
    goal = agent_maze[size-1][size-1]

    if maze[curr_start.x][curr_start.y].obstacle:
        sys.exit('Starting from an obstacle')

    for i in range(size):
        for j in range(size):
            agent_maze[i][j].h = abs(i - goal.x) + abs(j - goal.y)

    path = [curr_start]
    final_path = []
    start_time = time.time()
    print('original maze:')
    display_maze(maze, size, [None], None)
    print()
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
        path = A_star_tie(agent_maze, curr_start, goal)
        print('current state:')
        display_maze(agent_maze, size, path, goal)
        print()
        if curr_start == goal:
            print('found!')
            break
    print('final path:')
    final_path = list(set(final_path))
    display_maze(agent_maze, size, final_path, goal)
    print('executed forward A* break tie: ')
    print('path length: ' + str(len(final_path)))
    print('Execution time: %s' % (time.time() - start_time))

    #Backward A* execution
    _,agent_maze = generate_maze(size)
    curr_start = agent_maze[size-1][size-1]
    goal = agent_maze[0][0]

    if maze[curr_start.x][curr_start.y].obstacle:
        sys.exit('Starting from an obstacle')

    for i in range(size):
        for j in range(size):
            agent_maze[i][j].h = abs(i - goal.x) + abs(j - goal.y)

    path = [curr_start]
    final_path = []
    start_time = time.time()
    print('original maze:')
    display_maze(maze, size, [None], None)
    print()
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
        path = A_star_tie(agent_maze, curr_start, goal)
        print('current state:')
        display_maze(agent_maze, size, path, goal)
        print()
        if curr_start == goal:
            print('found!')
            break
    print('final path:')
    final_path = list(set(final_path))
    display_maze(agent_maze, size, final_path, goal)
    print('executed backward A* break tie')
    print('path length: ' + str(len(final_path)))
    print('Execution time: %s' % (time.time() - start_time))

    #adaptive A* execution
    _,agent_maze = generate_maze(size)
    curr_start = agent_maze[0][0]
    goal = agent_maze[size-1][size-1]

    if maze[curr_start.x][curr_start.y].obstacle:
        sys.exit('Starting from an obstacle')

    for i in range(size):
        for j in range(size):
            agent_maze[i][j].h = abs(i - goal.x) + abs(j - goal.y)

    path = [curr_start]
    final_path = []
    start_time = time.time()
    print('original maze:')
    display_maze(maze, size, [None], None)
    print()
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
        gs, path = adaptive_A_star(agent_maze, curr_start, goal)
        for node in gs:
            # print(agent_maze[node.x][node.y].h)
            agent_maze[node.x][node.y].h = len(path) - gs[node]
            # print(str(node.x) + ' ' + str(node.y) + ' ' + str(agent_maze[node.x][node.y].h))
        print('current state:')
        display_maze(agent_maze, size, path, goal)
        print()
        if curr_start == goal:
            print('found!')
            break
    print('final path:')
    final_path = list(set(final_path))
    display_maze(agent_maze, size, final_path, goal)
    print('executed adaptive A*: ')
    print('path length: ' + str(len(final_path)))
    print('Execution time: %s' % (time.time() - start_time))


    sys.stdout = orig_stdout
    f.close()

if __name__=="__main__":
    main()



# size = 10
# orig_stdout = sys.stdout
# f = open('output.txt', 'w')
# sys.stdout = f
# maze,agent_maze = generate_maze(size)

# #forward A* execution
# curr_start = agent_maze[0][0]
# goal = agent_maze[size-1][size-1]

# if maze[curr_start.x][curr_start.y].obstacle:
#     sys.exit('Starting from an obstacle')

# for i in range(size):
#     for j in range(size):
#         agent_maze[i][j].h = abs(i - goal.x) + abs(j - goal.y)

# path = [curr_start]
# final_path = []
# print('original maze:')
# display_maze(maze, size, [None], None)
# print()
# print('forward A*: ')
# while True:
#     while path:

#         temp = path.pop(-1)
#         if maze[temp.x][temp.y].obstacle:
#             if temp == goal:
#                 curr_start = temp
#                 break
#             else:
#                 agent_maze[temp.x][temp.y].obstacle = maze[temp.x][temp.y].obstacle
#                 break

#         curr_start = temp
#         final_path.append(curr_start)
#         if curr_start.x < len(agent_maze)-1:
#             #add right block
#             agent_maze[curr_start.x+1][curr_start.y].obstacle = maze[curr_start.x+1][curr_start.y].obstacle
#         if curr_start.x > 0:
#             #add left block
#             agent_maze[curr_start.x-1][curr_start.y].obstacle = maze[curr_start.x-1][curr_start.y].obstacle
#         if curr_start.y < len(agent_maze[0])-1:
#             #add down block
#             agent_maze[curr_start.x][curr_start.y+1].obstace = maze[curr_start.x][curr_start.y+1].obstacle
#         if curr_start.y > 0: 
#             #add up block
#             agent_maze[curr_start.x][curr_start.y-1].obstacle = maze[curr_start.x][curr_start.y-1].obstacle
#     path = A_star(agent_maze, curr_start, goal)
#     print('current state:')
#     display_maze(agent_maze, size, path, goal)
#     print()
#     if curr_start == goal:
#         print('found!')
#         break
# print('final path:')
# final_path = list(set(final_path))
# display_maze(agent_maze, size, final_path, goal)
# print('path length: ' + str(len(final_path)))

# #Backward A* execution
# _,agent_maze = generate_maze(size)
# curr_start = agent_maze[size-1][size-1]
# goal = agent_maze[0][0]

# if maze[curr_start.x][curr_start.y].obstacle:
#     sys.exit('Starting from an obstacle')

# for i in range(size):
#     for j in range(size):
#         agent_maze[i][j].h = abs(i - goal.x) + abs(j - goal.y)

# path = [curr_start]
# final_path = []
# print('original maze:')
# display_maze(maze, size, [None], None)
# print()
# print('backward A*: ')
# while True:
#     while path:

#         temp = path.pop(-1)
#         if maze[temp.x][temp.y].obstacle:
#             if temp == goal:
#                 curr_start = temp
#                 break
#             else:
#                 agent_maze[temp.x][temp.y].obstacle = maze[temp.x][temp.y].obstacle
#                 break

#         curr_start = temp
#         final_path.append(curr_start)
#         if curr_start.x < len(agent_maze)-1:
#             #add right block
#             agent_maze[curr_start.x+1][curr_start.y].obstacle = maze[curr_start.x+1][curr_start.y].obstacle
#         if curr_start.x > 0:
#             #add left block
#             agent_maze[curr_start.x-1][curr_start.y].obstacle = maze[curr_start.x-1][curr_start.y].obstacle
#         if curr_start.y < len(agent_maze[0])-1:
#             #add down block
#             agent_maze[curr_start.x][curr_start.y+1].obstace = maze[curr_start.x][curr_start.y+1].obstacle
#         if curr_start.y > 0: 
#             #add up block
#             agent_maze[curr_start.x][curr_start.y-1].obstacle = maze[curr_start.x][curr_start.y-1].obstacle
#     path = A_star(agent_maze, curr_start, goal)
#     print('current state:')
#     display_maze(agent_maze, size, path, goal)
#     print()
#     if curr_start == goal:
#         print('found!')
#         break
# print('final path:')
# final_path = list(set(final_path))
# display_maze(agent_maze, size, final_path, goal)
# print('path length: ' + str(len(final_path)))

# #adaptive A* execution
# _,agent_maze = generate_maze(size)
# curr_start = agent_maze[0][0]
# goal = agent_maze[size-1][size-1]

# if maze[curr_start.x][curr_start.y].obstacle:
#     sys.exit('Starting from an obstacle')

# for i in range(size):
#     for j in range(size):
#         agent_maze[i][j].h = abs(i - goal.x) + abs(j - goal.y)

# path = [curr_start]
# final_path = []
# print('original maze:')
# display_maze(maze, size, [None], None)
# print()
# print('adaptive A*: ')
# while True:
#     while path:

#         temp = path.pop(-1)
#         if maze[temp.x][temp.y].obstacle:
#             if temp == goal:
#                 curr_start = temp
#                 break
#             else:
#                 agent_maze[temp.x][temp.y].obstacle = maze[temp.x][temp.y].obstacle
#                 break

#         curr_start = temp
#         final_path.append(curr_start)
#         if curr_start.x < len(agent_maze)-1:
#             #add right block
#             agent_maze[curr_start.x+1][curr_start.y].obstacle = maze[curr_start.x+1][curr_start.y].obstacle
#         if curr_start.x > 0:
#             #add left block
#             agent_maze[curr_start.x-1][curr_start.y].obstacle = maze[curr_start.x-1][curr_start.y].obstacle
#         if curr_start.y < len(agent_maze[0])-1:
#             #add down block
#             agent_maze[curr_start.x][curr_start.y+1].obstace = maze[curr_start.x][curr_start.y+1].obstacle
#         if curr_start.y > 0: 
#             #add up block
#             agent_maze[curr_start.x][curr_start.y-1].obstacle = maze[curr_start.x][curr_start.y-1].obstacle
#     path = A_star(agent_maze, curr_start, goal)
#     print('current state:')
#     display_maze(agent_maze, size, path, goal)
#     print()
#     if curr_start == goal:
#         print('found!')
#         break
# print('final path:')
# final_path = list(set(final_path))
# display_maze(agent_maze, size, final_path, goal)
# print('path length: ' + str(len(final_path)))

# sys.stdout = orig_stdout
# f.close()