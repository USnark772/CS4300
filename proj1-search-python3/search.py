# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from queue import Queue
import heapq

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class MyPriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    return False
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                return True
        else:
            self.push(item, priority)
            return True

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    dfs_stack = util.Stack()
    visited = []
    start = problem.getStartState()
    came_from = dict()
    start_tup = (start, "", 0)
    dfs_stack.push(start_tup)
    next_node = None
    while not dfs_stack.isEmpty():
        next_node = dfs_stack.pop()
        state = next_node[0]
        if problem.isGoalState(state):
            break
        visited.append(state)
        for successor in problem.getSuccessors(state):
            if successor[0] not in visited:
                dfs_stack.push(successor)
                came_from[successor] = next_node
    ret = get_path(next_node, came_from)
    return ret

def get_path(last_node, came_from, node_info=None):
    result = []
    if not node_info:
        result.append(last_node[1])
        parent = came_from[last_node]
        while not parent[1] == "":
            result.append(parent[1])
            parent = came_from[parent]
    elif node_info:
        info = node_info[last_node]
        result.append(info[1])
        parent_name = came_from[last_node]
        parent_info = node_info[parent_name]
        while not parent_info[1] == '':
            result.append(parent_info[1])
            parent_name = came_from[parent_name]
            parent_info = node_info[parent_name]
    result.reverse()
    return result

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    frontier = Queue()
    visited = []
    came_from = dict()
    start_state = problem.getStartState()
    frontier.put((start_state, "", 0))
    states = [start_state]
    next_node = None
    while not frontier.empty():
        next_node = frontier.get()
        # print("getting next node:", next_node)
        try:
            states.remove(next_node[0])
        except:
            pass
        visited.append(next_node[0])
        if problem.isGoalState(next_node[0]):
            # print("found goal state:", next_node) 
            break
        for successor in problem.getSuccessors(next_node[0]):
            if successor[0] not in visited and successor[0] not in states:
                frontier.put(successor)
                states.append(successor[0])
                came_from[successor] = next_node
    return get_path(next_node, came_from)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontier = MyPriorityQueue()
    node_info = dict()
    came_from = dict()
    explored = []
    state_to_check = None
    start_state = problem.getStartState()
    node_info[start_state] = (start_state, '', 0)
    frontier.push(start_state, 0)
    states_on_frontier = [start_state]
    while not frontier.isEmpty():
        state_to_check = frontier.pop()
        info = node_info[state_to_check]
        try:
            states_on_frontier.remove(state_to_check)
        except:
            pass
        explored.append(state_to_check)
        if problem.isGoalState(state_to_check):
            break
        for successor in problem.getSuccessors(state_to_check):
            if successor[0] not in explored:
                if frontier.update(successor[0], info[2] + successor[2]):
                    came_from[successor[0]] = state_to_check
                    node_info[successor[0]] = (successor[0], successor[1], info[2] + successor[2])
    return get_path(state_to_check, came_from, node_info)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier = MyPriorityQueue()
    node_info = dict()
    came_from = dict()
    explored = []
    state_to_check = None
    start_state = problem.getStartState()
    node_info[start_state] = (start_state, '', 0)
    frontier.push(start_state, 0 + heuristic(start_state, problem))
    states_on_frontier = [start_state]
    while not frontier.isEmpty():
        state_to_check = frontier.pop()
        info = node_info[state_to_check]
        try:
            states_on_frontier.remove(state_to_check)
        except:
            pass
        explored.append(state_to_check)
        if problem.isGoalState(state_to_check):
            break
        for successor in problem.getSuccessors(state_to_check):
            if successor[0] not in explored:
                if frontier.update(successor[0], info[2] + successor[2] + heuristic(successor[0], problem)):
                    came_from[successor[0]] = state_to_check
                    node_info[successor[0]] = (successor[0], successor[1], info[2] + successor[2])
    return get_path(state_to_check, came_from, node_info)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
