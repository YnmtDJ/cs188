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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
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
    # util.raiseNotDefined()

    stack_state = util.Stack()  # 保存尝试遍历的状态
    # 初始时为开始节点
    stack_state.push({"successor": problem.getStartState(), "action": [], "cost": 0})
    visited = {}  # 记录某个节点是否遍历过

    # dfs
    while not stack_state.isEmpty():
        state = stack_state.pop()  # 取出栈顶的作为当前遍历的状态
        successor = state["successor"]  # 当前节点
        action = state["action"]  # 当前走过的路径
        cost = state["cost"]  # 当前付出的代价
        visited[successor] = True  # 标记
        if problem.isGoalState(successor):  # 找到目的地
            return action
        # 考虑邻接节点，加入新的尝试遍历的节点，优先考虑代价小的邻接节点
        edges = problem.getSuccessors(successor)
        edges.sort(key=lambda x: x[2], reverse=True)
        for edge in edges:
            near = edge[0]  # 邻接节点
            direct = edge[1]  # 方向
            edge_cost = edge[2]  # 边代价
            if visited.get(near) is None:  # 没有遍历过
                stack_state.push({"successor": near, "action": action + [direct],
                                  "cost": cost + edge_cost})

    return None  # 没有找到路径


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    que_state = util.PriorityQueue()  # 保存尝试遍历的状态
    # 初始时为开始节点，优先级先考虑深度，然后考虑总代价
    que_state.push({"successor": problem.getStartState(), "action": [], "depth": 0}, (0, 0))
    visited = {problem.getStartState(): True}  # 记录某个节点是否遍历过

    # bfs
    while not que_state.isEmpty():
        state = que_state.pop()  # 取出队头的作为当前遍历的状态
        successor = state["successor"]  # 当前节点
        action = state["action"]  # 当前走过的路径
        depth = state["depth"]  # 当前深度
        # visited[successor] = True  # 标记
        if problem.isGoalState(successor):  # 找到目的地
            return action
        # 考虑邻接节点，加入新的尝试遍历的节点，优先考虑代价小的邻接节点
        for edge in problem.getSuccessors(successor):
            near = edge[0]  # 邻接节点
            direct = edge[1]  # 方向
            edge_cost = edge[2]  # 边代价
            if visited.get(near) is None:  # 没有遍历过
                que_state.push({"successor": near, "action": action + [direct], "depth": depth + 1},
                               (depth + 1, problem.getCostOfActions(action + [direct])))
                visited[near] = True

    return None  # 没有找到路径


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    # 定义state状态类，successor表明节点名称，action表明到达该节点走过的路径
    class State:
        def __init__(self, successor, action):
            self.successor = successor
            self.action = action

        def __eq__(self, other):
            return self.successor == other.successor

    que_state = util.PriorityQueue()  # 保存尝试遍历的状态
    # 初始时为开始节点，优先级f=g+h, g表示已花费的代价, h表示当前节点到目标节点的估算代价
    que_state.push(State(problem.getStartState(), []),
                   heuristic(problem.getStartState(), problem))
    visited = {}  # 记录某个节点是否遍历过

    # A*
    while not que_state.isEmpty():
        state = que_state.pop()  # 取出队头的作为当前遍历的状态
        successor = state.successor  # 当前节点
        action = state.action  # 当前走过的路径
        visited[successor] = True  # 标记
        if problem.isGoalState(successor):  # 找到目的地
            return action
        # 考虑邻接节点，加入新的尝试遍历的节点，优先考虑代价小的邻接节点
        for edge in problem.getSuccessors(successor):
            near = edge[0]  # 邻接节点
            direct = edge[1]  # 方向
            edge_cost = edge[2]  # 边代价
            if visited.get(near) is None:  # 没有遍历过
                que_state.update(State(near, action+[direct]),
                                 problem.getCostOfActions(action + [direct])+heuristic(near, problem))

    return None  # 没有找到路径


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
