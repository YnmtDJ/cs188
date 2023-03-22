# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        action, score = self.MAX_VALUE(gameState, 0)
        return action

    def MAX_VALUE(self, gameState: GameState, depth):
        """
        考虑吃豆人所有合法的移动。在鬼足够聪明的情况下，
        选择一个移动，使得能够获取最大分数。
        并返回action, score
        :param gameState: 当前游戏状态
        :param depth: 吃豆人与鬼已经移动的步数(吃豆人与鬼都移动1次, depth+1)
        :return action, score: 移动方向，能获取的最大分数
        """
        if depth == self.depth or gameState.isWin() or gameState.isLose():  # 直接返回
            return None, self.evaluationFunction(gameState)
        max_score = -999999  # 记录最大分数
        best_action = None  # 最佳移动方向
        for action in gameState.getLegalActions(0):  # 考虑吃豆人所有可能的移动
            _, score = self.MIN_VALUE(gameState.generateSuccessor(0, action), depth, 1)
            if score > max_score:  # 更新最大值
                max_score = score
                best_action = action
        return best_action, max_score

    def MIN_VALUE(self, gameState: GameState, depth, agentIndex):
        """
        考虑鬼的所有合法移动。在其他鬼、吃豆人足够聪明的情况下，
        选择一个移动，使得能够获取最小的分数。
        并返回action, score
        :param gameState: 当前游戏状态
        :param depth: 吃豆人与鬼已经移动的步数(吃豆人与鬼都移动1次, depth+1)
        :param agentIndex: 鬼的索引(agentIndex>=1)
        :return action, score: 移动方向，能获取的最小分数
        """
        if depth == self.depth or gameState.isLose() or gameState.isWin():  # 直接返回
            return None, self.evaluationFunction(gameState)
        min_score = 999999  # 记录最小分数
        best_action = None  # 最佳移动方向
        for action in gameState.getLegalActions(agentIndex):  # 考虑鬼所有可能的移动
            if agentIndex < gameState.getNumAgents() - 1:  # 还有鬼未移动
                _, score = self.MIN_VALUE(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            else:  # 轮到吃豆人移动
                _, score = self.MAX_VALUE(gameState.generateSuccessor(agentIndex, action), depth + 1)
            if score < min_score:  # 更新最小值
                min_score = score
                best_action = action
        return best_action, min_score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        action, score = self.MAX_VALUE(gameState, 0, -999999, 999999)
        return action

    def MAX_VALUE(self, gameState: GameState, depth, alpha, beta):
        """
        考虑吃豆人所有合法的移动。在鬼足够聪明的情况下，
        选择一个移动，使得能够获取最大分数。
        并返回action, score
        :param gameState: 当前游戏状态
        :param depth: 吃豆人与鬼已经移动的步数(吃豆人与鬼都移动1次, depth+1)
        :param alpha: 目前吃豆人已经能够获取的最大分数
        :param beta: 目前鬼能够获取的最小分数
        :return action, score: 移动方向，能获取的最大分数
        """
        if depth == self.depth or gameState.isWin() or gameState.isLose():  # 直接返回
            return None, self.evaluationFunction(gameState)
        max_score = -999999  # 记录最大分数
        best_action = None  # 最佳移动方向
        for action in gameState.getLegalActions(0):  # 考虑吃豆人所有可能的移动
            _, score = self.MIN_VALUE(gameState.generateSuccessor(0, action), depth, 1, alpha, beta)
            if score > max_score:  # 更新最大值
                max_score = score
                best_action = action
                if max_score > beta:  # 剪枝
                    break
            alpha = max(alpha, max_score)
        return best_action, max_score

    def MIN_VALUE(self, gameState: GameState, depth, agentIndex, alpha, beta):
        """
        考虑鬼的所有合法移动。在其他鬼、吃豆人足够聪明的情况下，
        选择一个移动，使得能够获取最小的分数。
        并返回action, score
        :param gameState: 当前游戏状态
        :param depth: 吃豆人与鬼已经移动的步数(吃豆人与鬼都移动1次, depth+1)
        :param agentIndex: 鬼的索引(agentIndex>=1)
        :param alpha: 目前吃豆人已经能够获取的最大分数
        :param beta: 目前鬼能够获取的最小分数
        :return action, score: 移动方向，能获取的最小分数
        """
        if depth == self.depth or gameState.isLose() or gameState.isWin():  # 直接返回
            return None, self.evaluationFunction(gameState)
        min_score = 999999  # 记录最小分数
        best_action = None  # 最佳移动方向
        for action in gameState.getLegalActions(agentIndex):  # 考虑鬼所有可能的移动
            if agentIndex < gameState.getNumAgents() - 1:  # 还有鬼未移动
                _, score = self.MIN_VALUE(gameState.generateSuccessor(agentIndex, action),
                                          depth, agentIndex + 1, alpha, beta)
            else:  # 轮到吃豆人移动
                _, score = self.MAX_VALUE(gameState.generateSuccessor(agentIndex, action),
                                          depth + 1, alpha, beta)
            if score < min_score:  # 更新最小值
                min_score = score
                best_action = action
                if min_score < alpha:  # 剪枝
                    break
            beta = min(beta, min_score)
        return best_action, min_score


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        action, score = self.MAX_VALUE(gameState, 0)
        return action

    def MAX_VALUE(self, gameState: GameState, depth):
        """
        考虑吃豆人所有合法的移动。在鬼随机游走的情况下，
        选择一个移动，使得能够获取最大分数。
        并返回action, score
        :param gameState: 当前游戏状态
        :param depth: 吃豆人与鬼已经移动的步数(吃豆人与鬼都移动1次, depth+1)
        :return action, score: 移动方向，能获取的最大分数
        """
        if depth == self.depth or gameState.isWin() or gameState.isLose():  # 直接返回
            return None, self.evaluationFunction(gameState)
        max_score = -999999  # 记录最大分数
        best_action = None  # 最佳移动方向
        for action in gameState.getLegalActions(0):  # 考虑吃豆人所有可能的移动
            _, score = self.MIN_VALUE(gameState.generateSuccessor(0, action), depth, 1)
            if score > max_score:  # 更新最大值
                max_score = score
                best_action = action
        return best_action, max_score

    def MIN_VALUE(self, gameState: GameState, depth, agentIndex):
        """
        考虑鬼的所有合法移动。
        鬼每次随机选择一个移动，计算其能够获得的分数的期望。
        并返回action, score
        :param gameState: 当前游戏状态
        :param depth: 吃豆人与鬼已经移动的步数(吃豆人与鬼都移动1次, depth+1)
        :param agentIndex: 鬼的索引(agentIndex>=1)
        :return action, score: 移动方向，能获取的分数的期望
        """
        if depth == self.depth or gameState.isLose() or gameState.isWin():  # 直接返回
            return None, self.evaluationFunction(gameState)
        avg_score = 0.
        for action in gameState.getLegalActions(agentIndex):  # 考虑鬼的所有合法移动
            if agentIndex < gameState.getNumAgents() - 1:  # 还有鬼未移动
                _, score = self.MIN_VALUE(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            else:  # 轮到吃豆人移动
                _, score = self.MAX_VALUE(gameState.generateSuccessor(agentIndex, action), depth + 1)
            avg_score += score
        return None, avg_score / len(gameState.getLegalActions(agentIndex))


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    分数从三方面考虑：目标距离，剩余食物，已经移动的时间
    目标距离：目标以可是食物，胶囊，处于害怕状态的鬼。计算吃豆人与它们距离的最小值，用于计算分数。距离越小分数越高。
    剩余食物：游戏中若食物越少，说明越靠近目标状态，故分数越高。
    已经移动时间：想要在更短的时间内获胜，故移动时间越长分数越低。
    score = width + height - min_dis + currentGameState.getScore()
    其中currentGameState.getScore()是提供的接口函数，它综合考虑了剩余食物，已经移动时间两个方面进行评分。
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # 一些需要的信息
    walls = currentGameState.getWalls()
    height = walls.height
    width = walls.width
    foods = currentGameState.getFood()
    ghosts = currentGameState.getGhostPositions()
    ghosts_state = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    pacman = currentGameState.getPacmanPosition()
    alpha = 8  # 与鬼距离阈值，当与鬼接近时才去吃胶囊

    # 直接返回
    if currentGameState.isLose():
        return 0
    if currentGameState.isWin():
        return width + height + currentGameState.getScore()

    # 估计与鬼的距离
    ghost_dis = 999999
    for ghost, ghost_state in zip(ghosts, ghosts_state):
        if ghost_state.scaredTimer == 0:
            ghost_dis = min(ghost_dis, abs(pacman[0] - ghost[0]) + abs(pacman[1] - ghost[1]))

    # 距离计算1：bfs计算真实距离
    que = util.Queue()
    que.push((pacman, 0))
    visited = {pacman: True}
    min_dis = width*height
    while not que.isEmpty():
        state = que.pop()
        pos = state[0]
        depth = state[1]
        # 遇到食物
        if foods[pos[0]][pos[1]]:
            min_dis = depth
            break
        # 遇到胶囊
        i = 0
        while i < len(capsules):
            if capsules[i] == pos and ghost_dis < alpha:  # 当与鬼接近时才去吃胶囊
                min_dis = depth
                break
            i += 1
        if i < len(capsules): break
        # 遇到害怕的鬼
        i = 0
        while i < len(ghosts):
            if ghosts_state[i].scaredTimer > 0 and ghosts[i] == pos:
                min_dis = depth
                break
            i += 1
        if i < len(ghosts): break
        # 继续搜索
        directs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for direct in directs:
            next_pos = (pos[0]+direct[0], pos[1]+direct[1])
            if visited.get(next_pos) is None and (not walls[next_pos[0]][next_pos[1]]):
                que.push((next_pos, depth + 1))
                visited[next_pos] = True

    dis_score = width + height - min_dis
    return dis_score + currentGameState.getScore()


# Abbreviation
better = betterEvaluationFunction
