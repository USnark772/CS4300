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
import random
import util
from time import sleep
from pacman import GameState

from game import Agent, AgentState


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
        # print(legalMoves)
        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        # print(scores)
        bestScore = max(scores)
        # print(bestScore)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # print(bestIndices)
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)
        # print(chosenIndex)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        if action == "  ":
            return 0 - 10000000000
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        curPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        # newFood = successorGameState.getFood()
        curFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"

        retScore = successorGameState.getScore()
        posToCheck = [(newPos[0] + 1, newPos[1]), (newPos[0] - 1, newPos[1]),
                      (newPos[0], newPos[1] + 1), (newPos[0], newPos[1] - 1), newPos]
        posToCheck = list(set(posToCheck) - set(curPos))

        foodList = curFood.asList()
        if len(foodList) == 0:
            retScore = 0
        else:
            bestDist = None
            for dot in foodList:
                distToDot = manhattanDistance(newPos, dot) + len(foodList)
                if not bestDist:
                    bestDist = distToDot
                elif distToDot < bestDist:
                    bestDist = distToDot
            retScore = 0 - bestDist

        for x in newGhostStates:
            if x.getPosition() in posToCheck:
                retScore = 0 - 100000000000
        return retScore


def scoreEvaluationFunction(currentGameState):
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
        _max = float("-inf")
        action = None
        for move in gameState.getLegalActions(0):
            util = minimax(self.evaluationFunction, 1, 0,
                           gameState.generateSuccessor(0, move), self.depth)
            if util > _max or _max == float("-inf"):
                _max = util
                action = move

        return action


def minimax(evalFunc: classmethod, agent: int, depth: int, gameState: GameState, maxDepth: int) -> float:
    if gameState.isLose() or gameState.isWin() or depth == maxDepth:
        return evalFunc(gameState)
    if agent == 0:
        return max(minimax(evalFunc, 1, depth, gameState.generateSuccessor(agent, state), maxDepth) for state in gameState.getLegalActions(agent))
    else:
        nextAgent = agent + 1
        if gameState.getNumAgents() == nextAgent:
            nextAgent = 0
        if nextAgent == 0:
            depth += 1
        return min(minimax(evalFunc, nextAgent, depth, gameState.generateSuccessor(agent, state), maxDepth) for state in gameState.getLegalActions(agent))


# def abMiniMax(evalFunc: classmethod, agent: int, depth: int, gameState: GameState, maxDepth: int, alpha: float = float("-inf"), beta: float = float("inf")) -> float:
#     if gameState.isLose() or gameState.isWin() or depth == maxDepth:
#         return evalFunc(gameState)
#     if agent == 0:
#         pacScore = float("-inf")
#         for move in gameState.getLegalActions(agent):
#             pacScore = max(pacScore, abMiniMax(
#                 evalFunc, 1, depth, gameState.generateSuccessor(agent, move), maxDepth, alpha, beta))
#             if pacScore >= beta:
#                 return pacScore
#             alpha = max(alpha, pacScore)
#         return pacScore
#     else:
#         nextAgent = agent + 1
#         if gameState.getNumAgents() == nextAgent:
#             nextAgent = 0
#         if nextAgent == 0:
#             depth += 1
#         ghostScore = float("inf")
#         for move in gameState.getLegalActions(agent):
#             ghostScore = min(ghostScore, abMiniMax(evalFunc, nextAgent, depth,
#                                                    gameState.generateSuccessor(agent, move), maxDepth, alpha, beta))
#             if ghostScore <= alpha:
#                 return ghostScore
#             beta = min(beta, ghostScore)
#         return ghostScore



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        MultiAgentSearchAgent.__init__(self, evalFn, depth)
        self.re_init()

    def re_init(self):
        self.alpha = float("-inf")
        self.beta = float("inf")
        # self.currentDepth = 0

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        _max = float("-inf")
        action = None
        for move in gameState.getLegalActions(0):
            self.re_init()
            util = self.abMiniMax(gameState, 1, 0)
            if util > _max or _max == float("-inf"):
                _max = util
                action = move

        return action

    def abMiniMax(self, state: GameState, agent, depth):
        if state.isLose() or state.isWin() or depth == self.depth:
            return self.evaluationFunction(state)
        if agent == 0:
            res = self.maxValue(state, agent, depth)
            return res
        else:
            agent += 1
            if state.getNumAgents() == agent:
                agent = 0
                depth += 1
            return self.minValue(state, agent, depth)


    def maxValue(self, state: GameState, agent, depth):
        v = float("-inf") 
        depth += 1
        listOfMoves = state.getLegalActions(agent)
        for move in listOfMoves:
            v = max(v, self.abMiniMax(state.generateSuccessor(agent, move), agent, depth))
            if v >= self.beta:
                return v
            self.alpha = max(self.alpha, v)
        return v


    def minValue(self, state: GameState, agent, depth):
        v = float("inf")
        listOfMoves = state.getLegalActions(agent)
        for move in listOfMoves:
            v = min(v, self.abMiniMax(state.generateSuccessor(agent, move), agent))
            if v <= self.alpha:
                return v
            self.beta = min(self.beta, v)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
