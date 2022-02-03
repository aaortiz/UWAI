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

import math

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        import math
        pos_inf = math.inf
        neg_inf = -math.inf

        # Check terminal states: Goal state and fail states (eaten by ghost)
        if successorGameState.isWin():
            return pos_inf
        elif successorGameState.isLose():
            return neg_inf

        currPacPosition = currentGameState.getPacmanPosition()
        currGhostPositions = currentGameState.getGhostPositions()
        currCapsulePositions = currentGameState.getCapsules()
        currFoodPositions = currentGameState.getFood().asList()

        newFoodPositions = newFood.asList()
        newGhostPositions = successorGameState.getGhostPositions()

        score = -(len(newFoodPositions) + len(currCapsulePositions))

        # Food
        # Less food is good
        if len(newFoodPositions) < len(currFoodPositions):
            score += 100

        # Same amount of food is bad
        if len(newFoodPositions) == len(currFoodPositions):
            score -= 50

        # Closer to food is good
        dists_to_new_food = [util.manhattanDistance(newPos, fp)
                             for fp in newFoodPositions]

        dists_to_curr_food = [util.manhattanDistance(newPos, fp)
                              for fp in currFoodPositions]

        if dists_to_new_food:
            score += (5/max(min(dists_to_new_food), 1))

        # incentivize food at all distances
        if dists_to_curr_food:
            closest_food_dist = min(dists_to_curr_food)
            avg_food_dist = sum(dists_to_curr_food) / len(dists_to_curr_food)
            furthest_food_dist = max(dists_to_curr_food)
            score += 10 / max(closest_food_dist, 1)
            score += 5 / max(avg_food_dist, 1)
            score += 2.5 / max(furthest_food_dist, 1)

        # Capsules
        # Eating a capsule is very good
        if newPos in currCapsulePositions:
            score += 100

        # closer to capsule is good
        capsuleDists = [util.manhattanDistance(newPos, cap)
                        for cap in currCapsulePositions]
        if capsuleDists:
            score += 5 / max(min(capsuleDists), 1)

        # Ghosts
        # Scared Ghosts is very good
        score += sum(newScaredTimes)

        # Further from ghosts is good
        dists_to_new_ghosts = [util.manhattanDistance(newPos, gp)
                               for gp in newGhostPositions]

        dists_to_curr_ghosts = [util.manhattanDistance(currPacPosition, gp)
                               for gp in currGhostPositions]

        if sum(dists_to_new_ghosts) > sum(dists_to_curr_ghosts):
            score += (5/sum(dists_to_new_ghosts))

        score -= 50 * min(dists_to_curr_ghosts)

        # # Bias to explore
        # if currPacPosition != newPos:
        #     score += 25
        # elif currPacPosition == newPos:
        #     score -= 25

        return score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        minimax_action, __ = self.minimaxDispatch(gameState)
        return minimax_action

    def minimaxDispatch(self, gameState):

        currdepth = 0
        minimax_action, minimax_score = self.maxAgentActionValue(gameState,
                                                                 currdepth)

        return minimax_action, minimax_score

    def maxAgentActionValue(self, gameState, currdepth):

        if gameState.isWin() or gameState.isLose() or self.depth == currdepth:
            action = "None"
            score = self.evaluationFunction(gameState)
            return action, score

        max_score, actions = -math.inf, gameState.getLegalActions(0)

        for action in actions:
            next_state = gameState.generateSuccessor(0, action)
            min_action, min_score = self.minAgentActionValue(next_state,
                                                             currdepth,
                                                             1) # 1 represents first ghost

            if max_score < min_score:
                max_score = min_score
                max_action = action

        return max_action, max_score

    def minAgentActionValue(self, gameState, currdepth, agent_idx):

        num_agents = gameState.getNumAgents()

        if agent_idx >= num_agents:
            next_depth = currdepth + 1
            return self.maxAgentActionValue(gameState, next_depth)

        if gameState.isWin() or gameState.isLose() or self.depth == currdepth:
            action = "None"
            score = self.evaluationFunction(gameState)
            return action, score

        min_score, min_actions = math.inf, gameState.getLegalActions(agent_idx)

        for a in min_actions:
            next_state = gameState.generateSuccessor(agent_idx, a)
            next_agent_idx = agent_idx + 1
            curr_min_a, curr_min_s = self.minAgentActionValue(next_state,
                                                              currdepth,
                                                              next_agent_idx)

            if curr_min_s < min_score:
                min_score = curr_min_s
                min_action = a

        return min_action, min_score

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
