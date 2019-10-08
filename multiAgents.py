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
import sys

from game import Agent

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "*** YOUR CODE HERE IF NECESSARY ***"
        

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
        newGhostPos = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # was good for food heuristic before (but with maze dist)
        # food_distance_variable = 1.0/util.getMinManhattanToFood(newPos, newFood) - 1.0/util.getMaxManhattanToFood(newPos, newFood)
        (min_to_food, max_to_food) = util.getMinMaxManhattanToFood(newPos, newFood)
        # get which ghost is closest. if it's scared go towards it. else away
        # maximize dist to closest ghost when not scared
        close_ghost_data = util.getMinManhattanGhost(newPos, newGhostPos)
        if newScaredTimes[close_ghost_data[0]] > 0:
          ghost_term = 1/close_ghost_data[1] # go get ghosts
        else:
          if close_ghost_data[1] > 15: # try not to stay in the corner
            ghost_term = 0
          else:
            ghost_term = close_ghost_data[1]/(2*(newFood.width + newFood.height)) # stay away from ghosts
        # food term is not high-enough influence
        food_term = currentGameState.getNumFood() - successorGameState.getNumFood() # current
        food_dist_term = 1.0/(min_to_food + max_to_food) # future

        print "ghost: " + str(ghost_term)
        print "food num: " + str(food_term) # food increases in val as number decreases
        print "food dist: " + str(food_dist_term)
        return food_term + ghost_term + food_dist_term

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
    def maxState(self, gameState, depth): # get max state or max action?
      if gameState.isLose() or gameState.isWin() or depth >= self.depth:
        return (None, gameState) # problem with no action here?
      actions = gameState.getLegalActions()
      max_state = None
      max_action = None
      for action in actions:
        local_max_successor = self.minState(1, gameState.generateSuccessor(0, action), depth)[1] # is 1 here general enough?
        if max_state == None or self.evaluationFunction(local_max_successor) > self.evaluationFunction(max_state):
          max_state = local_max_successor
          max_action = action
      # print "Max of " + str(self.evaluationFunction(max_state))
      return (max_action, max_state)
    
    def minState(self, agentIndex, gameState, depth):
      if gameState.isLose() or gameState.isWin():
        return (None, gameState) # problem with no action here?
      actions = gameState.getLegalActions(agentIndex)
      min_state = None
      min_action = None
      for action in actions:
        if agentIndex == gameState.getNumAgents() - 1:
          local_min_successor = self.maxState(gameState.generateSuccessor(agentIndex, action), depth + 1)[1]
        else:
          local_min_successor = self.minState(agentIndex + 1, gameState.generateSuccessor(agentIndex, action), depth)[1]
        if min_state == None or self.evaluationFunction(local_min_successor) < self.evaluationFunction(min_state):
          min_state = local_min_successor
      # print "Min of " + str(self.evaluationFunction(min_state))
      return (min_action, min_state)
    
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
        """
        best_move = self.maxState(gameState, 0)
        # print "Choice is: " + str(self.evaluationFunction(best_move[0]))
        return best_move[0]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxState(self, gameState, depth, a, b): # get max state or max action?
      if gameState.isLose() or gameState.isWin() or depth >= self.depth:
        return (None, gameState) # problem with no action here?
      actions = gameState.getLegalActions()
      max_state = None
      max_action = None
      for action in actions:
        local_max_successor = self.minState(1, gameState.generateSuccessor(0, action), depth, a, b)[1] # is 1 here general enough?
        if max_state == None or self.evaluationFunction(local_max_successor) > self.evaluationFunction(max_state):
          max_state = local_max_successor
          max_action = action
        if self.evaluationFunction(max_state) > b:
          # print "Returning by beta test: " + str(self.evaluationFunction(max_state))
          return (max_action, max_state)
        a = max(a, self.evaluationFunction(max_state))
      # print "Max of " + str(self.evaluationFunction(max_state))
      return (max_action, max_state)
    
    def minState(self, agentIndex, gameState, depth, a, b):
      if gameState.isLose() or gameState.isWin():
        return (None, gameState) # problem with no action here?
      actions = gameState.getLegalActions(agentIndex)
      min_state = None
      min_action = None
      for action in actions:
        if agentIndex == gameState.getNumAgents() - 1:
          local_min_successor = self.maxState(gameState.generateSuccessor(agentIndex, action), depth + 1, a, b)[1]
        else:
          local_min_successor = self.minState(agentIndex + 1, gameState.generateSuccessor(agentIndex, action), depth, a, b)[1]
        if min_state == None or self.evaluationFunction(local_min_successor) < self.evaluationFunction(min_state):
          min_state = local_min_successor
        if self.evaluationFunction(min_state) < a:
          # print "Returning by alpha test: " + str(self.evaluationFunction(min_state))
          return (min_action, min_state)
        b = min(b, self.evaluationFunction(min_state))
      # print "Min of " + str(self.evaluationFunction(min_state))
      return (min_action, min_state)
    
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
        """
        a = -sys.maxsize
        b = sys.maxsize
        # print "A: " + str(a) + " B: " + str(b)
        best_move = self.maxState(gameState, 0, a, b)
        # print "Choice is: " + str(self.evaluationFunction(best_move[0]))
        return best_move[0]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def maxState(self, gameState, depth): # get max state or max action?
      if gameState.isLose() or gameState.isWin() or depth >= self.depth:
        return None
      actions = gameState.getLegalActions()
      max_eval = None
      max_action = None
      for action in actions:
        expectation = self.expState(1, gameState.generateSuccessor(0, action), depth)
        if max_action == None or expectation > max_eval:
          max_eval = expectation
          max_action = action
      print "Max move: " + str(max_action)
      return max_action
      # get it to be a leaf by returning leaf through expState?
      # can't be sure of which to choose
    
    def expState(self, agentIndex, gameState, depth):
      # be sure to return states of leaves so max can see them
      if gameState.isLose() or gameState.isWin():
        return self.evaluationFunction(gameState)
      actions = gameState.getLegalActions(agentIndex)
      expectation = 0.0
      for action in actions:
        if agentIndex == gameState.getNumAgents() - 1:
          successor = gameState.generateSuccessor(agentIndex, action) # this is a player
          ma = self.maxState(successor, depth + 1) # the max action from this player gameState
          if ma == None: # reached leaf
            expectation += self.evaluationFunction(successor)
          else: # get expectation of subtree which max has chosen -- REDUNDANT?
            successor_2_along_max_path = successor.generateSuccessor(0, ma)
            expectation += self.expState(1, successor_2_along_max_path, depth + 1) # get expectation of the chosen max action
        else:
          expectation += self.expState(agentIndex + 1, gameState.generateSuccessor(agentIndex, action), depth)
      expectation /= float(len(actions))
      return expectation
    
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
        """
        a = -sys.maxsize
        b = sys.maxsize
        # print "A: " + str(a) + " B: " + str(b)
        best_move = self.maxState(gameState, 0)
        # print "Choice is: " + str(self.evaluationFunction(best_move[0]))
        return best_move[0]
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
