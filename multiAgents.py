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
        
        (min_to_food, max_to_food) = util.getMinMaxManhattanToFood(newPos, newFood)
        
        close_ghost_data = util.getMinManhattanGhost(newPos, newGhostPos)
        stay_alive = 0
        food_dist_term = 0
        if newScaredTimes[close_ghost_data[0]] == 0 and close_ghost_data[1] < 15 and close_ghost_data[1] > 0: # try not to stay in the corner
          stay_alive = 1.0/close_ghost_data[1]
        
        food_term = currentGameState.getNumFood() - successorGameState.getNumFood() # did I just eat food?
        if min_to_food != 0 and max_to_food != 0:
          food_dist_term = 0.1/min_to_food

        get_points = 0
        score_delta = successorGameState.getScore() - currentGameState.getScore()

        if score_delta >= 100:
          # print "eval: 1"
          return 1 # you should eat a ghost if won't die immediately (usually), should always go to win state
        if newScaredTimes[close_ghost_data[0]] > 0 and newScaredTimes[close_ghost_data[0]] >= close_ghost_data[1]: # if we can eat the closest ghost, go all-out for it
          get_points = 1.0/close_ghost_data[1] # go get ghosts
        elif newScaredTimes[close_ghost_data[0]] == 0 and close_ghost_data[1] < 3:
          get_points = 0 # if about to die get out of there
        else:
          get_points = food_dist_term + 0.25*food_term # default is try to get more food in short & very short term
        
        # print "eval: " + str(get_points - 0.5*stay_alive)
        # print "food num: " + str(food_term) # food increases in val as number decreases
        # print "food dist: " + str(food_dist_term)
        return get_points - 0.5*stay_alive

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
    def maxState(self, gameState, depth):
      if gameState.isLose() or gameState.isWin() or depth >= self.depth:
        return (None, gameState)
      actions = gameState.getLegalActions()
      max_state = None
      max_action = None
      for action in actions:
        local_max_successor = self.minState(1, gameState.generateSuccessor(0, action), depth)[1]
        if max_state == None or self.evaluationFunction(local_max_successor) > self.evaluationFunction(max_state):
          max_state = local_max_successor
          max_action = action
      return (max_action, max_state)
    
    def minState(self, agentIndex, gameState, depth):
      if gameState.isLose() or gameState.isWin():
        return (None, gameState)
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
      return (min_action, min_state)
    
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
        """
        best_move = self.maxState(gameState, 0)
        return best_move[0]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxState(self, gameState, depth, a, b):
      if gameState.isLose() or gameState.isWin() or depth >= self.depth:
        return (None, gameState)
      actions = gameState.getLegalActions()
      max_state = None
      max_action = None
      for action in actions:
        local_max_successor = self.minState(1, gameState.generateSuccessor(0, action), depth, a, b)[1]
        if max_state == None or self.evaluationFunction(local_max_successor) > self.evaluationFunction(max_state):
          max_state = local_max_successor
          max_action = action
        if self.evaluationFunction(max_state) > b:
          return (max_action, max_state)
        a = max(a, self.evaluationFunction(max_state))
      return (max_action, max_state)
    
    def minState(self, agentIndex, gameState, depth, a, b):
      if gameState.isLose() or gameState.isWin():
        return (None, gameState)
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
          return (min_action, min_state)
        b = min(b, self.evaluationFunction(min_state))
      return (min_action, min_state)
    
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
        """
        a = -sys.maxsize
        b = sys.maxsize
        best_move = self.maxState(gameState, 0, a, b)
        return best_move[0]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def maxAction(self, gameState, depth):
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
      return max_action
    
    def expState(self, agentIndex, gameState, depth):
      # be sure to return states of leaves so max can see them
      if gameState.isLose() or gameState.isWin():
        return self.evaluationFunction(gameState)
      actions = gameState.getLegalActions(agentIndex)
      expectation = 0.0
      for action in actions:
        if agentIndex == gameState.getNumAgents() - 1:
          successor = gameState.generateSuccessor(agentIndex, action) # this is a player
          ma = self.maxAction(successor, depth + 1) # the max action from this player gameState
          if ma == None: # reached leaf
            expectation += self.evaluationFunction(successor)
          else: # get expectation of subtree which max has chosen
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
        best_move = self.maxAction(gameState, 0)
        return best_move
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION (predictions before testing):
        MINIMIZE food closeness - less important if many ghosts
        - add reciprocal
        - priority 3
        MINIMIZE min ghost distance if a) scared and b) manhattan <= time scared - more important if many ghosts
        - add reciprocal
        - priority 4
        MAXIMIZE min ghost distance if not scared - more important if many ghosts - this is how you stay alive!
        - subtract reciprocal
        - priority 1
        MAXIMIZE average ghost distance if not scared - more important if many ghosts
        - subtract reciprocal
        - priority 5
        MAXIMIZE total scared time - more important when average ghost distance is low
        - subtract reciprocal
        - priority 6
        MAXIMIZE score - account for 1) ghost eating and 2) immediate food eating
        - subtract reciprocal 
        - priority 2
        At a winning state, return max value of combo of some stats + give bonus for score metric

        WEAKNESS: does not know when to eat ghosts when right next to them!
        - attempted a score coefficient booster, but we require knowledge of previuous states to really solve
        this problem in the way which I've thought about it
    """
    magd_coeff = 50      # min active dist 50
    score_coeff = 0.5     # score 1
    max_food_coeff = 5  # food closeness 5
    msgd_coeff = 0      # min scared dist 1
    tagd_coeff = 0      # total active dist (average) 1
    tsgt_coeff = 0      # total scared time 1
    tsgd_coeff = 0      # total scared distance (average) 1

    score = currentGameState.getScore()
    if currentGameState.isWin() and score != 0:
      return magd_coeff + max_food_coeff + msgd_coeff + tagd_coeff + tsgt_coeff + tsgd_coeff + score_coeff*1.0/score # 21.0 + 1.0/score

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    ghostPos = currentGameState.getGhostPositions()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    (min_food_dist, max_food_dist) = util.getMinMaxManhattanToFood(pos, food)

    min_active_ghost_dist = sys.maxsize
    total_active_ghost_dist = 0

    min_scared_ghost_dist = sys.maxsize
    total_scared_ghost_dist = 0
    total_scared_ghost_time = 0

    ghost_count = len(ghostPos)
    scared_ghost_count = 0
    
    for i in range(ghost_count):
        d = util.manhattanDistance(pos, ghostPos[i])
        if scaredTimes[i] > 0:
          total_scared_ghost_time += scaredTimes[i]
          total_scared_ghost_dist += d
          if min_scared_ghost_dist > d:
            min_scared_ghost_dist = d
          scared_ghost_count += 1
        else:
          total_active_ghost_dist += d
          if min_active_ghost_dist > d:
            min_active_ghost_dist = d
    
    if ghost_count - scared_ghost_count != 0:
      total_active_ghost_dist /= float(ghost_count - scared_ghost_count)
    if scared_ghost_count != 0:
      total_scared_ghost_dist /= float(scared_ghost_count)
    if min_scared_ghost_dist != 0:
      min_scared_ghost_dist = 1.0/min_scared_ghost_dist
    if min_active_ghost_dist != 0:
      min_active_ghost_dist = 1.0/min_active_ghost_dist
    if total_active_ghost_dist != 0:
      total_active_ghost_dist = 1.0/total_active_ghost_dist
    if total_scared_ghost_dist != 0:
      total_scared_ghost_dist = 1.0/total_scared_ghost_dist
    if total_scared_ghost_time != 0:
      total_scared_ghost_time = 1.0/total_scared_ghost_time
    if score != 0:
      score = 1.0/score
    
    return (max_food_coeff * 1.0/max_food_dist + 
            msgd_coeff * min_scared_ghost_dist - 
            magd_coeff * min_active_ghost_dist -
            tagd_coeff * total_active_ghost_dist -
            tsgd_coeff * total_scared_ghost_dist - 
            tsgt_coeff * total_scared_ghost_time - 
            score_coeff * score)

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
