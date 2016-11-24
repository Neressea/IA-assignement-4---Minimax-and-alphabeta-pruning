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
import random, util, sys

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
        return successorGameState.getScore()

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
        """

        #The initial depth is 0
        return self.maxValue(gameState)

    """
     * Compute the best possible move for PACMAN. To do so, computes for each possible move the utility of the move.
     * This utility is based on the moves the ghosts can do after this decision, which implies recursive calls.
     * 
     * And to compute the best adversary move, we will loop to know what is the best move PACMAN can do in this situation.
     * We loop until the game is over, or we reach the depth limit.
     *
     * gameState: the initial current state of the game
    """
    def maxValue(self, gameState):

    	#At the beginning, we are at a depth of 0
        best = self.bestValueAndMove(gameState, 0)

        return best['move'] #if we went back to the first call, the recursion is over and we return the best move to make 

    """
     * The recursion for the best move computation.
     *
     * gameState: the current state of the game
     * depth: the current depth in the actions' tree
    """
    def recurseMaxValue(self, gameState, depth):

    	#The game has two possible ends: either we win or we loose. If we encouter one of these state before reaching the depth limit, we go back with this value.
    	if gameState.isWin() or gameState.isLose():
    		return self.evaluationFunction(gameState) #In this case, we compute our final score to see if it is the best possible

        #If we reached the maximum depth, we just return the evaluation of the state we have
    	if depth == self.depth:
            return self.evaluationFunction(gameState)

        #If we aren't at the end, we call the function that returns the dictionary (bestMove, bestUtility)
        best = self.bestValueAndMove(gameState, depth)

        return best['utility'] #During the recursion, we just return the utility value. The best move is only returned by the first call, as a final decision

    """
     * Execute each possible move for PACMAN and recurse on the ghosts'possible actions.
     *
     * gameState: the current state of the game
     * depth: the current depth in the actions' tree
    """
    def bestValueAndMove(self, gameState, depth):

    	v = -sys.maxsize #We set the best utility at the worst possible value, so any other value is better than that
        moves = gameState.getLegalActions(self.index) #We get all the possible moves

    	#For each move, we see which one gives the max utility
        for move in moves:

        	state = gameState.generateSuccessor(self.index, move) #We compute the state after this move

        	#And we compute what the ennemies can do to perturb us, what is the minimal utility they can give us
        	#This method will recurse on itself to compute all the possible enemy combination to find the worst one
        	vtemp = self.minValue(state, depth, 1) 

        	#If this new utility is better than the previous one, we keep it
        	if vtemp > v:
        		v = vtemp
        		best_move = move

        return {'move': best_move, 'utility': v}

    """
     * Compute the best combination of moves for ghosts. 
     * To do so, it will compute every possible move for the current ghost, and will compute all the possible moves for the following ghosts.
     * This way, we will have the best combination for all ghosts. We recurse until the game is over, or we reach the depth limit.
     *
     * This function is recursive on itself, because there are more than 1 ennemy, so we have to compute the situation for all the possible
     * combination of ghosts moves.
     *
     * gameState: the current state of the game
     * depth: the current recursion depth
     * agent: the index of the ghost
    """
    def minValue(self, gameState, depth, agent):

    	#The game has two possible ends: either we win or we loose. If we encouter this before reaching the depth limit, we go back with this value.
    	if gameState.isWin() or gameState.isLose():
    		return self.evaluationFunction(gameState) #In this case, we compute our final score to see if it is the best possible

        #If we reached the maximum depth, we just return the evaluation of the state we have
    	if depth == self.depth:
            return self.evaluationFunction(gameState)

    	#We have to check that we aren't on the last ghost
    	max_ghosts = gameState.getNumAgents()

        v = sys.maxsize #The min utility for this ghost for the moment
        vtemp=sys.maxsize
        moves = gameState.getLegalActions(agent) #We get all the possible moves for this ghost

        #For each move, we compare to see if it gives the min utility
        for move in moves:
            state = gameState.generateSuccessor(agent, move) #We compute the state when we do this move

            if agent == max_ghosts - 1:
            	#If this ghost is the last one, after that we will compute what Pacman can do
                vtemp = self.recurseMaxValue(state, depth+1)
            else:
            	#If not, we will compute the possible moves for the next ghosts. The depth doesn't change, because we are exploring siblings.
                vtemp = self.minValue(state, depth, agent+1)

             #The aim of ghosts is to minimize the utility of the user.
            if vtemp < v:
                v = vtemp

        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.maxValue(gameState)

	"""
     * Compute the best possible move for PACMAN. To do so, computes for each possible move the utility of the move.
     * This utility is based on the moves the ghosts can do.
     * And to compute the best adversary move, we will loop to know what is the best move PACMAN can do in this situation.
     * We loop until the game is over, or we reach the depth limit.
     *
     * gameState: the current state of the game
     * depth: the current recursion depth
    """
    def maxValue(self, gameState):

        best = self.bestValueAndMove(gameState, -sys.maxsize, sys.maxsize, 0)

        return best['move'] #if we went back to the first call, the recursion is over and we return the best move to make 

    """
     * The recursion for the best move computation.
     *
     * gameState: the current state of the game
     * depth: the current depth in the actions' tree
    """
    def recurseMaxValue(self, gameState, alpha, beta, depth):

    	#The game has two possible ends: either we win or we loose. If we encouter this before reaching the depth limit, we go back with this value.
    	if gameState.isWin() or gameState.isLose():
    		return self.evaluationFunction(gameState) #In this case, we compute our final score to see if it is the best possible

        #If we reached the maximum depth, we just return the evaluation of the state we have
    	if depth == self.depth:
            return self.evaluationFunction(gameState)

        #If we aren't at the end, we call the function that returns the dictionary (bestMove, bestUtility)
        best = self.bestValueAndMove(gameState, alpha, beta, depth)

        return best['utility'] #During the recursion, we just return the utility value. The best move is only returned by the first call, as a final decision

    """
     * Execute each possible move for PACMAN and recurse on the ghosts'possible actions.
     *
     * gameState: the current state of the game
     * alpha: the current alpha value
     * beta: the current beta value
     * depth: the current depth in the actions' tree
    """
    def bestValueAndMove(self, gameState, alpha, beta, depth):

    	v = -sys.maxsize #We set the best utility at the worst possible value, so any other value is better than that
        moves = gameState.getLegalActions(self.index) #We get all the possible moves for PACMAN

    	#For each move, we see which one gives the max utility
        for move in moves:

        	state = gameState.generateSuccessor(self.index, move) #We compute the state after this move

        	#And we compute what the ennemies can do to perturb us, what is the minimal utility they can give us
        	#This method will recurse on itself to compute all the possible enemy combination to find the worst one
        	vtemp = self.minValue(state, alpha, beta, depth, 1)

        	#If this new utility is better than the previous one, we keep it
        	if vtemp > v:
        		v = vtemp
        		best_move = move

        	#We update the inf bound if this value is higher. This way, narrow the segment we explore.
        	alpha = max(alpha, v)

        	#If we went over the sup bound, we stopeverything and return this value
        	if v > beta:
    			return {'move': best_move, 'utility': v}

        return {'move': best_move, 'utility': v}

    """
     * Compute the best combination of moves for ghosts. 
     * To do so, it will compute every possible move for the current ghost, and will compute all the possible moves for the following ghosts.
     * This way, we will have the best combination for all ghosts. We recurse until the game is over, or we reach the depth limit.
     *
     * This function is recursive on itself, because there are more than 1 ennemy, so we have to compute the situation for all the possible
     * combination of ghosts moves.
     *
     * gameState: the current state of the game
     * alpha: the current alpha value
     * beta: the current beta value
     * depth: the current recursion depth
     * agent: the index of the ghost
    """
    def minValue(self, gameState, alpha, beta, depth, i):
    	
    	#The game has two possible ends : either we win or we loose. If we encouter this before reaching the depth limit, we go back with this value.
    	if gameState.isWin() or gameState.isLose():
    		return self.evaluationFunction(gameState) #In this case, we compute our final score to see if it is the best possible

        #If we reached the maximum depth, we just return the evaluation of the state we have
    	if depth == self.depth:
            return self.evaluationFunction(gameState)

        #We have to check that we aren't on the last ghost
    	max_ghosts = gameState.getNumAgents()

        v = sys.maxsize #The min utility for this ghost for the moment
        vtemp=v
        moves = gameState.getLegalActions(i) #We get all the possible moves for this ghost

    	for move in moves:
            state = gameState.generateSuccessor(i, move) #We compute the state when we do this move

            if i == max_ghosts - 1:
            	#If this ghost is the last one, after that we will compute what Pacman can do
                vtemp = self.recurseMaxValue(state, alpha, beta, depth+1)
            else:
            	#If not, we will compute the possible moves for the next ghosts 
                vtemp = self.minValue(state, alpha, beta, depth, i+1)

            if vtemp < v:
            	v = vtemp

            #We update the sup bound if this value is lower. This way, narrow the segment we explore.
            beta = min(beta, v)

            #If we aren't in the bounds, we stop everything and return this value
            if v < alpha:
                return v

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

