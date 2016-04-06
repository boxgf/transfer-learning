# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
import numpy as np
import scipy as sp
import types
from scipy import linalg
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from pacman import GameState
from collections import defaultdict
import random,util,math
import time
lookup = {}
pellets = []
for i in range(16):
	lookup[i] = bin(i).count('1')
def stop():
	print "Stop Called"
	time.sleep(10000)
def possibleTransitions(action,T = GameState.transitionDynamics):
	if action == 'Stop': return {action:1}
	errorAdditions = [0,1,2,3,4]
	possibility = {}
	actiontoNumberMap = {'North':0,'West':1,'South':2,'East':3}
	numbertoActionMap = {0:'North',1:'West',2:'South',3:'East'}
	for i in range(len(T)):
		if T[i]>0:
			
			possibility[numbertoActionMap[(actiontoNumberMap[action]+errorAdditions[i])%4]] = T[i]
	return possibility
		  
def possibleMeasurements(state,b):
	trueObservation = state.getObservations([1,0])
	possibility ={}
	correctProb = state.sensorDynamics[0]
	(x,y) = state.getPacmanPosition()
	flag = pellets[x][y]
	reward =  -1 * (not flag) + 10* flag
	for i in range(16):
		mismatch = lookup[trueObservation^i]
		P = ((1-correctProb)**mismatch * correctProb**(4-mismatch))*b
		#if P>0: possibility[(i,reward)] = P
		possibility[(i,reward)] = P
	return possibility
	
Actions = ['North','West','East','South']
Obs = []
def normalize(bel):
	alpha = sum(bel.values())
	if alpha == 0: return bel
	for i in bel:
		bel[i] = float(bel[i])/alpha
		
	return bel
	
D ={}

toExpand = [[(),()]]
universalBelief = {}
universalBelief[((),)] ={}
sampleState = 0

def posGenerateSuccessor(pos,action):
	global sampleState
	actiontoOffsetMap = {'North':(0,1),'West':(-1,0),'South':(0,-1),'East':(1,0)}
	offset = actiontoOffsetMap[action]
	ret = pos[0]+offset[0],pos[1]+offset[1]
	if sampleState.hasWall(ret[0],ret[1]): return pos
	else: return ret
	
def beliefSuccessor(belief,sampleState, history = ()):


	
	for action in Actions:
		newBelief = {}
		possibleActionOutcomes = possibleTransitions(action)

		for i in possibleActionOutcomes:
			for state in belief:
				b = belief[state]
				childProb = b * possibleActionOutcomes[i]
				sampleState.setPacmanPosition(state)
				child = GameState(sampleState.generateSuccessor(0,i,[1,0,0,0,0]))
				if child not in newBelief: newBelief[child.getPacmanPosition()] = childProb
				else: newBelief[child.getPacmanPosition()] += childProb
				
		possibleSensorOutcomes = {}
		b_dash = {}

		'''
		for i in allChildren:
			if i[0] not in b_dash: b_dash[i] = i[1]
			else: b_dash[i] += i[1]
		'''
		
		listOfBelief = {}
		for state in newBelief:
			b = newBelief[state]
			sampleState.setPacmanPosition(state)
			o = possibleMeasurements(sampleState,b)
			index = 0
			for i in o:
				if i not in listOfBelief: listOfBelief[i]={}
				listOfBelief[i][state] = o[i]
				index += 1
				if i not in possibleSensorOutcomes: possibleSensorOutcomes[i]= o[i]
				else: possibleSensorOutcomes[i]+= o[i]
		
		global Obs
		if history not in D: D[history]={}
		for i in Obs:	
			
			if i in possibleSensorOutcomes: D[history][((action,i),)] = possibleSensorOutcomes[i]
			else: D[history][((action,i),)] = 0
			
			
		for obs in listOfBelief:
			listOfBelief[obs] = normalize(listOfBelief[obs])
			if history not in universalBelief: universalBelief[history] = {}
			universalBelief[history][(action,obs)] = listOfBelief[obs]
		
	
	
	#for actionobs in universalBelief[history]:
	#	toExpand.append((actionobs,history))
	'''
	if recursionFlag:
		if len(history)==2:
			print len(D), history
			stop()
		count = 0
		
		for actionobs in universalBelief[history]:
				count +=1
				newHistory = list(history)
				newHistory.append(actionobs)
				newBelief = universalBelief[history][actionobs]
				beliefSuccessor(universalBelief[history][actionobs],sampleState,tuple(newHistory),count == len(universalBelief[history]))
	'''
def independentRow(matrix):
    """
	independentRows=[]
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[0]):
			if i != j:
				inner_product = np.inner(
					matrix[:,i],
					matrix[:,j]
				)
				norm_i = np.linalg.norm(matrix[:,i])
				norm_j = np.linalg.norm(matrix[:,j])

				print 'I: ', matrix[:,i]
				print 'J: ', matrix[:,j]
				print 'Prod: ', inner_product
				print 'Norm i: ', norm_i
				print 'Norm j: ', norm_j
				if np.abs(inner_product - norm_j * norm_i) < 1E-5:
					print 'Dependent'
				else:
					print 'Independent'
	"""
    tol=1.e-10
    R=sp.linalg.qr(matrix)[1]
    independentRows=np.where(abs(np.sum(R,1))>tol)
    return independentRows					
	
def construct(D_mat,coreHistory,coreTest,prevRankD):
	rowOffset=D_mat.shape[0]
	colOffset=D_mat.shape[1]

	terminate=False
	if rowOffset == 0: D_mat = np.zeros([129,128])
	else: 
		
		new_cols=np.zeros([D_mat.shape[0],D_mat.shape[1]*len(Actions)*len(Obs)])
		D_mat=np.append(D_mat,new_rows,axis=1)
		new_rows=np.zeros([D_mat.shape[0]*len(Actions)*len(Obs),D_mat.shape[1]])
		D_mat=np.append(D_mat,new_cols,axis=0)
	oneStepHistory=[]
	oneStepTest=[]
	oneStepHistory+=coreHistory
	oneStepTest+=coreTest
	#print Actions,Obs
	#stop()
	for a in Actions:
		for o in Obs:
	#		print "a",(a,o) 	
		#for o in xrange(len(Obs)):
		#	for r in [-1,10]:
			for ch in coreHistory:
				if ch==():
						#oneStepHistory.append((a,(o,r)))
						oneStepHistory.append(((a,o),))
				else:	
						oneStepHistory.append(ch+((a,o),))
			if coreTest==[]:
					oneStepTest.append(((a,o),))
					continue	
			for ct in coreTest: 
					oneStepTest.append(ct+((a,o),)) 	
					 
	#stop()
	row,col = 0,colOffset

	#stop()
	for h in oneStepHistory:
		for t in oneStepTest[len(coreTest):]:
			#if not(t in coreTest and h in coreHistory):
			
			D_mat[row][col] = generalFormula(t[0],t[1:],h)
			col+=1    
		row+=1
		col=colOffset
		
	row,col = rowOffset,0
	for h in oneStepHistory[len(coreHistory):]:
		for t in coreTest:
			D_mat[row][col]=D[h][t]
			col+=1
		row+=1
		col=0	
		
	
	indRows=independentRow(D_mat)[0]    
	indCols=independentRow(D_mat.T)[0]
	curRankD=len(indRows)
	if curRankD==prevRankD:
		 terminate=True
	newCoreHistory=[oneStepHistory.pop(index) for index in reversed(indRows)]
	newCoreTest=[oneStepTest.pop(index) for index in reversed(indCols)] 
#	for h in oneStepHistory:
#		del D[h]
#	for t in oneStepTest:
#		for h in newCoreHistory:      
#			del D[h][t]
	return D_mat,newCoreHistory,newCoreTest,terminate,curRankD
	
	
	    		
def modelLinearPSR(state):
	currentDmat = np.zeros([0,0])
	
	global Obs
	global pellets
	global sampleState 
	sampleState = state
	pellets = state.getFood()
	universalBelief[((),)][()]= {state.getPacmanPosition():1}
	rank = -1
	Obs = state.getObservationset()
	coreHistory=[()]
	coreTest=[]
	for a in Actions:
		for o in Obs:
			toExpand.append(((a,o),))
	#print toExpand
	#stop()
	converged = 0
	N = 0              # N is the dimension of D found till now
	while not converged:
		
		while toExpand:
			
			print len(toExpand)
			hao = toExpand.pop(0)
			newHistory = tuple(hao)

			history = newHistory[:-1]
			
			actionobs = newHistory[-1]
			
		
			if newHistory==((),()):newHistory=()
			
			if (history not in universalBelief) or ((history in universalBelief) and (actionobs not in universalBelief[history])):
				if history not in D:D[history]={}
				D[history][(actionobs,)]=0
			
				
				D[newHistory]=defaultdict(lambda: 0,{})
				
				continue
			
			beliefSuccessor(universalBelief[history][actionobs],state,newHistory)
		
		currentDmat,coreHistory,coreTest,converged,rank = construct(currentDmat,coreHistory,coreTest,rank)
		print N,rank,currentDmat
		stop()
		if N==1:
			stop()
	
		N+=1
		historyOffset = coreHistory[:]
		for i in xrange(2*N):
			print "i",i
			temp = []
			while historyOffset:
				chao = historyOffset.pop(0)
			#	print "chao",chao,len(historyOffset)
				for a in Actions:
					for o in Obs:
						toExpandhistory = chao + ((a,o),)
						temp.append(toExpandhistory)
						toExpand.append(toExpandhistory)


			historyOffset=temp
		print "o"
		
def generalFormula(actionobs,test,history,depth=0):
	#actionobstest=list(actionobstest)
	#actionobs=actionobstest.pop(0)
	#test=tuple(actionobstest)
	
	
	actionobstest=(actionobs,)+test
#	print "t",actionobs,test,history,history in D,actionobstest,type(D[history])
	
	if (history in D) and ((actionobstest in D[history]) or (type(D[history])==defaultdict)): return D[history][actionobstest]
	
	historyactionobs= history + (actionobs,)
	if (historyactionobs not in D) or ((historyactionobs in D) and (test not in D[historyactionobs])):
	#	print "all",test[0],test[1:],historyactionobs,depth+1
		D[historyactionobs][test]=generalFormula(test[0],test[1:],historyactionobs,depth+1)
	return D[history][actionobs]*D[historyactionobs][test]
					
			
		
		

class TransferLearningAgent(ReinforcementAgent):
	observationHistory = []
	actionHistory = []
	
	def __init__(self,numTraining=0, **args):
		args['numTraining'] = numTraining
		ReinforcementAgent.__init__(self, **args)

			
	def getAction(self,state):

		modelLinearPSR(state)
		action = random.choice(['East','West','North','South'])
		self.actionHistory.append(action)
		self.doAction(state,action)
		return 'East'
		
	def update(self, state, action, nextState, reward):
		#print self.initialState.getPacmanPosition()
		self.observationHistory.append(tuple([nextState.getObservations(),reward]))
		
	def final(self, state):
		print self.observationHistory
		print self.actionHistory
		#print state.getScore()
		pass
class QLearningAgent(ReinforcementAgent):
    
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        
        "*** YOUR CODE HERE ***"
        
        self.QValue = util.Counter()
        self.LegalActions = util.Counter()
        self.NoTimes = util.Counter()
        self.bonus = 10
        
        
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        bonus=float(self.bonus)/float(self.NoTimes[(state,action)]+1)
        
        
        return self.QValue[(state,action)]+bonus
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        
        self.LegalActions[state] = self.getLegalActions(state)
        
        if len(self.LegalActions[state])==0:
            return 0
        v=-float("infinity")
        for action in self.LegalActions[state]:
             v=max(v,self.getQValue(state,action))
        
        return v
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        if state not in self.LegalActions:
            
        
            self.LegalActions[state] = h
        legalActions = self.LegalActions[state]
        if len(self.LegalActions[state]) == 0:
            return None
        
        ActionDic={}
        for action in self.LegalActions[state]:
            
            QValueOfAction=self.getQValue(state,action)
            if QValueOfAction not in ActionDic:
                ActionDic[QValueOfAction] = []
            ActionDic[QValueOfAction] += [action]
        
            
        
        return random.choice(ActionDic[max(ActionDic)])
        
        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        
        # Pick Action
        if state not in self.LegalActions:
            
        
            self.LegalActions[state] = self.getLegalActions(state)
        legalActions = self.LegalActions[state]
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        
        return self.getPolicy(state)
        
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        return action
    
            
        
    
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        if state not in self.LegalActions:
            legalActions = self.getLegalActions(state)
            self.LegalActions[state] = legalActions
        self.NoTimes[(state, action)]+=1
        
        
        WhatWeGot = reward + self.discount*self.getValue(nextState)
        
        Difference = WhatWeGot - self.QValue[(state, action)]
        self.QValue[(state, action)] += self.alpha*Difference
        
        return self.QValue[(state, action)] 
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)
        
    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        
    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        
        
        return self.weights*self.featExtractor.getFeatures(state,action)
        
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        WhatWeGot = reward + self.discount*self.getValue(nextState)
        Difference = WhatWeGot - self.getQValue(state, action)
        Allfeatures = self.featExtractor.getFeatures(state,action)
        
        #print Allfeatures
         
        for oneFeature in Allfeatures:
            self.weights[oneFeature] += self.alpha*Difference*Allfeatures[oneFeature]
       
        return self.weights
        
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
tla = TransferLearningAgent
