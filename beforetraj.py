# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
import sympy as smp
import sklearn
#from sklearn import MLPRegressor
import matlab.engine
eng = matlab.engine.start_matlab()
import numpy as np
import scipy as sp
from scipy import linalg
import types
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from pacman import GameState
from collections import defaultdict
import random,util,math
import pickle
import time
lookup = {}
pellets = []

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
		  
def possibleMeasurements(pos,b,query = 0):
	trueObservation = sampleState.getObservations(pos,[1,0])
	correctProb = sampleState.sensorDynamics[0]
	flag = pellets[pos[0]][pos[1]]
	reward =  -1 * (not flag) + 10* flag
	
	if sampleState.pelletDistance:
		if query:
			if reward != query[1]: return 0
			
			dist = abs(query[0] - trueObservation)
			if dist == 1: return 0.5*(1-correctProb)*b
			elif dist ==0:  return correctProb*b
			else: return 0
		possibility ={}

		possibility[(trueObservation+1,reward)] = 0.5*(1-correctProb)*b
		possibility[(trueObservation-1,reward)] = 0.5*(1-correctProb)*b
		possibility[(trueObservation,reward)] = correctProb*b

		return possibility
	l = len(sampleState.obsDimension)

	if query:
		if reward != query[1]: return 0
		mismatch = lookup[trueObservation^query[0]]
		return ((1-correctProb)**mismatch * correctProb**(l-mismatch))*b
	
	possibility ={}
	
	for i in xrange(2**l):
		mismatch = lookup[trueObservation^i]
		possibility[(i,reward)] = ((1-correctProb)**mismatch * correctProb**(l-mismatch))*b
	return possibility
	
Actions = ['North','West','South','East']
Obs = []
def normalize(bel):
	alpha = sum(bel.values())
	if alpha == 0: return bel
	for i in bel:
		bel[i] = float(bel[i])/alpha
		
	return bel
	
D ={}


universalBelief = {}
universalBelief[((),)] ={}
sampleState = 0
def selectAction(QN,belief):
    action=argmax([QN.predict(np.append(belief,ActionAppend[a])) for a in Actions])
    return action

	
def trainQNetworkPSR(belTraj):
    QN = MLPRegressor(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    train=[]
    InputSet=[]
    OutputSet=[]
    ActionAppend={'North':np.array(1,0,0,0),'East':np.array(0,1,0,0),'West':np.array(0,0,1,0),'South':np.array(0,0,0,1)}
    for b in belTraj:
        bT,aT,rT,bnT=b
        i=np.append(bT,ActionAppend[aT])
        o=rT+discount*max([QN(np.append(bnT,ActionAppend[a])) for a in Actions])
        InputSet.append(i)
        OutputSet.append(o)
    QN.fit(InputSet,OutputSet)    
    return QN

def buildBeliefSetPSR(trajectory,modelPSR):
    M,initQ=modelPSR
    belTraj=[]
    currentBelief=initQ
    for ao in trajectory:
        prevBelief=currentBelief
        ao=(ao,)
        currentBelief=beliefUpdatePSR(M,ao,currentBelief)
        belTraj.append((prevBelief,ao[0],ao[1][1],currentBelief))
    return belTraj 
		
def posGenerateSuccessor(pos,action):
	global sampleState
	actiontoOffsetMap = {'North':(0,1),'West':(-1,0),'South':(0,-1),'East':(1,0)}
	offset = actiontoOffsetMap[action]
	ret = pos[0]+offset[0],pos[1]+offset[1]
	if sampleState.hasWall(ret[0],ret[1]): return pos
	else: return ret
def outputModel(coreTest,coreHistory,D_mat):
		mao={}
		Maoq={}
		for a in Actions:
			for o in Obs:
				actionObs=((a,o),)
				temp=np.zeros(len(coreHistory))
				#print "\nc",coreHistory
				for i,ch in enumerate(coreHistory):
				#	print "\nc",coreHistory
					temp[i]=D[ch][actionObs]
				mao[actionObs]=np.dot(np.linalg.inv(D_mat),temp)
				for ct in coreTest:
					temp=np.zeros(len(coreHistory))
					aoct=actionObs+ct
					for i,ch in enumerate(coreHistory):
						temp[i]=D[ch][aoct]
					if actionObs not in Maoq:Maoq[actionObs]=[]	
					Maoq[actionObs].append(np.dot(np.linalg.inv(D_mat),temp))
		return mao,Maoq
		
def beliefUpdatePSR(M,actionObs,currentBelief):
	mao,Maoq = M

	newBelief=np.dot(currentBelief,np.asarray(Maoq[actionObs]).T)/np.dot(currentBelief,mao[actionObs])
	return newBelief		
	
def beliefSuccessor(belief, history = (),test=()):
	global sampleState
	


	if test:
		for t in xrange(len(test)):
			
			if t-1 >= 0: history += (test[t-1],)

			if (history in universalBelief):
				if (test[t] in universalBelief[history]): 
					belief = universalBelief[history][test[t]]
					continue
			
			if (history in D and (type(D[history])==defaultdict)): return

			action = test[t][0]
			newBelief = {}
			possibleActionOutcomes = possibleTransitions(action)
			for i in possibleActionOutcomes:
				for state in belief:
					b = belief[state]
					childProb = b * possibleActionOutcomes[i]
					child = posGenerateSuccessor(state,i)
					if child not in newBelief: newBelief[child] = childProb
					else: newBelief[child] += childProb

			possibleSensorOutcomes = 0
			b_dash = {}
			listOfBelief = {}
			obs = test[t][1]
			for state in newBelief:
				
				
				b = newBelief[state]
				o = possibleMeasurements(state,b,obs)				#make better#done!!
				
				if o>0: 
					listOfBelief[state] = o

					
					possibleSensorOutcomes += o
			if history not in D: D[history]={}
			D[history][(test[t],)] = possibleSensorOutcomes

			listOfBelief = normalize(listOfBelief)
			if history not in universalBelief: universalBelief[history] = {}
			universalBelief[history][test[t]] = listOfBelief
			
			belief =listOfBelief

		return

	
	global Obs
	for action in Actions:
		newBelief = {}
		possibleActionOutcomes = possibleTransitions(action)
		for i in possibleActionOutcomes:
			for state in belief:
				b = belief[state]
				childProb = b * possibleActionOutcomes[i]
		
				child = posGenerateSuccessor(state,i)
				if child not in newBelief: newBelief[child] = childProb
				else: newBelief[child] += childProb
		possibleSensorOutcomes = {}
		b_dash = {}
		listOfBelief = {}
		for state in newBelief:
			b = newBelief[state]
			o = possibleMeasurements(state,b)
			
			for i in o:
				if i not in listOfBelief: listOfBelief[i]={} 
				listOfBelief[i][state] = o[i]
				
				if i not in possibleSensorOutcomes: possibleSensorOutcomes[i]= o[i]
				else: possibleSensorOutcomes[i]+= o[i]
		if history not in D: D[history]={}
		for i in Obs:	
			
			if i in possibleSensorOutcomes: D[history][((action,i),)] = possibleSensorOutcomes[i]
			else: D[history][((action,i),)] = 0
		for obs in listOfBelief:
			listOfBelief[obs] = normalize(listOfBelief[obs])
			if history not in universalBelief: universalBelief[history] = {}
			universalBelief[history][(action,obs)] = listOfBelief[obs]
'''
def independentRows(matrix):
	matrix = matrix.T*10000
	matrix=smp.Matrix(matrix)
	independentCols=matrix.rref()[1]
	return independentCols
'''	
def independentRows(A, tol = 0.001):
    
    '''
    shape =  np.shape(A)

    smaller = shape.index(min(shape))

    larger = (not smaller)*1

    if larger:
       new = (A[shape[smaller]-1,:])
       new = np.array([list(new)]*(shape[larger]-shape[smaller]))
       A=np.append(A,new,axis=smaller)
    else:
       new = (A[:,shape[smaller]-1])
       print np.shape(new)
       new = np.array([list(new)]*(shape[larger] - shape[smaller]))
       new = new.T
       
       A=np.append(A,new,axis=smaller)
    print np.shape(A),'here'
    # new = np.zeros([larger*(shape[larger] - shape[smaller])+smaller*shape[larger],larger*shape[larger]+smaller*(shape[larger] - shape[smaller])])
    '''


    


    #A = smp.Matrix(A)
    #Q, R = A.QRdecomposition()
    A_t = A.T
    [Q, R, E] = eng.qr(matlab.double(A_t.tolist()),0.0,nargout=3)
    R = np.asarray(R)
   
    if min(np.shape(R)) > 1: R = abs(np.diag(R))
    else: R = R[0]
    rank = 1
    for i,r in enumerate(R):
        if r<R[0]*tol:
            rank=i
            break
    indRows = np.sort(np.asarray(E[0][:rank],int)-1)
    [Q, R, E] = eng.qr(matlab.double(A.tolist()),0.0,nargout=3)
    R = np.asarray(R)
    if min(np.shape(R)) > 1: R = abs(np.diag(R))
    else: R = R[0]

            
            
    indCols = np.sort(np.asarray(E[0][:rank],int)-1 )     

    return (indCols,indRows)
    
    
    '''
    if ~isvector(R)
        diagr = abs(diag(R));
    else
        diagr = R(1);   
    end
    diagr
    r = find(diagr >= tol*diagr(1), 1, 'last'); %rank estimation
    idx=sort(E(1:r));
    Q, R  = linalg.qr(A)
	
	
    independent = np.where(np.abs(R.diagonal()) > tol)[0]
    print independent
    return independent
    '''

def removeZeros(A):
	selected = []
	
	for i,row in enumerate(A):

		for ele in row: 
			if ele>0: 
				selected.append(i)
				break
 

	return (selected,A[selected,:])
								
								
								
def generalFormula(actionobs,test,history,depth=0):
	
	actionobstest=(actionobs,)+test	
	if (history in D) and ((actionobstest in D[history]) or (type(D[history])==defaultdict)): return D[history][actionobstest]
	historyactionobs= history + (actionobs,)
	if (historyactionobs not in D) or ((historyactionobs in D) and (test not in D[historyactionobs])):
		D[historyactionobs][test]=generalFormula(test[0],test[1:],historyactionobs,depth+1)
	D[history][actionobstest] = D[history][(actionobs,)]*D[historyactionobs][test]
	return D[history][actionobstest]
	
	
def construct(D_mat,coreHistory,coreTest,prevRankD):
	rowOffset=D_mat.shape[0]
	colOffset=D_mat.shape[1]

	terminate=False
	if rowOffset == 0: D_mat = np.zeros([1+len(Actions)*len(Obs),len(Actions)*len(Obs)])
	else: 
		
		new_cols=np.zeros([D_mat.shape[0],D_mat.shape[1]*len(Actions)*len(Obs)])
		D_mat=np.append(D_mat,new_cols,axis=1)
		new_rows=np.zeros([D_mat.shape[0]*len(Actions)*len(Obs),D_mat.shape[1]])
		D_mat=np.append(D_mat,new_rows,axis=0)

	oneStepHistory=coreHistory[:]
	oneStepTest=coreTest[:]

	for a in Actions:
		for o in Obs:

			for ch in coreHistory:
				if ch==():
						oneStepHistory.append(((a,o),))
				else:	
						oneStepHistory.append(  ch+ ((a,o),)  )
			if coreTest==[]:
					oneStepTest.append(((a,o),))
					continue	
			for ct in coreTest: 
					oneStepTest.append(  ((a,o),)  +  ct) 	

	row,col = 0,colOffset

	for h in oneStepHistory:
		for t in oneStepTest[len(coreTest):]:

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

	#if ((('West', (0, -1)), ('West', (0, -1))) in D) and ((('West', (1, -1)), ('North', (3, -1))) in D[(('West', (0, -1)), ('West', (0, -1)))]) :
	#	print D[(('West', (0, -1)), ('West', (0, -1)))][(('West', (1, -1)), ('North', (3, -1)))]
	#	stop()
	(ind,D_mat)= removeZeros(D_mat)

	oneStepHistory = [oneStepHistory[i] for i in ind]
	(ind,D_mat)= removeZeros(D_mat.T)
	D_mat = D_mat.T
	
	oneStepTest = [oneStepTest[i] for i in ind]
  
	(indCols,indRows) = independentRows(D_mat)
  
	
	D_mat = D_mat[np.ix_(indRows,indCols)]
	print 'numpy',np.linalg.matrix_rank(D_mat),'numpyT',np.linalg.matrix_rank(D_mat.T),'shape after conversion', np.shape(D_mat)
	curRankD=len(indRows)
	if curRankD==prevRankD:
		 terminate=True
	newCoreHistory=[oneStepHistory[index] for index in indRows]
	newCoreTest=[oneStepTest[index] for index in indCols] 
	return D_mat,newCoreHistory,newCoreTest,terminate,curRankD
	
	
	    		
def modelLinearPSR(state):
	currentDmat = np.zeros([0,0])
	toExpand = [[(),[(),()]]]
	global Obs
	global pellets
	global sampleState 
	global lookup
	for i in range(2**len(state.obsDimension)):
		lookup[i] = bin(i).count('1')
	sampleState = state
	pellets = state.getFood()
	ini = {state.getPacmanPosition():1} #remove
	universalBelief[((),)][()]= {state.getPacmanPosition():1}
	rank = -1
	Obs = state.getObservationset()
	coreHistory=[()]
	coreTest=[]
	noOfStates = 0
	
	(width , height) = sampleState.getDimensions()
	for i in xrange(width):
		for j in xrange(height):
			if not sampleState.hasWall(i,j): noOfStates+=1

	
	for a in Actions:
		for o in Obs:
			toExpand.append(((),((a,o),)))

	
	converged = 0
	N = 0              # N is the dimension of D found till now
	extra=True
	finalCoreHistory=[]
	finalCoreTest=[]
	while (not converged and len(coreTest) < noOfStates) or extra:
		
		if not(not converged and (len(coreTest) < noOfStates)) and extra:
			finalCoreHistory = coreHistory
			finalCoreTest = coreTest
			pastDmat=currentDmat
			extra=False
		'''
		if not extra and myflag:
			extra= 1
			myflag = 0
		elif not myflag:
			extra = 0
		'''
		while toExpand:
			hao = toExpand.pop(0)

			t = hao[0]
			hao = hao[1]
			newHistory = tuple(hao)
			history = newHistory[:-1]
			actionobs = newHistory[-1]
			if newHistory==((),()):newHistory=()
			if (history not in universalBelief) or ((history in universalBelief) and (actionobs not in universalBelief[history])):
				if history not in D:D[history]={}
				D[history][(actionobs,)]=0
				D[newHistory]=defaultdict(lambda: 0,{})
				continue
			beliefSuccessor(universalBelief[history][actionobs],newHistory,t)
		toExpand = []

		
		
		print 'constructing dmat'
		currentDmat,coreHistory,coreTest,converged,rank = construct(currentDmat,coreHistory,coreTest,rank)
		print 'construction of dmat over'
		
		N+=1

		print N,rank
		historyOffset = coreHistory[:]
		for i in xrange(2):
			temp = []
			while historyOffset:
				chao =historyOffset.pop(0)
				for a in Actions:
					for o in Obs:
						toExpandhistory = chao + ((a,o),)
						temp.append(toExpandhistory)
						toExpand.append(toExpandhistory)
			historyOffset=temp
		
		temp = []
		for ct in coreTest:
			for ele in toExpand:
				temp.append((ct,ele))
		toExpand = temp

		print N, coreHistory[[len(x) for x in coreHistory].index(max([len(x) for x in coreHistory]))]

	M= outputModel(finalCoreTest,finalCoreHistory,pastDmat)	
	initQ=np.zeros(len(finalCoreTest))
	for i,ct in enumerate(finalCoreTest):
		initQ[i]=D[()][ct]
	modelPSR=[M,initQ]	
	#pickle.dump([modelPSR,finalCoreTest],open("source2.txt","wb"))
	print 'done'
	stop()
	trajectory= (('North', (4, -1)), ('North', (7, -1)), ('West', (4, -1)),('West', (4, -1)))

	currentBelief=initQ
	for ao in trajectory:
		ao=(ao,)
		currentBelief=beliefUpdatePSR(M,ao,currentBelief)
	print currentBelief
	
def createTrajectories(state):
	pos = state.getPacmanPosition()
	
	noOfTrajectories = 100
	length = 100
	while noOfTrajectories>0:
		for i in xrange(length):
			for action in Actions:
			newBelief = {}
			possibleActionOutcomes = possibleTransitions(action)
			for i in possibleActionOutcomes:
				for state in belief:
					b = belief[state]
					childProb = b * possibleActionOutcomes[i]
			
					child = posGenerateSuccessor(state,i)
					if child not in newBelief: newBelief[child] = childProb
					else: newBelief[child] += childProb
			possibleSensorOutcomes = {}
			b_dash = {}
			listOfBelief = {}
			for state in newBelief:
				b = newBelief[state]
				o = possibleMeasurements(state,b)
				
				for i in o:
					if i not in listOfBelief: listOfBelief[i]={} 
					listOfBelief[i][state] = o[i]
					
					if i not in possibleSensorOutcomes: possibleSensorOutcomes[i]= o[i]
					else: possibleSensorOutcomes[i]+= o[i]
			if history not in D: D[history]={}
			for i in Obs:	
				
				if i in possibleSensorOutcomes: D[history][((action,i),)] = possibleSensorOutcomes[i]
				else: D[history][((action,i),)] = 0
			for obs in listOfBelief:
				listOfBelief[obs] = normalize(listOfBelief[obs])
				if history not in universalBelief: universalBelief[history] = {}
				universalBelief[history][(action,obs)] = listOfBelief[obs]
		
		
		

class TransferLearningAgent(ReinforcementAgent):
	observationHistory = []
	actionHistory = []
	
	def __init__(self,numTraining=0, **args):
		args['numTraining'] = numTraining
		ReinforcementAgent.__init__(self, **args)

			
	def getAction(self,state):
		
		
		modelLinearPSR(state)
		createTrajectories(state)
		action = 'East'
		self.actionHistory.append(action)
		self.doAction(state,action)
		return action
		
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
