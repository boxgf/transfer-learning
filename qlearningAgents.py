# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

# Transfer Learning was developed by Siddharthan Rajasekaran
# All the functionalities of the transfer learning agent including the Genetic Algorithm was extended by Siddharthan
import sympy as smp
# sklearn not required if you have aproblem you can comment it. for more info read README
import sklearn
from sklearn import neural_network
#from sklearn.neural_network import MLPRegressor
import matlab.engine

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
import util
pellets = []
eng = 0
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
    for i,b in enumerate(belTraj):
        bT,aT,rT,bnT=b
        inp=np.append(bT,ActionAppend[aT])
        if i==0:o=rT
        else:o=rT+discount*max([QN.predict(np.append(bnT,ActionAppend[a])) for a in Actions])
        #InputSet.append(i)
        #OutputSet.append(o)
        #QN.fit(InputSet,OutputSet)
        QN.fit(inp,o)    
    return QN

def buildBeliefSetPSR(trajectory,modelPSR):
    M,initQ=modelPSR
    belTraj=[]
    currentBelief=initQ
    for ao in trajectory:
        prevBelief=currentBelief
        ao=(ao,)
        currentBelief=beliefUpdatePSR(M,ao,currentBelief)
        belTraj.append((prevBelief,ao[0][0],ao[0][1][1],currentBelief))
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
		inverse = np.linalg.pinv(D_mat)
		for a in Actions:
			for o in Obs:
			
				actionObs=((a,o),)
				
				temp=np.zeros(len(coreHistory))
				for i,ch in enumerate(coreHistory):
					temp[i]=generalFormula(actionObs[0],(),ch) #D[ch][actionObs]
				
				mao[actionObs]=np.dot(inverse,temp)
				for ct in coreTest:
					temp=np.zeros(len(coreHistory))
					aoct=actionObs+ct
					for i,ch in enumerate(coreHistory):
						temp[i]=generalFormula(actionObs[0],ct,ch)#D[ch][aoct]
						
						
					if actionObs not in Maoq:Maoq[actionObs]=[]
						
					Maoq[actionObs].append(np.dot(inverse,temp))
					
				Maoq[actionObs] = np.asarray(Maoq[actionObs]).T
		return mao,Maoq
		
def beliefUpdatePSR(M,actionObs,currentBelief,tol = 1e-10):
	mao,Maoq = M
	
	#print 'here',np.dot(currentBelief,mao[actionObs]),np.dot(currentBelief,np.asarray(Maoq[actionObs]).T)
	denominator = np.dot(currentBelief,mao[actionObs])
	ret = np.dot(currentBelief,Maoq[actionObs])/denominator
	#ret = ret * (abs(ret)>tol)
	return ret
	#else : return currentBelief*0
	
def trajectoryBeliefUpdatePSR(M,traj,currentBelief,tol = 1e-10):
	mao,Maoq = M
	
	for actionObs in traj:
		actionObs = (actionObs,)
	
		denominator = np.dot(currentBelief,mao[actionObs])
		currentBelief = np.dot(currentBelief,Maoq[actionObs])/denominator
	#currentBelief = currentBelief * (abs(currentBelief)>tol)
	return currentBelief	
Obs_p = 0
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
def independentRows(A, tol = 1e-10):
    
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
    E_row = E
    
    if min(np.shape(R)) > 1: R = abs(np.diag(R))
    else: R = R[0]
    rank = 1
    for i,r in enumerate(R):
        if r<R[0]*tol:
            rank=i
            break
    rank_row = rank
    
    [Q, R, E] = eng.qr(matlab.double(A.tolist()),0.0,nargout=3)
    R = np.asarray(R)
    E_col= E
    if min(np.shape(R)) > 1: R = abs(np.diag(R))
    else: R = R[0]


    rank = 1
    for i,r in enumerate(R):
        if r<R[0]*tol:
            rank=i
            break
    rank_col = rank 
    
    #rank = min(rank_row,rank_col)  
    indRows = np.sort(np.asarray(E_row[0][:rank_row],int)-1)  
    indCols = np.sort(np.asarray(E_col[0][:rank_col],int)-1 )     

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
								
def projection(M,test,history=0):
	m_t,M_t = M
	M_proj = []
	for i,ct in enumerate(test):
		
		m_proj = m_t[(ct[-1],)]

		for j,ao in enumerate(ct[:-1]):
			m_proj = np.dot(M_t[(ct[-(j+2)],)],m_proj )
		M_proj.append(m_proj)
	 
	return M_proj
					
								
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
	print 'shape after conversion', np.shape(D_mat)
	curRankD=len(indRows)
	if curRankD==prevRankD:
		 terminate=True
	newCoreHistory=[oneStepHistory[index] for index in indRows]
	newCoreTest=[oneStepTest[index] for index in indCols] 
	return D_mat,newCoreHistory,newCoreTest,terminate,curRankD
	
init_belief = 0
def initialize_globals(state):

	global init_belief
	global Obs_p
	global Obs
	global pellets
	global sampleState 
	global lookup
	Obs = state.getObservationset()	
	length = len(Obs)/2
	
	space = (state.getDimensions()[0]-2)*(state.getDimensions()[1]-2)

	Obs_p = [space-1]*length +[1]*length

	Obs_p = np.asarray(Obs_p).astype(float)/sum(Obs_p)


	for i in range(2**len(state.obsDimension)):
		lookup[i] = bin(i).count('1')
	sampleState = state
	pellets = state.getFood()  
	init_belief =  {state.getPacmanPosition():1} 
	universalBelief[((),)][()]= init_belief
	

def modelLinearPSR(state):
	global eng
	eng = matlab.engine.start_matlab()
	toExpand = [[(),[(),()]]]
	currentDmat = np.zeros([0,0])
	rank = -1
	
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

	noOfStates 
	converged = 0
	N = 0              # N is the dimension of D found till now
	extra=True
	finalCoreHistory=[]
	finalCoreTest=[]
	while not converged  or extra:
		if converged and extra:
			finalCoreHistory = coreHistory
			finalCoreTest = coreTest
			pastDmat=currentDmat
			
			extra=False
		print 'construction of dmat over'

		'''
		if not extra and myflag:
			extra= 1
			myflag = 0
		elif not myflag:
			extra = 0
		'''
		for hao in toExpand:
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
		print converged

		
		N+=1
		print N,rank
		aotoExpand =[]
		for ch in coreHistory:
			if ch:
				aotoExpand.append(((),ch))
				for a in Actions:
					for o in Obs:
						ao = ((a,o),)
						aotoExpand.append(((),ch+ao))			
		
		historyOffset = coreHistory[:]
		for i in xrange(2):
			temp = []
			for chao in historyOffset:
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
		toExpand = aotoExpand + toExpand
		print N #,'len', len(coreTest), coreHistory[[len(x) for x in coreHistory].index(max([len(x) for x in coreHistory]))]

	M= outputModel(finalCoreTest,finalCoreHistory,pastDmat)	
	initQ=np.zeros(len(finalCoreTest))
	
	for i,ct in enumerate(finalCoreTest):
		beliefSuccessor(init_belief, () ,ct)
		initQ[i] = generalFormula(ct[0],ct[1:],())
	for i,ct in enumerate(finalCoreTest):
		
		initQ[i] = generalFormula(ct[0],ct[1:],())
	modelPSR=[M,initQ]	
	#print generalFormula(('East',(3,-1)),(('East',(3,-1)),)*0+ (('East',(3,10)),),())
	#stop()
	store = "newSrc2.txt"
	pickle.dump([modelPSR,finalCoreTest],open(store,"wb"))
	print 'done stored in', store
	
	stop()
	return (modelPSR,finalCoreHistory)
	
def createTrainingSet(model,state):
	
	
	noOfTrajectories = 100
	length = 100
	trainingSet = []
	while noOfTrajectories>0:
		noOfTrajectories -= 1
		traj = []
		pos = state.getPacmanPosition()
		for i in xrange(length):
			action = random.choice(Actions)
			possibleActionOutcomes = possibleTransitions(action)
			keys = possibleActionOutcomes.keys()
			values = [possibleActionOutcomes[i] for i in keys]
			act_action = np.random.choice(keys,1,p=values)[0]
			child = posGenerateSuccessor(pos,act_action)
			o = possibleMeasurements(child,1)
			keys = o.keys()
			values = [o[i] for i in keys]
			obs = keys[np.random.choice(range(len(keys)),1,p=values)]
			pos = child
			traj.append((action,obs),)
		trainingSet+=buildBeliefSetPSR(traj,model)
	return trainingSet

			
		
def selectBestSource(M_proj,history,allBelief):
	belief_proj = {}
	for i in M_proj:
		belief_proj[i] = np.dot(M_proj[i],allBelief['t'])
	
	#traj,s_belief,t_belief,source_M, M_proj,cap = 1000
	maxi = [-float('inf'),0]
	for i in belief_proj:
			belief_proj[i] = fitness((),allBelief[i],allBelief['t'],(0,0), M_proj[i])
			if maxi[0] < belief_proj[i]:
				maxi[0] = belief_proj[i]
				maxi[1] = i
	print maxi[1],belief_proj
	return maxi[1]
	
				
def fitness(projS,projT,trajProb):
	



	metric = np.dot(projS/np.linalg.norm(projS),projT/np.linalg.norm(projT)) * trajProb
	if metric <0: return 0


		
	return metric
	
	
import math

def select(pop,prob,percent):
	
	#pop = range(1,pop+1)

	
	n=len(pop)
	weight = np.asarray(range(n))*2.0/(n*(n-1))
	pop = np.asarray(pop)
	pop = pop.astype(float)/sum(pop)
	
	return np.random.choice(range(n), percent*.01*n, p = weight)
	
def cross(curPopulation,selected):
	temp = []
	paired = []
	for i in selected:
		temp.append(i)
		if len(temp) ==2:
			paired.append(temp)
			temp = []
	crossed = []
	for i,pair in enumerate(paired):
		parent1 = curPopulation[pair[0]][0]
		parent2 = curPopulation[pair[1]][0]
		maxLength = max(len(parent1),len(parent2))
		if maxLength >1:
			minLength = min(len(parent1),len(parent2))

			prob = [100]*(minLength-1) + [1]*(maxLength- minLength)
			prob = np.asarray(prob).astype(float)/sum(prob)
			cutPoint = np.random.choice(range(1,maxLength),1, p =  prob)[0]
		else: cutPoint = 0
		child1 = parent1[:cutPoint]+parent2[cutPoint:]
		child2 = parent2[:cutPoint]+parent1[cutPoint:]


		crossed.append(child1)
		crossed.append(child2)
	return crossed

	
def mutate(crossed,prob,maxLength,threshold = 1e6):


	for k,history in enumerate(crossed):
		
		for i,ele in enumerate(history):
			
			if util.flipCoin(prob):
				#if util.flipCoin(threshold * 1./fitness(ele,s_belief,t_belief,source_M, M_proj))
				if util.flipCoin(0.8):
					if util.flipCoin(0.5):
						mutated = list(ele)
						mutated[0] = random.choice(Actions)
						crossed[k][i] = tuple(mutated)
					else:
						mutated = list(ele)
						mutated[1] = Obs[np.random.choice(range(len(Obs)),1,p=Obs_p)[0]]
						crossed[k][i] = tuple(mutated)
				else:
					mutated = list(ele)
					mutated[0] = random.choice(Actions)
					mutated[1] = Obs[np.random.choice(range(len(Obs)),1,p=Obs_p)[0]]
					crossed[k][i] = tuple(mutated)
		if util.flipCoin(prob):
			if util.flipCoin(0.5) and len(history) < maxLength:
				mutated = [0,0]
				#mutated[0] = random.choice(Actions)
				#mutated[1] = Obs[np.random.choice(range(len(Obs)),1,p=Obs_p)[0]]
				mutated =  history[-1]
				crossed[k] = history + [tuple(mutated)] 

			elif len(history)>=maxLength-1:
				crossed[k].pop()
	return crossed
	
	
def kill(pop,prob,perc):
	
	half = len(pop)/2
	good = pop[half:]
	good_p = [x[1] for x in good]
	if sum(good_p): good_p = [float(x)/sum(good_p) for x in good_p]
	else: return good
	bad = pop[:half]
	selected = []
	
	for i in xrange(int(len(pop)*100/perc)):
		if util.flipCoin(prob):
			selected.append(random.choice(bad))
		else:
			selected.append(good[np.random.choice(range(len(good)),1,p = good_p)[0]])
	
	return selected
	
	
def findHistoryOffset(sBelief,tBelief,MprojT,MprojS,sM,totalPopulation = 30,iters = 100):

	AOspace = []
	for o in Obs:
		for a in Actions:
			AOspace.append((a,o))
	best = {}
	
	memberLength = range(1,8)
	projT = np.dot(MprojT,tBelief)
	curPopulation = []
	length = len(AOspace)/2
	prob = [Obs_p[0]] * length + [Obs_p[-1]] * (len(AOspace) - length)
	prob = [float(i)/sum(prob) for i in prob]
	
		
	for i in xrange(totalPopulation):
		selected = np.random.choice(range(len(AOspace)),random.choice(memberLength),p = prob) 
		curPopulation.append([AOspace[j] for j in selected])
	s = 0
	jk = 0

	percentRandom = 0.1
	percentageExplosion = 200
	solved = 0
	allProj = projection(sM,curPopulation)
	for i,ele in enumerate(curPopulation):
		
		curPopulation[i] = [ele, fitness(np.dot(MprojS,trajectoryBeliefUpdatePSR(sM,ele,sBelief)),projT,np.dot(sBelief,allProj[i]))]
	
		
		
	maxSimililar = [0,-float('inf')]
	
	for epochs in xrange(iters):
		
		randomAddition = []
		for i in xrange(int(totalPopulation*percentRandom)):
			selected = np.random.choice(range(len(AOspace)),random.choice(memberLength),p = prob) 
			history = [AOspace[j] for j in selected]
			randomAddition.append([history,fitness(np.dot(MprojS,trajectoryBeliefUpdatePSR(sM,history,sBelief)),projT,np.dot(sBelief,projection(sM,(history,))[0]))])
		for i in randomAddition:
			curPopulation[random.choice(range(int(totalPopulation*(1-percentRandom))))] = i

		#curPopulation[-3] = [[('East', (3, -1)),('East', (3, -1)),('East', (3, -1)),('East', (3, -1))], fitness(ele,np.dot(MprojS,trajectoryBeliefUpdatePSR(sM,ele,sBelief)),projT,np.dot(sBelief,projection(sM,(ele,))[0]))]
		
		curPopulation = sorted(curPopulation, key=lambda x: x[1])
		if maxSimililar[1] < curPopulation[-1][1]:
			maxSimililar = curPopulation[-1]
			
		selected = select([x[1] for x in curPopulation], 0.2,percentageExplosion)


		crossed = cross(curPopulation,selected)
		mutated = mutate(crossed,0.25,len(memberLength))
		
		curPopulation = mutated
		allProj = projection(sM,curPopulation)
		for i,ele in enumerate(curPopulation):
			
			curPopulation[i] = [ele,fitness(np.dot(MprojS,trajectoryBeliefUpdatePSR(sM,ele,sBelief)),projT,np.dot(sBelief,allProj[i]))]
		curPopulation = sorted(curPopulation, key=lambda x: x[1])
		curPopulation = kill(curPopulation,0.25,percentageExplosion)
	return maxSimililar
from sklearn.svm import SVR

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


ActionAppend={'North':np.array([1,0,0,0]),'East':np.array([0,1,0,0]),'West':np.array([0,0,1,0]),'South':np.array([0,0,0,1])}
def selectAction(QN,belief):
    QV= [QN.predict(np.append(belief,ActionAppend[a])) for a in Actions]
    action=np.argmax(QV)
    return Actions[action]
    	
class TransferLearningAgent(ReinforcementAgent):
	observationHistory = []
	actionHistory = []
	first = 1
	allBelief ={}
	target_M = 0
	source1_M = 0
	source2_M = 0
	M_proj = {}
	policy =  {'1':'East','2': 'West'}
	QN = pickle.load(open("myQN","rb"))
	#modelPSR=[M,initQ]
	#pickle.dump([modelPSR,finalCoreTest],open("newtarget.txt","wb"))
	
	#validatingTest = ((('East',(3,-1)),)*1 + (('East',(3,10)),),(('East',(3,-1)),)*3+ (('East',(3,10)),),(('East',(3,-1)),)*5 + (('East',(3,10)),),(('West',(3,-1)),)*1 + (('West',(3,10)),),(('West',(3,-1)),)*3 + (('West',(3,10)),),(('West',(3,-1)),)*5 + (('West',(3,10)),))
	validatingTest = ((('South',(2,-1)),)*2+(('East',(1,-1)),)*2,(('East',(1,-1)),)*2, (('North',(2,-1)),('North',(0,-1)),('East',(1,-1)),('East',(1,-1))) ,(('North',(2,-1)),('North',(0,-1)),('East',(2,-1)),('East',(2,10))))
	#(('East',(2,-1)),('North',(2,-1)),('North',(0,-1)))
	def __init__(self,numTraining=0, **args):
		args['numTraining'] = numTraining
		ReinforcementAgent.__init__(self, **args)

			
	def getAction(self,state):
		
		if self.first: 
			initialize_globals(state)
			self.first =0
			
			#model,fc = modelLinearPSR(state)
			#stop()
			[self.target_M,self.allBelief ['t']],ct = pickle.load(open("newTarget.txt","rb"))
			[self.source1_M,self.allBelief ['1']],ct = pickle.load(open("newSrc1.txt","rb"))
			[self.source2_M,self.allBelief ['2']],ct = pickle.load(open("newSrc2.txt","rb"))
			mao,Maoq = self.target_M
			
			self.M_proj['t'] = projection(self.target_M,self.validatingTest)
			self.M_proj['1'] = projection(self.source1_M,self.validatingTest)
			self.M_proj['2'] = projection(self.source2_M,self.validatingTest)
			
			
			
		
		

		
		stop()
		historryOffset1 = findHistoryOffset(self.allBelief ['1'],self.allBelief ['t'], self.M_proj['t'], self.M_proj['1'], self.source1_M,17,30)
		historryOffset2 = findHistoryOffset(self.allBelief ['2'],self.allBelief ['t'], self.M_proj['t'], self.M_proj['2'], self.source2_M,30,30)
		#print self.allBelief ['t']
		print historryOffset1,	historryOffset2
		if historryOffset1[1]>historryOffset2[1]: 
			bestSource =0
		
			bel = trajectoryBeliefUpdatePSR(self.source1_M,historryOffset1[0],self.allBelief ['1'])
		
		else: 
			bestSource = 1
			bel = trajectoryBeliefUpdatePSR(self.source2_M,historryOffset2[0],self.allBelief ['2'])
		
		action = selectAction(self.QN[bestSource],bel)
		print action
		
		
		
		#action = random.choice(['South','South','South','East'])
		self.actionHistory.append(action)
		self.doAction(state,action)
	
		return action
		
	def update(self, state, action, nextState, reward):

		
		actionObs = ((action,(nextState.getObservations(),reward)),)
		
		self.allBelief['t'] = beliefUpdatePSR(self.target_M,actionObs,self.allBelief['t'])
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
