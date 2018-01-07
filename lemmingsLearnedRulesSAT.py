from __future__ import print_function
import sys
from ortools.constraint_solver import pywrapcp
from lemmings import *
from lemmingsLearnRules import *
from lemmingsSAT import * 
import lemmingsMILP
import copy
    
   
   
def decisionTreeToCNF(estimator):
                # from http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    
    # convert the decision tree into two Conjontive Normal Forms that are negation of each other ? 
    # CNF[0](input) => result
    # CNF[1](input) => not(result)
   
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    value=estimator.tree_.value

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
   
    
    rule=[]
    CNF=[[] for i in range(2)]
   
    
    def addrules(node_id):
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            rule.append((feature[node_id],1))
            addrules(children_left[node_id])
            rule.pop()
            rule.append((feature[node_id],0))
            addrules(children_right[node_id])
            rule.pop()
        else:
            if value[node_id][0][0]>value[node_id][0][1]:
                CNF[0].append(rule[:])
            else:
                CNF[1].append(rule[:])
            
            
    addrules(0)


    return CNF     


def checkCNF(allCNF,solution):
    nbFailed=0                
    for iclause,NF in enumerate(allCNF):
        clauseCheck=0
        for e in NF:
            if e[1]==1:
                clauseCheck=clauseCheck or solution[e[0]]
            else:
                clauseCheck=clauseCheck or (1-solution[e[0]])
        if not(clauseCheck) :      
            nbFailed+=1   
    return nbFailed
    

if __name__ == "__main__":
    
    
    
    #game=randomGame(height=10, width=10,seed=15,nbMaxBlocks=4,startLeftCorner=True) 
    game=readGameFile('level.txt')
    
    nbFrames=game.nbFrames
    height=game.height
    width=game.width
    solutionsSAT=solveGame(game)
  
    solutionSAT=solutionsSAT[0] # get a solution using the SAT solver with hand coded constraints , this will be used to check the validity on the learned constraints    
    
    #solutionSAT=lemmingsMILP.solveGame(game)

    displayLemmingsAnimation(solutionSAT['lemmingsMap'],solutionSAT['obstaclesMap'],game.targetsMap,gifZoom=30)    
    
    
    nbGamestraining=50
    
    games=generateGames(5,5,10,nbLemmings=1,nbGames=30,startLeftCorner=False)
    
    
    inputs, outputs,inputNames,outputNames=generateTrainingData(games)
    useFeatureAugmentation=False
    regs=learnModel(inputs, outputs,regressionType='DecisionTree',useFeatureAugmentation=False)
    for idOutput in range(2):
        displayDecisionTree(regs[idOutput],inputNames,outputNames[idOutput])
    lemmingsMaps=np.zeros((nbFrames,height,width,2))
    lemmingsMaps[0]=game.lemmingsMapsInit
    
    simulateTrainedModel(regs,lemmingsMaps,game.obstaclesMap,game.targetsMap,game.nbFrames,useFeatureAugmentation=useFeatureAugmentation,useRounding=True) 
      
    displayLemmingsAnimation(lemmingsMaps,game.obstaclesMap,game.targetsMap,'images/learnedSimulationWithRounding.gif') 
    
   
    solver = pywrapcp.Solver("lemmings")
    
  
    localCNF=[decisionTreeToCNF(regs[0]),decisionTreeToCNF(regs[1])]
    
    
   
    # printing the local CNF as strings
    for d in range(2):  # loop over the directions of the lemming
        print ('\n\n')
        clauses=[]
        for b in range(2): # bloolean value of the ouput
            for localNF in localCNF[d][b]:
                clauseStrs=[]
                for e in localNF:
                    if e[1]==1:
                        clauseStrs.append(inputNames[e[0]])
                    else:
                        clauseStrs.append('not('+inputNames[e[0]])  
                if b==1:
                    clauseStrs.append(outputNames[d])
                else:
                    clauseStrs.append('not('+outputNames[d]+')')
                clauses.append('('+' or '.join(clauseStrs)+')')
        print (' and \n'.join(clauses))
            
                
    
    # checking clause on the training data
    nbFailed=0 
    for inp,out in zip(inputs,outputs):    
        for d in range(2):  # loop over the direction of the lemming
            for b in range(2): # bloolean vqlue of the ouput
                for localNF in localCNF[d][b]:
                    NF=[]
                    for r in localNF:
                        NF.append((inp[r[0]],r[1]))
                    NF.append((out[d],b))
                    
                    clauseEval=0
                    for e in NF:
                        if e[1]==1:
                            clauseEval=clauseEval or e[0]
                        else:
                            clauseEval=clauseEval or (1-e[0])                    
                    if not(clauseEval):
                        nbFailed+=1
                        
                    
            
    inputs=[]
    outputs=[]
    nbVars=0
    
    obstaclesMapIds=(np.arange(height*width)+nbVars).reshape(height,width)
    nbVars=nbVars+obstaclesMapIds.size

    
    lemmingsMapsIds=(np.arange(nbFrames*height*width*2)+nbVars).reshape(nbFrames,height,width,2)
    nbVars=nbVars+lemmingsMapsIds.size    
    
    obstaclePatchesIds=sklearn.feature_extraction.image.extract_patches_2d(obstaclesMapIds, [3,3])
    
    allVars=[]
    for idVar in range(nbVars):
        allVars.append(solver.BoolVar( 'var%d'%idVar))
   
   
    allSolutionSATAllVars=[]
    for idSol in range(len(solutionsSAT)):
        solutionSATAllVars=np.zeros((nbVars))
        solutionSATAllVars[lemmingsMapsIds]=solutionSAT['lemmingsMap']
        solutionSATAllVars[obstaclesMapIds]=solutionSAT['obstaclesMap']
        allSolutionSATAllVars.append(solutionSATAllVars)

   
    iFrame=0                     
   
           
    for i in range(height):
        for j in range(width):
            for d in range(2):    
                solver.Add(allVars[lemmingsMapsIds[iFrame,i,j,d]] ==   int(game.lemmingsMapsInit[i,j,d]) ) 
                for solutionSATAllVars in allSolutionSATAllVars:
                    assert(solutionSATAllVars[lemmingsMapsIds[iFrame,i,j,d]] ==   int(game.lemmingsMapsInit[i,j,d]))
                
                
                # the learned predictor cqnnot predict the lemmings map on the boundary as we do not use image wrapping in the patch extraction
                # we cheat a bit by forning the lemmings to be zero on the boundraries
    for iFrame in range(0,nbFrames):
        for i in range(height):
            for d in range(2):    
                solver.Add(allVars[lemmingsMapsIds[iFrame,i,0,d]] == 0 ) 
                solver.Add(allVars[lemmingsMapsIds[iFrame,i,width-1,d]] == 0 ) 
                for solutionSATAllVars in allSolutionSATAllVars:
                    assert(solutionSATAllVars[lemmingsMapsIds[iFrame,i,0,d]]  ==  0)
                    assert(solutionSATAllVars[lemmingsMapsIds[iFrame,i,width-1,d]] ==   0)
        for j in range(width):
            for d in range(2):    
                solver.Add(allVars[lemmingsMapsIds[iFrame,0,j,d]] == 0 ) 
                solver.Add(allVars[lemmingsMapsIds[iFrame,height-1,j,d]] == 0 )   
                for solutionSATAllVars in allSolutionSATAllVars:
                    assert(solutionSATAllVars[lemmingsMapsIds[iFrame,0,j,d]] ==  0)
                    assert(solutionSATAllVars[lemmingsMapsIds[iFrame,height-1,j,d]]  ==   0)                

    for i in range(height):
        for j in range(width):
            solver.Add(allVars[obstaclesMapIds[i,j]] > game.obstaclesMap[i,j]-1) 
            for solutionSATAllVars in allSolutionSATAllVars:
                assert(solutionSATAllVars[obstaclesMapIds[i,j]] > game.obstaclesMap[i,j]-1)
   
   
   
    for idFrame in range(1,nbFrames):
        lemmingsPatchesIds=[]
        for d in range(2):
            lemmingsPatchesIds.append(sklearn.feature_extraction.image.extract_patches_2d(lemmingsMapsIds[idFrame-1,:,:,d], [3,3]))
        inputs.append(np.stack((obstaclePatchesIds[:,None,:,:],lemmingsPatchesIds[0][:,None,:,:],lemmingsPatchesIds[1][:,None,:,:]), axis=1))
        outputs.append(lemmingsMapsIds[idFrame,1:-1,1:-1,:].reshape(-1,2))
       
    inputs=np.vstack(inputs)
    inputs=inputs.reshape(inputs.shape[0],-1)  
    outputs=np.vstack(outputs)
    outputs=outputs.reshape(outputs.shape[0],-1)        
        
    allCNF=[]
    for inp,out in zip(inputs,outputs):    
        for d in range(2):  # loop over the direction of the lemming
            for b in range(2): # bloolean vqlue of the ouput
                for localNF in localCNF[d][b]:
                    NF=[]
                    for r in localNF:
                        NF.append((inp[r[0]],r[1]))
                    NF.append((out[d],b))
                    allCNF.append(NF)
                    
    for solutionSATAllVars in allSolutionSATAllVars:
                        
        nbFailed=0                
        for iclause,NF in enumerate(allCNF):
            clause=[]
            clauseCheck=0
            for e in NF:
                if e[1]==1:
                    clause.append( allVars[e[0]])
                    clauseCheck=clauseCheck + solutionSATAllVars[e[0]]
                else:
                    clause.append(1-allVars[e[0]])
                    clauseCheck=clauseCheck + (1-solutionSATAllVars[e[0]])
            if not(clauseCheck>0) :      
                nbFailed+=1   
            solver.Add(solver.Sum(clause)>0)    # works also with solver.Add(solver.Max(clause)==1)        
          
        if nbFailed>0:
            print('the solution provided violates %d constraints'%nbFailed)
            
        nbFailed=checkCNF(allCNF,solutionSATAllVars)
                        
    # adding target constraint     
    solver.Add(solver.Sum([allVars[lemmingsMapsIds[nbFrames-1,i,j,d]]*game.targetsMap[i,j] for i in range(height) for j in range(width) for d in range(2)] )==game.nbLemmings) 
    for solutionSATAllVars in allSolutionSATAllVars:
        assert(np.array([solutionSATAllVars[lemmingsMapsIds[nbFrames-1,i,j,d]]*game.targetsMap[i,j] for i in range(height) for j in range(width) for d in range(2)] ).sum()==game.nbLemmings)
    # adding constraint that only 3 bricks should be added 
    nbMaxBlocks=game.nbMaxBlocks
    solver.Add(solver.Sum([allVars[obstaclesMapIds[i,j]] for i in range(height) for j in range(width)] )<(nbMaxBlocks+np.sum(game.obstaclesMap)+1)) 
    for solutionSATAllVars in allSolutionSATAllVars:
        assert(np.array([solutionSATAllVars[obstaclesMapIds[i,j]] for i in range(height) for j in range(width)] ).sum()<nbMaxBlocks+np.sum(game.obstaclesMap+1))
    db = solver.Phase(allVars, solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE)
    solver.Solve(db)

    count = 0

    solutions=[]
    timestart=time.time()
    while solver.NextSolution():
        count += 1
        print("Solution" , count, '\n')
        
        solution=np.zeros((len(allVars)),dtype=np.bool)
        for i in range(len(allVars)):
            solution[i]=allVars[i].Value()
        
        
        nbFailed=checkCNF(allCNF,solution)
        assert(nbFailed==0)
        
        lemmingsMapsNp=np.zeros((nbFrames,height,width,2))
        obstaclesMapNp=np.zeros((height,width))
        targetsMapNp=np.zeros((height,width))
       
        
        for iFrame in range(nbFrames):
            for i in range(height):
                for j in range(width):
                    for d in range(2):
                        lemmingsMapsNp[iFrame,i, j,d]=allVars[lemmingsMapsIds[iFrame,i, j,d]].Value()
                        

        for i in range(height):
            for j in range(width):
                    obstaclesMapNp[i, j]=allVars[obstaclesMapIds[i, j]].Value()   
                    
        solvedGame=copy.deepcopy(game)
       
        lemmingsMap=solvedGame.simulateMap(obstaclesMapNp)
        assert(np.all(lemmingsMap==lemmingsMapsNp))
                    
                    
        #checkSolution(obstaclesMapNp,game.targetsMap,lemmingsMapsNp,auxVarsNp)
        #assert(np.array([solution[lemmingsMapsIds[nbFrames-1,i,j,d]]*solution[targetsMapIds[i,j]] for i in range(height) for j in range(width)] ).sum()==game.nbLemmings)
        print (np.sum(np.sum(lemmingsMapsNp[-1],axis=2)*game.targetsMap))
        
        print ('hash lemmingsMapsNp=%s'%hash(lemmingsMapsNp.tostring()))
        print ('hash obstaclesMapNp=%s'%hash(obstaclesMapNp.tostring()))
        print ('hash targetsMapNp=%s'%hash(targetsMapNp.tostring()))
        print ('hash solution=%s'%hash(solution.tostring()))
        solutions.append( {'obstaclesMap':obstaclesMapNp,'lemmingsMap':lemmingsMapsNp})
        
        print ('finding %d solutions took %d seconds'%(len(solutions),time.time()-timestart))
    print ('finding all solutions took %d seconds'%(time.time()-timestart))
    if count==0:
        print('could not find any solution') 
        
    assert(count==len(allSolutionSATAllVars))# checking we found as many solution wit the the learnt rules as with the hand coded rules
    
    for count,solution in enumerate(solutions):
        gifName='images/SolutionLearnedRulesSAT%d.gif'%count               
        displayLemmingsAnimation(solution['lemmingsMap'],solution['obstaclesMap'],game.targetsMap,gifName=gifName,gifZoom=30)    

        
