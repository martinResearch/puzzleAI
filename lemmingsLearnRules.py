import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.feature_extraction
from lemmings import *

def generateGames(height,width,nbFrames,nbLemmings=1,startLeftCorner=True,nbGames=30):
    games=[]
    for idSimu in range(nbGames):
        games.append(randomGame(height,width,startLeftCorner=startLeftCorner))  
    return games

def signedIntToStr(i):
    if i>0:
        return '+%d'%i
    if i==0:
        return ''
    if i<0:
        return '%d'%i
    

def generateTrainingData(games):
    # generate a random scene , run the simulation for some iterations
    # learn localized rules (like physical rules) that are localized in space and time with the immediate neighboors
    # for each pixel predict its value from the neighboring ones at frame-1 (assuming we observe all variables like the orientation) 
    # this assume full state observable ? 
    inputs=[]
    outputs=[]
   
    
    for idSimu in range(len(games)):
        
   
        game=games[idSimu]
        lemmingsPositions,lemmingsDirections=game.simulate(game.obstaclesMap)
        lemmingsMaps=np.zeros((game.nbFrames,game.height,game.width,2))
        drawLemmingsInMap(lemmingsMaps,lemmingsPositions,lemmingsDirections)
        #displayLemmingsAnimation(lemmingsMaps,game.obstaclesMap,game.targetsMap)
        
    
        obstaclePatches=sklearn.feature_extraction.image.extract_patches_2d(game.obstaclesMap, [3,3])
        
        for idFrame in range(1,game.nbFrames):
            lemmingsPatches=[]
            for d in range(2):
                lemmingsPatches.append(sklearn.feature_extraction.image.extract_patches_2d(lemmingsMaps[idFrame-1,:,:,d], [3,3]))
            inputs.append(np.stack((obstaclePatches[:,None,:,:],lemmingsPatches[0][:,None,:,:],lemmingsPatches[1][:,None,:,:]), axis=1))
            outputs.append(lemmingsMaps[idFrame,1:-1,1:-1,:].reshape(-1,2))
    inputNames=[]       
    for i in range(3):
        for j in range(3):
            inputNames.append('obstacle[iFrame-1,i%s,j%s]'%(signedIntToStr(i-1),signedIntToStr(j-1)))
    for i in range(3):
        for j in range(3):            
            inputNames.append('lemmings[iFrame-1,i%s,j%s,0]'%(signedIntToStr(i-1),signedIntToStr(j-1)))
    for i in range(3):
        for j in range(3):
            inputNames.append('lemmings[iFrame-1,i%s,j%s,1]'%(signedIntToStr(i-1),signedIntToStr(j-1)))
    
    outputNames=['lemmings[iFrame,i,j,0]','lemmings[iFrame,i,j,1]']
    inputs=np.vstack(inputs)
    inputs=inputs.reshape(inputs.shape[0],-1)  
    outputs=np.vstack(outputs)
    outputs=outputs.reshape(outputs.shape[0],-1) 
    #def extractPatches(table,contextsize):
    return inputs,outputs,inputNames,outputNames

def augmentFeaturesWithProducts(inputs):
    nbExamples=inputs.shape[0]
    order2inputs=(inputs[:,None,:]*inputs[:,:,None]).reshape(nbExamples,-1)
    augmentedinput=np.hstack((inputs,order2inputs))  
    #augmentedinput=inputs
    return augmentedinput






def learnModel(inputs,outputs,regressionType,useFeatureAugmentation):
    
    #reg.fit(inputs, outputs[:,0]) 
    #print 'mean error using order 1 features %s'%np.mean((reg.predict (inputs) -outputs[:,0])**2)
    if useFeatureAugmentation:
        inputs = augmentFeaturesWithProducts(inputs)
    regs=[]
    for d in range(2):
        if regressionType=='LogisticRegression':
            reg = linear_model.LogisticRegression()
        elif  regressionType=='DecisionTree':
            reg = DecisionTreeClassifier(random_state=0)
        elif  regressionType=='MLP':    
            reg = MLPClassifier()
        else:
            raise "unkown type"
        
        reg.fit(inputs, outputs[:,d]) 
        regs.append(reg)
        print 'mean error  %s'%np.mean((reg.predict (inputs) -outputs[:,0])**2)
    return regs

# using order " feature is too big in memory    
#order3inputs=(order2inputs[:,None,:]*inputs[:,:,None]).reshape(nbExamples,-1)
#augmentedinput2=np.hstack((inputs,order2inputs,order3inputs))    
#reg.fit(augmentedinput2, outputs[:,0]) 
#print 'mean error using order 1 and 2 features %s'%np.mean((reg.predict (augmentedinput2) -outputs[:,0])**2)


def simulateTrainedModel(regs,lemmingsMaps,obstaclesMap,targetsMap,nbFrames,useRounding,useFeatureAugmentation,normalizeEachFrame=False):
    obstaclePatches=sklearn.feature_extraction.image.extract_patches_2d(obstaclesMap, [3,3])
   
    height,width=targetsMap.shape
    for idFrame in range(1,nbFrames):
        lemmingsPatches=[]
        for d in range(2):
            lemmingsPatches.append(sklearn.feature_extraction.image.extract_patches_2d(lemmingsMaps[idFrame-1,:,:,d], [3,3]))
       
        inputs=np.stack((obstaclePatches[:,None,:,:],lemmingsPatches[0][:,None,:,:],lemmingsPatches[1][:,None,:,:]), axis=1)
        inputs=inputs.reshape(inputs.shape[0],-1) 
        if useFeatureAugmentation:
            inputs   =augmentFeaturesWithProducts(inputs)
        lemmingsMaps[idFrame].fill(0)
        for d in range(2):
            
            #lemmingsMaps[idFrame,1:-1,1:-1,d]=(regs[d].predict (augmentedinput).reshape(height-2,width-2))
            lemmingsMaps[idFrame,1:-1,1:-1,d]=(regs[d].predict_proba (inputs)[:,1].reshape(height-2,width-2))
            
        
        lemmingsMaps[idFrame]=np.maximum(lemmingsMaps[idFrame], 0)
        lemmingsMaps[idFrame]=np.minimum(lemmingsMaps[idFrame], 1)
        if normalizeEachFrame and np.sum(lemmingsMaps[idFrame])>0:
            lemmingsMaps[idFrame]=lemmingsMaps[idFrame]/np.sum(lemmingsMaps[idFrame])
        
        if useRounding:
            lemmingsMaps[idFrame]=np.round(lemmingsMaps[idFrame])  
            
            
def displayDecisionTree(estimator,inputNames=None,outputName=None):
                # from http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
                
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    if inputNames is None:
        inputNames=['X[:, %d]'%d for d in   range(np.max(feature)+1)]
    if outputName is None:
        outputName='ouput'

    threshold = estimator.tree_.threshold
    value=estimator.tree_.value
    
    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
    
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
           
    
    print("The binary tree structure to predict %s has %s nodes and has "
          "the following tree structure:"
          % (outputName,n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            v= value[i][0][0]<value[i][0][1]
            print("%snode=%s leaf node with value %d." % (node_depth[i] * "\t", i,v))
        else:
            print("%snode=%s test node: go to node %s if %s <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     inputNames[feature[i]],
                     threshold[i],
                     children_right[i],
                     ))
                   


if __name__ == "__main__":

    #game=randomGame(height=10, width=10,seed=15,nbMaxBlocks=4,startLeftCorner=True) 
    game=readGameFile('level.txt')
    
    lemmingsMaps=np.zeros((game.nbFrames,game.height,game.width,2))
    
   
    lemmingsMaps[0]=game.lemmingsMapsInit
    nbLemmings=1
    games=generateGames(5,5,10,nbLemmings=1,nbGames=30,startLeftCorner=False)
    
    inputs, outputs,inputNames,outputNames=generateTrainingData(games)
    
    useFeatureAugmentation=True
    regs=learnModel(inputs, outputs,regressionType='LogisticRegression',useFeatureAugmentation=True)
    
    simulateTrainedModel(regs,lemmingsMaps,game.obstaclesMap,game.targetsMap,game.nbFrames,useFeatureAugmentation=useFeatureAugmentation,useRounding=True)    
    displayLemmingsAnimation(lemmingsMaps,game.obstaclesMap,game.targetsMap,'images/learnedSimulationLogisticRegressionWithRounding.gif') 

    simulateTrainedModel(regs,lemmingsMaps,game.obstaclesMap,game.targetsMap,game.nbFrames,useFeatureAugmentation=useFeatureAugmentation,useRounding=False)    
    displayLemmingsAnimation(lemmingsMaps,game.obstaclesMap,game.targetsMap,'images/learnedSimulationLogisticRegressionWithoutRounding.gif')  
    displayLemmingsAnimation(lemmingsMaps*30,game.obstaclesMap,game.targetsMap,'images/learnedSimulationLogisticRegressionWithoutRoundingHigherContrast.gif') 
    
    useFeatureAugmentation=False
    regs=learnModel(inputs, outputs,regressionType='MLP',useFeatureAugmentation=useFeatureAugmentation)
    
    simulateTrainedModel(regs,lemmingsMaps,game.obstaclesMap,game.targetsMap,game.nbFrames,useFeatureAugmentation=useFeatureAugmentation,useRounding=True)    
    displayLemmingsAnimation(lemmingsMaps,game.obstaclesMap,game.targetsMap,'images/learnedSimulationMLPWithRounding.gif') 

    simulateTrainedModel(regs,lemmingsMaps,game.obstaclesMap,game.targetsMap,game.nbFrames,useFeatureAugmentation=useFeatureAugmentation,useRounding=False)    
    displayLemmingsAnimation(lemmingsMaps,game.obstaclesMap,game.targetsMap,'images/learnedSimulationMLPWithoutRounding.gif')    
    displayLemmingsAnimation(lemmingsMaps*30,game.obstaclesMap,game.targetsMap,'images/learnedSimulationMLPWithoutRoundingHigherContrast.gif') 
    
    
    useFeatureAugmentation=False
    regs=learnModel(inputs, outputs,regressionType='DecisionTree',useFeatureAugmentation=useFeatureAugmentation)
    for idOutput in range(2):
        displayDecisionTree(regs[idOutput],inputNames,outputNames[idOutput])
    
    simulateTrainedModel(regs,lemmingsMaps,game.obstaclesMap,game.targetsMap,game.nbFrames,useFeatureAugmentation=useFeatureAugmentation,useRounding=True)    
    displayLemmingsAnimation(lemmingsMaps,game.obstaclesMap,game.targetsMap,'images/learnedSimulationDecisionTreeWithRounding.gif') 

    simulateTrainedModel(regs,lemmingsMaps,game.obstaclesMap,game.targetsMap,game.nbFrames,useFeatureAugmentation=useFeatureAugmentation,useRounding=False)    
    displayLemmingsAnimation(lemmingsMaps,game.obstaclesMap,game.targetsMap,'images/learnedSimulationDecisionTreeWithoutRounding.gif')    
    displayLemmingsAnimation(lemmingsMaps*30,game.obstaclesMap,game.targetsMap,'images/learnedSimulationDecisionTreeWithoutRoundingHigherContrast.gif') 