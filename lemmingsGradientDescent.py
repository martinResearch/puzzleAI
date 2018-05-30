from lemmings import *

def lemmingsSimulationStateSpace(lemmingsMaps,obstaclesMap,targetsMap,nbFrames):
    for iFrame in range(1,nbFrames):
        for d in range(2):
            s=d*2-1
            lemmingsMaps[iFrame,:,:,d]+=np.roll(lemmingsMaps[iFrame-1,:,:,d],1,axis=0)*(1-obstaclesMap)
            lemmingsMaps[iFrame,:,:,d]+=np.roll(lemmingsMaps[iFrame-1,:,:,d],s,axis=1)*(1-obstaclesMap)*np.roll(np.roll(obstaclesMap,s,axis=1),-1,axis=0)
            lemmingsMaps[iFrame,:,:,d]+=lemmingsMaps[iFrame-1,:,:,1-d]*np.roll(obstaclesMap,s,axis=1)*np.roll(obstaclesMap,-1,axis=0)
            
            
      

def lemmingsSimulationStateSpace_B(lemmingsMaps,obstaclesMap,targetsMap,lemmingsMaps_B,obstaclesMap_B):
    nbFrames=lemmingsMaps.shape[0]
    for iFrame in range(nbFrames-1,0,-1):  
        for d in range(2):
            s=d*2-1
            #lemmingsMaps[iFrame,:,:,d]+=np.roll(lemmingsMaps[iFrame-1,:,:,d],1,axis=0)*(1-obstaclesMap)
            lemmingsMaps_B[iFrame-1,:,:,d]+=np.roll(lemmingsMaps_B[iFrame,:,:,d]*(1-obstaclesMap),-1,axis=0)
            obstaclesMap_B+=-lemmingsMaps_B[iFrame,:,:,d] *np.roll(lemmingsMaps[iFrame-1,:,:,d],1,axis=0)       
            #lemmingsMaps[iFrame,:,:,d]+=np.roll(lemmingsMaps[iFrame-1,:,:,d],s,axis=1)*(1-obstaclesMap)*np.roll(np.roll(obstaclesMap,s,axis=1),-1,axis=0)
            lemmingsMaps_B[iFrame-1,:,:,d]+=np.roll(lemmingsMaps_B[iFrame,:,:,d]*(1-obstaclesMap)*np.roll(np.roll(obstaclesMap,s,axis=1),-1,axis=0),-s,axis=1)
            obstaclesMap_B+=-lemmingsMaps_B[iFrame,:,:,d]*np.roll(lemmingsMaps[iFrame-1,:,:,d],s,axis=1)*np.roll(np.roll(obstaclesMap,s,axis=1),-1,axis=0)
            obstaclesMap_B+=np.roll(np.roll(lemmingsMaps_B[iFrame,:,:,d]*np.roll(lemmingsMaps[iFrame-1,:,:,d],s,axis=1)*(1-obstaclesMap),-s,axis=1),1,axis=0)
            #lemmingsMaps[iFrame,:,:,d]+=lemmingsMaps[iFrame-1,:,:,1-d]*np.roll(obstaclesMap,s,axis=1)*np.roll(obstaclesMap,-1,axis=0)
            lemmingsMaps_B[iFrame-1,:,:,1-d]+=lemmingsMaps_B[iFrame,:,:,d]*(np.roll(obstaclesMap,s,axis=1))*np.roll(obstaclesMap,-1,axis=0)
            obstaclesMap_B+=np.roll(lemmingsMaps_B[iFrame,:,:,d]*lemmingsMaps[iFrame-1,:,:,1-d]*np.roll(obstaclesMap,-1,axis=0),-s,axis=1)
            obstaclesMap_B+=np.roll(lemmingsMaps_B[iFrame,:,:,d]*lemmingsMaps[iFrame-1,:,:,1-d]*np.roll(obstaclesMap,s,axis=1),+1,axis=0)
            lemmingsMaps_B[iFrame-1,:,:,d]+=lemmingsMaps_B[iFrame,:,:,d]*targetsMap
            

            
# we code the function that we will provide to a gradient descent based solver from scipy            
global iter 
iter = 0   

def funcToOptimize(x,targetsMap,lemmingsMapsInit,nbFrames,obstaclesMaps,alpha):
    height,width=targetsMap.shape
    lemmingsMaps=np.zeros((nbFrames,height,width,2))
    lemmingsMaps[0]=lemmingsMapsInit
    obstaclesMap=x.reshape(targetsMap.shape)
    
    global iter
    iter+=1
    obstaclesMaps[iter,:,:]=obstaclesMap
    
    lemmingsSimulationStateSpace(lemmingsMaps,obstaclesMap,targetsMap,nbFrames)    
    score=np.sum(np.sum(lemmingsMaps[-1],axis=2)*targetsMap)-alpha*np.sum(obstaclesMap)
    obstaclesMap_B=np.zeros(targetsMap.shape)-alpha
    
    
    lemmingsMaps_B=np.zeros((nbFrames,height,width,2))
    lemmingsMaps_B[nbFrames-1]=targetsMap[:,:,None] 
    lemmingsSimulationStateSpace_B(lemmingsMaps,obstaclesMap,targetsMap,lemmingsMaps_B,obstaclesMap_B)
    obstaclesMap_B=obstaclesMap_B*obstaclesMapMask
        
    print (score)
    return   -score   ,-obstaclesMap_B.flatten()    




if __name__ == "__main__":
    # setting the number of lemmings and the size of the map
    
    #game=randomGame(height=10, width=10,seed=15,nbMaxBlocks=4,startLeftCorner=True) 
    game=readGameFile('level.txt')
    
    lemmingsMaps=np.zeros((game.nbFrames,game.height,game.width,2))
    

    obstaclesMap=game.obstaclesMap.copy()   
    
    # simulating the lemmings
    lemmingsPositions,lemmingsDirections=game.simulate(obstaclesMap)
    
    # in order to visualize the simulation as an animation we draw the lemmings in an array and then we use matplotlib
    gifZoom=30    
    # visualizing the animation
    
    lemmingsMaps=np.zeros((game.nbFrames,game.height,game.width,2))
    drawLemmingsInMap(lemmingsMaps,lemmingsPositions,lemmingsDirections)
    displayLemmingsAnimation(lemmingsMaps,obstaclesMap,game.targetsMap,'images/firstAnimation.gif')
    
    
    
    #can use a ridge regression or lasso, can augment with products of pairs of binary local variables or even three local variables()    
    
    # this is bit like learning markov field higher potential terms on 3D grid ? 
    # if we do not observe the speed we need to learn a hidden random markov field ? 
    # can the convex input neural netwok learn with hidden states ? or do we need loops i nthe graph ? 
    # as there is time , we can learn conditional probabilities
    # can we use these learned rules instead of the orginal hard coded ones to solve the problem/puzzle ? 
    
    
    
    # score is define as the number of lemmings that made it to the target
    
    score=finalScore(lemmingsMaps,game.targetsMap)
    
    # we can discretize the state space and represent the state of the lemmings through an occupency map

    lemmingsMaps2=np.zeros((game.nbFrames,game.height,game.width,2))
    lemmingsMaps2[0]=game.lemmingsMapsInit
    lemmingsSimulationStateSpace(lemmingsMaps2,obstaclesMap,game.targetsMap,game.nbFrames)
    
    assert(np.all(lemmingsMaps==lemmingsMaps2))
    
    
    # we aim at finding the obstacle map that will maximize the score
        # using this full state discretization normulation we can relax the constraint that the obsacleMap take binary values and use obstacle maps with values beeing a float number between 0 and 1
    lemmingsMaps=np.zeros((game.nbFrames,game.height,game.width,2))
    obstaclesMap=np.maximum(game.obstaclesMap,0.3)
    obstaclesMap[np.sum(game.lemmingsMapsInit,axis=2)>0]=0
    obstaclesMap[:,0]=1
    obstaclesMap[:,-1]=1
    obstaclesMap[-1,:]=1
    
    lemmingsMaps[0]=game.lemmingsMapsInit*10
    lemmingsSimulationStateSpace(lemmingsMaps,obstaclesMap,game.targetsMap,game.nbFrames)
    displayLemmingsAnimation(lemmingsMaps,obstaclesMap,game.targetsMap,'images/relaxedSimulation.gif')
    
    # thanks to the relaxation from integer to float for the obsacleMap in our simulation we can compute the derivative of the score with respect to the obsacleMap and do a gradient ascent
    # this can be done efficenlty by implementing the adjoin of the function lemmingsSimulationStateSpace, also known as back propagation in the neural network context
        
    # we define a mask for the region of the obstacles map we are allowed to change.
    
    obstaclesMapMask=1-game.obstaclesMap
    obstaclesMapMask[np.sum(game.lemmingsMapsInit,axis=2)>0]=0
    obstaclesMapMask[:,0]=0
    obstaclesMapMask[:,-1]=0
    obstaclesMapMask[-1,:]=0   
    
    
    bounds=np.zeros((obstaclesMap.size,2))
    bounds[:,1]=1
    display=False
    nbIter=200
    obstaclesMaps=np.ones((nbIter+10,game.height,game.width))
    obstaclesMaps[iter,:,:]=obstaclesMap
    alpha=1e-2
    result=scipy.optimize.fmin_l_bfgs_b(funcToOptimize, obstaclesMap.flatten(),args=[game.targetsMap,game.lemmingsMapsInit,game.nbFrames,obstaclesMaps,alpha],bounds=bounds,maxfun =np.floor(nbIter*0.8),factr =10)
    obstaclesMapOpt=result[0].reshape(obstaclesMap.shape)
    displayMovie(obstaclesMaps[:iter+1,:,:,None]*obstacleColor,'images/obstaclesOptimization.gif')
    
    lemmingsMaps=np.zeros((game.nbFrames,game.height,game.width,2))
    lemmingsMaps[0]=game.lemmingsMapsInit
    lemmingsSimulationStateSpace(lemmingsMaps,obstaclesMapOpt,game.targetsMap,game.nbFrames)
    displayLemmingsAnimation(lemmingsMaps,obstaclesMapOpt,game.targetsMap,'images/solutionSimulation1.gif') 
    
    