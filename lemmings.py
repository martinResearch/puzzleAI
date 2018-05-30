# we will use numpyn matplotlib, and some other libraries

import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import animation
import scipy.signal
from scipy.optimize import fmin_l_bfgs_b
import imageio
from  lemmingsGraphics import displayLemmingsAnimationPyGame

obstacleColor=np.array([0.8,0.8,0.8])
lemmingsColor =  np.array([0,1,1])                
targetColor = np.array([0,1,0 ])


def zoomImage(image,factor):
    return np.dstack([scipy.ndimage.zoom(image[:,:,idChannel], factor,order=0) for idChannel in range(3)])


def randomObstacles(targetMap,lemmingsPositionsInit,height,width):
    obstaclesMap=np.random.rand(height,width)>0.7 
    obstaclesMap[:,0]=1
    obstaclesMap[0,:]=1
    obstaclesMap[:,-1]=1
    obstaclesMap[-1,:]=1
    obstaclesMap[lemmingsPositionsInit[:,0],lemmingsPositionsInit[:,1]]=0
    obstaclesMap=np.minimum(1-targetMap, obstaclesMap)
    return obstaclesMap


def drawLemmingsInMap(lemmingsMaps,lemmingsPositions,lemmingsDirections):
    nbFrames=lemmingsMaps.shape[0]
    nbLemmings=lemmingsPositions.shape[1] 
    for iFrame in range(nbFrames):
        for i in range(nbLemmings):
            lemmingsMaps[iFrame,lemmingsPositions[iFrame,i,0],lemmingsPositions[iFrame,i,1],int(0.5*(lemmingsDirections[iFrame,i])+1)]+=1  
  
def displayMovie(movieFrames,gifName,gifZoom=30):
    if not gifName is None:
        frames=[]
        for idFrame in range(len(movieFrames)):
            #frames.append(cv2.resize(movieFrames[idFrame],None,fx=gifZoom, fy=gifZoom, interpolation = cv2.INTER_NEAREST))
            frames.append(zoomImage(movieFrames[idFrame],gifZoom))
        imageio.mimsave(gifName, frames)        
    fig = plt.figure()
    iFrame=0
    im=plt.imshow(movieFrames[0],interpolation="nearest")
    
    def init():
        return im,
    
    def animate(iFrame):
        im.set_data(movieFrames[iFrame])
        return im,

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=movieFrames.shape[0], interval=100, blit=True)
    plt.show() 
 
 
   

def displayLemmingsAnimation(lemmingsMaps,obstaclesMap,targetsMap,gifName=None,gifZoom=30):
    lemmingsMapsSum=np.sum(lemmingsMaps,axis=3)
    movieFrames=np.minimum(targetsMap[None,:,:,None]*targetColor+obstaclesMap[None,:,:,None]*obstacleColor+lemmingsMapsSum[:,:,:,None]*lemmingsColor,1)
    displayMovie(movieFrames,gifName=gifName,gifZoom=gifZoom) 
    




def finalScore(lemmingsMaps,targetsMap):
    return np.sum(np.sum(lemmingsMaps[-1],axis=2)*targetsMap)


def randomGame(height,width,nbFrames=None,nbLemmings=1,seed=None,nbMaxBlocks=3,startLeftCorner=True):
    if nbFrames is None:
        nbFrames=height*2
 
    if not seed is None:
        np.random.seed(seed)   
    
    # setting up the target
    nbTarget=1
    target=np.array([[height-2,width-2]],dtype=np.int32)

        
    # initializing lemmings positions randomly    
    if startLeftCorner:
        lemmingsPositionsInit=np.array([[1,1]])
        np.random.rand(nbLemmings,2)
    else:
        lemmingsPositionsInit=np.floor(np.random.rand(nbLemmings,2)*np.array([1,width-3])).astype(np.int32)+np.array([1,1])
    lemmingsDirectionsInit=np.round(np.random.rand(nbLemmings)).astype(np.int32)*2-1
    targetsMap=np.zeros((height,width),dtype=np.bool)
    for i in range(nbTarget):
        targetsMap[target[i,0],target[i,1]]+=1      
    # setting up the obstacles
    obstaclesMap=randomObstacles(targetsMap,lemmingsPositionsInit,height,width)    
    
    game=lemmingsGame(obstaclesMap, targetsMap,lemmingsPositionsInit, 
                     lemmingsDirectionsInit, nbFrames,nbMaxBlocks)
    return game

def readGameFile(filename):
    with open(filename, 'r') as f:
        nbFrames=int(f.readline())
        nbMaxBlocks=int(f.readline())
        data = f.readlines()
        
        height=len(data)
        for i,line in enumerate(data):
            if i==0:
                width=len(line)-1
                obstaclesMap=np.zeros((height,width),dtype=np.int)
                targetsMap=np.zeros((height,width),dtype=np.int)
            for j,c in enumerate(line):
                if c=='X':
                    obstaclesMap[i,j]=1
                elif c=='R':
                    lemmingsPositionsInit    =np.array([[i,j]])
                    lemmingsDirectionsInit= np.array([1])
                elif c=='L' :  
                    lemmingsPositionsInit    =np.array([[i,j]])
                    lemmingsDirectionsInit= np.array([-1])
                elif c=='T':
                    targetsMap[i,j]=1
                elif c=='.':
                    pass
                else:
                    print('unkown element %s',c)
                    raise
            
            
                
        game=lemmingsGame(obstaclesMap, targetsMap,lemmingsPositionsInit, 
                          lemmingsDirectionsInit, nbFrames,nbMaxBlocks)
        return game                
    
        
       


class lemmingsGame():
    def __init__(self,obstaclesMap,targetsMap,lemmingsPositionsInit,lemmingsDirectionsInit,nbFrames,nbMaxBlocks=3):
        height,width=obstaclesMap.shape
        self.obstaclesMap=obstaclesMap
        nbLemmings=len(lemmingsPositionsInit)
        self.height=height
        self.width=width
        self.nbFrames=nbFrames
        self.nbLemmings=nbLemmings
        self.nbMaxBlocks=nbMaxBlocks
        self.lemmingsPositionsInit=lemmingsPositionsInit
        self.lemmingsDirectionsInit=lemmingsDirectionsInit
        self.targetsMap=targetsMap
        self.lemmingsMapsInit=np.zeros((height,width,2))
        drawLemmingsInMap(self.lemmingsMapsInit[None,:,:],self.lemmingsPositionsInit[None,:,:],self.lemmingsDirectionsInit[None,:])        
       
        
                
    def simulate(self,obstaclesMap):
        # this function simulate lemmings, in order to keep the code as simple as possible we handle the boundary using the modulo operator.
        # therefore obsacle must be added on the boundary to avoid the lemmings reintering the map from the opposite side
        nbLemmings=len(self.lemmingsPositionsInit)    
        height,width=obstaclesMap.shape
        lemmingsPositions=np.zeros((self.nbFrames,self.nbLemmings,2),dtype=np.int32)
        lemmingsDirections=np.zeros((self.nbFrames,self.nbLemmings),dtype=np.int32)
        lemmingsPositions[0]=self.lemmingsPositionsInit.copy()
        lemmingsDirections[0]=self.lemmingsDirectionsInit.copy() 
        for iFrame in range(1,self.nbFrames):
            lemmingsDirections[iFrame]=lemmingsDirections[iFrame-1]
            lemmingsPositions[iFrame]=lemmingsPositions[iFrame-1]        
            for i in range(self.nbLemmings):
                if  obstaclesMap[(lemmingsPositions[iFrame-1,i,0]+1)%height,lemmingsPositions[iFrame-1,i,1]]==0 :
                    lemmingsPositions[iFrame,i,0]=(lemmingsPositions[iFrame-1,i,0]+1) % height
                elif  obstaclesMap[lemmingsPositions[iFrame-1,i,0],(lemmingsPositions[iFrame-1,i,1]+lemmingsDirections[iFrame-1,i]) % width]==0:
                    lemmingsPositions[iFrame,i,1]=(lemmingsPositions[iFrame-1,i,1]+lemmingsDirections[iFrame,i]) % width                    
                else:
                    lemmingsDirections[iFrame,i]=-lemmingsDirections[iFrame-1,i]
                    
        return lemmingsPositions,lemmingsDirections 
    
    
    def simulateMap(self,obstaclesMap):
        lemmingsPositions,lemmingsDirections=self.simulate(obstaclesMap)
        lemmingsMaps=np.zeros((self.nbFrames,self.height,self.width,2))
        drawLemmingsInMap(lemmingsMaps,lemmingsPositions,lemmingsDirections) 
        return lemmingsMaps
        

if __name__ == "__main__":
    
    
    
    #game=randomGame(height=10, width=10,seed=15,nbMaxBlocks=4,startLeftCorner=True) 
    game=readGameFile('level.txt')
    
    lemmingsMaps=np.zeros((game.nbFrames,game.height,game.width,2))
    

    obstaclesMap=game.obstaclesMap.copy()   
    # simulating the lemmings
    lemmingsPositions,lemmingsDirections=game.simulate(game.obstaclesMap)
    
    # in order to visualize the simulation as an animation we draw the lemmings in an array and then we use matplotlib
    gifZoom=30    
    # visualizing the animation
    
    lemmingsMaps=np.zeros((game.nbFrames,game.height,game.width,2))
    drawLemmingsInMap(lemmingsMaps,lemmingsPositions,lemmingsDirections)
    
    displayLemmingsAnimationPyGame(lemmingsMaps, obstaclesMap, game.targetsMap,'images/firstAnimationB.gif')
    displayLemmingsAnimation(lemmingsMaps,obstaclesMap,game.targetsMap,'images/firstAnimation.gif')
    

    

