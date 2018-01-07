from pysparselp.SparseLP import SparseLP
import numpy as np
from lpsolvers import solveLPWithCBC,solveLPWithCLP,solveLPWithGLPK
from lemmingsLP import *
from pysparselp.constraintPropagation import greedy_round
from lemmings import *
import copy
import time

def SimulationForLPCheck(lemmingsMaps,obstaclesMap,targetsMap,obstaclesMapsPenalization,auxVars,nbFrames,display=False):   
    for iFrame in range(1,nbFrames):
        lemmingsMaps[iFrame].fill(0)
        for d in range(2):
            s=d*2-1
            auxVars[0,iFrame,:,:,d] = np.roll(lemmingsMaps[iFrame-1,:,:,d],1,axis=0)*(1-obstaclesMap)
            auxVars[1,iFrame,:,:,d] = np.roll(lemmingsMaps[iFrame-1,:,:,d],s,axis=1)*(1-obstaclesMap)*np.roll(np.roll(obstaclesMap,s,axis=1),-1,axis=0)
            auxVars[2,iFrame,:,:,d] = lemmingsMaps[iFrame-1,:,:,1-d]*(np.roll(obstaclesMap,s,axis=1))*np.roll(obstaclesMap,-1,axis=0)
            lemmingsMaps[iFrame,:,:,d]=auxVars[0,iFrame,:,:,d]+auxVars[1,iFrame,:,:,d]+auxVars[2,iFrame,:,:,d]
            


    
class linearProgramFormulation(SparseLP):
    
    def setup(self,nbFrames,height,width,lemmingsMapsInit,targetsMap,obstaclesMapsPenalization,maxDensity,obstaclesMapInit,solutionDict=None):
        self.solutionDict=solutionDict
        lemmingsMapsIds=self.addVariablesArray((nbFrames,height,width,2),upperbounds=maxDensity,lowerbounds=0,name='lemmingsMaps')
        obstaclesMapIds=self.addVariablesArray((height,width),upperbounds=1,lowerbounds=0,costs=obstaclesMapsPenalization,name='obstaclesMap')
        auxVarsIds=self.addVariablesArray((4,nbFrames,height,width,2),upperbounds=maxDensity,lowerbounds=0,name='auxVars')
        if not solutionDict is None:
            checkSolution=True
            self.solution=np.zeros((self.nb_variables))
            for key,value in solutionDict.iteritems():
                self.solution[self.getVariablesIndices(key)]=value
            auxVars=solutionDict['auxVars']
            obstaclesMap=solutionDict['obstaclesMap']
            lemmingsMaps=solutionDict['lemmingsMaps']
        else:
            self.solution=None
            checkSolution=False
        

        self.setBoundsOnVariables(lemmingsMapsIds[0,:,:,:],lemmingsMapsInit,lemmingsMapsInit)
        self.setBoundsOnVariables(obstaclesMapIds,obstaclesMapInit,1-(np.sum(lemmingsMapsInit,axis=2)>0))
        self.setBoundsOnVariables(obstaclesMapIds[:,0],1,1)
        
        self.setBoundsOnVariables(obstaclesMapIds[:,width-1],1,1)
        
        self.setBoundsOnVariables(obstaclesMapIds[-1,:],1,1)
        self.setBoundsOnVariables(auxVarsIds[:,0,:,:,:],0,0)
        if checkSolution:
            assert(self.checkSolution(self.solution))
        
        tol=1e-10
        for iFrame in range(1,nbFrames): 
            for d in range(2):
                s=d*2-1
                if checkSolution:
                    assert(np.all(auxVars[0,iFrame,:,:,d]<=np.roll(lemmingsMaps[iFrame-1,:,:,d],1,axis=0))+tol)
                self.addInequalities([(auxVarsIds[0,iFrame,:,:,d],1), (np.roll(lemmingsMapsIds[iFrame-1,:,:,d],1,axis=0),-1)],None,0)
                if checkSolution:
                    assert(np.all(auxVars[0,iFrame,:,:,d]<=maxDensity*(1-obstaclesMap)+tol))                
                self.addInequalities([(auxVarsIds[0,iFrame,:,:,d],1),(obstaclesMapIds,maxDensity)],None,maxDensity)
                if checkSolution:
                    assert(np.all(auxVars[0,iFrame,:,:,d]+obstaclesMap+np.roll(lemmingsMaps[iFrame-1,:,:,d],1,axis=0)*(-1)>=0-tol))
                self.addInequalities([(auxVarsIds[0,iFrame,:,:,d],1),(np.roll(lemmingsMapsIds[iFrame-1,:,:,d],1,axis=0),-1),(obstaclesMapIds,1)],0,None)
                if checkSolution:
                    assert(np.all(auxVars[1,iFrame,:,:,d]<=np.roll(lemmingsMaps[iFrame-1,:,:,d],s,axis=1)+tol))
                self.addInequalities([(auxVarsIds[1,iFrame,:,:,d],1),(np.roll(lemmingsMapsIds[iFrame-1,:,:,d],s,axis=1),-1)],None,0)
                if checkSolution:
                    assert(np.all(auxVars[1,iFrame,:,:,d]+maxDensity*obstaclesMap<=maxDensity+tol))
                self.addInequalities([(auxVarsIds[1,iFrame,:,:,d],1),(obstaclesMapIds,maxDensity)],None,maxDensity)
                if checkSolution:
                    assert(np.all(auxVars[1,iFrame,:,:,d]-maxDensity*np.roll(np.roll(obstaclesMap,s,axis=1),-1,axis=0)<=tol ))     
                self.addInequalities([(auxVarsIds[1,iFrame,:,:,d],1),(np.roll(np.roll(obstaclesMapIds,s,axis=1),-1,axis=0) ,-maxDensity)],None,0)
                if checkSolution:
                    assert(np.all(auxVars[2,iFrame,:,:,d]<=lemmingsMaps[iFrame-1,:,:,1-d]+tol))
                self.addInequalities([(auxVarsIds[2,iFrame,:,:,d],1),(lemmingsMapsIds[iFrame-1,:,:,1-d] ,-1)],None,0)
                if checkSolution:
                    assert(np.all(auxVars[2,iFrame,:,:,d]<=maxDensity*np.roll(obstaclesMap,s,axis=1)+tol))
                self.addInequalities([(auxVarsIds[2,iFrame,:,:,d],1),(np.roll(obstaclesMapIds,s,axis=1),-maxDensity)],None,0)
                if checkSolution:
                    assert(np.all(auxVars[2,iFrame,:,:,d]<=maxDensity*np.roll(obstaclesMap,-1,axis=0)+tol))
                self.addInequalities([(auxVarsIds[2,iFrame,:,:,d],1),(np.roll(obstaclesMapIds,-1,axis=0),-maxDensity)],None,0)
                if checkSolution:
                    assert(np.all(lemmingsMaps[iFrame,:,:,d]<=auxVars[0,iFrame,:,:,d]+auxVars[1,iFrame,:,:,d]+auxVars[2,iFrame,:,:,d]+tol))
                self.addInequalities([(lemmingsMapsIds[iFrame,:,:,d],1),(auxVarsIds[0,iFrame,:,:,d],-1),(auxVarsIds[1,iFrame,:,:,d],-1),(auxVarsIds[2,iFrame,:,:,d],-1)],None,0)
         
        #adding local conservation of lemmings number constraints
        for iFrame in range(1,nbFrames): 
            for d in range(2):  
                s=d*2-1
                if checkSolution:
                    maxdiff=np.max(np.abs(-lemmingsMaps[iFrame-1,:,:,d]+  np.roll(auxVars[0,iFrame,:,:,d],-1,axis=0)+np.roll(auxVars[1,iFrame,:,:,d],-s,axis=1)+auxVars[2,iFrame,:,:,1-d]+auxVars[3,iFrame,:,:,d]))
                    assert(maxdiff<1e-6    )           
                self.addInequalities([(lemmingsMapsIds[iFrame-1,:,:,d],1),( np.roll(auxVarsIds[0,iFrame,:,:,d],-1,axis=0),-1),(np.roll(auxVarsIds[1,iFrame,:,:,d],-s,axis=1),-1),(auxVarsIds[2,iFrame,:,:,1-d],-1),(auxVarsIds[3,iFrame,:,:,d],-1)],0,0)

        for iFrame in range(1,nbFrames): 
            self.addLinearConstraintRow(lemmingsMapsIds[iFrame,:,:,:], np.ones(lemmingsMapsIds[iFrame,:,:,:].shape),maxDensity,maxDensity)  

        # maximize score=np.sum(np.sum(lemmingsMaps[-1],axis=2)*targetsMap)-np.sum(obstaclesMapsPenalization*obstaclesMap)
        for d in range(2):
            self.setCostsVariables(lemmingsMapsIds[-1,:,:,d], -targetsMap)
        self.setCostsVariables(obstaclesMapIds, obstaclesMapsPenalization) 
        

            

def displayLPSolution(x):
    
    lemmingsMaps=x[LP.getVariablesIndices('lemmingsMaps')]
    obstaclesMap=x[LP.getVariablesIndices('obstaclesMap')]
    auxVars=x[LP.getVariablesIndices('auxVars')]
    lemmingsMapsSum=np.sum(lemmingsMaps,axis=3)
    #movieFrames=np.minimum(targetsMap[None,:,:,None]*np.array([0,1,0 ])+obstaclesMap[None,:,:,None]*np.array([1,0,0 ])+lemmingsMapsSum[:,:,:,None]*np.array([0,0.3,.3]),1)
    #score=np.sum(np.sum(lemmingsMaps[-1],axis=2)*targetsMap)-np.sum(obstaclesMapsPenalization*obstaclesMap)
    #displayMovie(movieFrames)
    plt.ion()
    plt.imshow(obstaclesMap[:,:,None]*np.array([1,0,0 ]),interpolation="nearest")
    plt.show()
    
    
def displayLPSolutionMovie(LP,game,solution,imageName):

    lemmingsMaps=solution[LP.getVariablesIndices('lemmingsMaps')]
    obstaclesMap=solution[LP.getVariablesIndices('obstaclesMap')]
    auxVars=solution[LP.getVariablesIndices('auxVars')]
    lemmingsMapsSum=np.sum(lemmingsMaps,axis=3)
    movieFrames=np.minimum(game.targetsMap[None,:,:,None]*targetColor+obstaclesMap[None,:,:,None]*obstacleColor+lemmingsMapsSum[:,:,:,None]*lemmingsColor,1)
    score=np.sum(np.sum(lemmingsMaps[-1],axis=2)*game.targetsMap)-np.sum(obstaclesMapsPenalization*obstaclesMap)
    plt.ioff()
    displayMovie(movieFrames,imageName)
    
def solveGame(game) :
    height,width=game.height,game.width
    
    nbFrames=game.nbFrames
    lemmingsMaps=np.zeros((game.nbFrames,game.height,game.width,2))
    
    
    # starting from the relaxed full state formulation, we derive as set of linear inequality constraints, and define the problem as an interger linear program
    # i reuse the libray https://github.com/martinResearch/PySparseLP (need to be updated from the mercurial repository)
    # in order to use the tools to make it easier to define the LP
    # could instead use cvxpy ? https://www.lfd.uci.edu/~gohlke/pythonlibs/#cvxpy
    
    
    # i generate a feasible solution to check that i did not make mistaks when defining the constraints  
    
    maxDensity=int(np.sum(game.lemmingsMapsInit))
    
    alpha=0
    obstaclesMapsPenalization=np.full((height,width),alpha)
    auxVars=np.zeros((3,game.nbFrames, height, width,2))
    
    lemmingsMaps[0]=game.lemmingsMapsInit
    
    SimulationForLPCheck(lemmingsMaps,game.obstaclesMap,game.targetsMap,obstaclesMapsPenalization,auxVars,game.nbFrames)
    score=np.sum(np.sum(lemmingsMaps[-1],axis=2)*game.targetsMap)-np.sum(obstaclesMapsPenalization*game.obstaclesMap)
    
    
    solutionDict={}
    solutionDict['lemmingsMaps']=lemmingsMaps
    solutionDict['obstaclesMap']=game.obstaclesMap
    solutionDict['auxVars']=auxVars
    LP=linearProgramFormulation()
    LP.setup(nbFrames, height, width,game.lemmingsMapsInit,game.targetsMap.astype(np.int32),obstaclesMapsPenalization,maxDensity,game.obstaclesMap,solutionDict)    
    
    
    tmp=np.nonzero(LP.costsvector)[0]
    nbMaxBlocks=game.nbMaxBlocks
    expectedScore=1-alpha*(nbMaxBlocks+np.sum(game.obstaclesMap))
    LP.addLinearConstraintRow(tmp, LP.costsvector[tmp],None,-expectedScore)
    
    
    # trying other solvers
    LP.convertToOnesideInequalitySystem()
    print 'exporting problem to MPS file format...',
    LP.saveMPS('lemmings.mps')
    print 'done'
    

    solution=solveLPWithCBC('lemmings.mps',LP.nb_variables)
    lemmingsMaps=solution[LP.getVariablesIndices('lemmingsMaps')]
    obstaclesMap=solution[LP.getVariablesIndices('obstaclesMap')]
    auxVars=solution[LP.getVariablesIndices('auxVars')]        
    return {'obstaclesMap':obstaclesMap,'lemmingsMap':lemmingsMaps,'auxVars':auxVars}
    


if __name__ == "__main__":
    
    # setting the number of lemmings and the size of the map
    
    #game=randomGame(height=10, width=10,seed=15,nbMaxBlocks=4,startLeftCorner=True) 
    game=readGameFile('level.txt')
    
    lemmingsMaps=np.zeros((game.nbFrames,game.height,game.width,2))
    
    
    # starting from the relaxed full state formulation, we derive as set of linear inequality constraints, and define the problem as an interger linear program
    # i reuse the libray https://github.com/martinResearch/PySparseLP (need to be updated from the mercurial repository)
    # in order to use the tools to make it easier to define the LP
    # could instead use cvxpy ? https://www.lfd.uci.edu/~gohlke/pythonlibs/#cvxpy
    
    
    # i generate a feasible solution to check that i did not make mistaks when defining the constraints  
    
    maxDensity=int(np.sum(game.lemmingsMapsInit))
    
    alpha=0.01
    obstaclesMapsPenalization=np.full((game.height,game.width),alpha)
    auxVars=np.zeros((4,game.nbFrames, game.height, game.width,2))
    
    lemmingsMaps[0]=game.lemmingsMapsInit
    
    SimulationForLPCheck(lemmingsMaps,game.obstaclesMap,game.targetsMap,obstaclesMapsPenalization,auxVars,game.nbFrames)
    score=np.sum(np.sum(lemmingsMaps[-1],axis=2)*game.targetsMap)-np.sum(obstaclesMapsPenalization*game.obstaclesMap)
    
    
    
    
    solutionDict={}
    solutionDict['lemmingsMaps']=lemmingsMaps
    solutionDict['obstaclesMap']=game.obstaclesMap
    solutionDict['auxVars']=auxVars
    LP=linearProgramFormulation()
    LP.setup(game.nbFrames, game.height, game.width,game.lemmingsMapsInit,game.targetsMap.astype(np.int32),obstaclesMapsPenalization,maxDensity,game.obstaclesMap,solutionDict)    

    
    tmp=np.nonzero(LP.costsvector)[0]
    expectedScore=1-alpha*(game.nbMaxBlocks+np.sum(game.obstaclesMap))
    LP.addLinearConstraintRow(tmp, LP.costsvector[tmp],None,-expectedScore)
    
    
    # trying various solvers
    LP.convertToOnesideInequalitySystem()
    print 'exporting problem to MPS file format...',
    LP.saveMPS('lemmings.mps')
    print 'done'
    
    
    timestart=time.time()
    solution=solveLPWithCLP('lemmings.mps',LP.nb_variables)
    print('solving with CLP took %d seconds'%time.time()-timestart)
    displayLPSolutionMovie(LP,game,solution,'images/solutionCLP.gif')
    
    
    timestart=time.time()
    solution=solveLPWithCBC('lemmings.mps',LP.nb_variables)
    print('solving with CBC took %d seconds'%time.time()-timestart)
    displayLPSolutionMovie(LP,game,solution,'images/solutionCBC.gif')
    
    timestart=time.time()
    solution=solveLPWithGLPK('lemmings.mps')
    print('solving with GLPK took %d seconds'%time.time()-timestart)
    displayLPSolutionMovie(LP,game,solution,'images/solutionGLPK.gif')
 

    

    
    









