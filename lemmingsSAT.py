from __future__ import print_function
import sys
from ortools.constraint_solver import pywrapcp
from lemmings import *
import time


def checkSolution(obstaclesMap,targetsMap,lemmingsMaps,auxVars):
    nbFrames=lemmingsMaps.shape[0]
    height=lemmingsMaps.shape[1]
    width=lemmingsMaps.shape[2]
    for iFrame in range(1,nbFrames):
            for i in range(height):
                for j in range(width):
                    for d in range(2):
                            s=d*2-1
                            assert(np.all(auxVars[0,iFrame,:,:,d] == np.roll(lemmingsMaps[iFrame-1,:,:,d],1,axis=0)*(1-obstaclesMap)))
                            assert(np.all(auxVars[1,iFrame,:,:,d] == np.roll(lemmingsMaps[iFrame-1,:,:,d],s,axis=1)*(1-obstaclesMap)*np.roll(np.roll(obstaclesMap,s,axis=1),-1,axis=0)))
                            assert(np.all(auxVars[2,iFrame,:,:,d] == lemmingsMaps[iFrame-1,:,:,1-d]*(np.roll(obstaclesMap,s,axis=1))*np.roll(obstaclesMap,-1,axis=0)))
                            assert(np.all(lemmingsMaps[iFrame,:,:,d]==auxVars[0,iFrame,:,:,d]+auxVars[1,iFrame,:,:,d]+auxVars[2,iFrame,:,:,d] ))   
                            
                            
                            
                            
def solveGame(game) :
    
    height,width=game.targetsMap.shape
 
    lemmingsMaps=np.zeros((game.nbFrames,height,width,2))

    solver = pywrapcp.Solver("lemmings")
    
    allVars=[]
    obstaclesMap ={}
    for i in range(height):
        for j in range(width):
            obstaclesMap[i, j] = solver.BoolVar( 'obstaclesMap[%i,%i]' % (i, j)) 
            allVars.append(obstaclesMap[i, j])
    lemmingsMaps = {}
    for iFrame in range(game.nbFrames):
        for i in range(height):
            for j in range(width):
                for d in range(2):
                    lemmingsMaps[iFrame,i, j,d] = solver.BoolVar( 'lemmingsMaps[%i,%i,%i,%i]' % (iFrame,i, j,d))    
                    allVars.append(lemmingsMaps[iFrame,i, j,d])        
            
    auxVars ={}
    for iAux in range(3):
        for iFrame in range(game.nbFrames):
            for i in range(height):
                for j in range(width):
                    for d in range(2):
                        auxVars[iAux,iFrame,i,j,d]= solver.BoolVar( 'auxVars[%i,%i,%i,%i,%i]' % (iAux,iFrame,i, j,d))  
                        allVars.append(auxVars[iAux,iFrame,i,j,d])
    iFrame=0                     
    for iAux in range(3):
        for i in range(height):
            for j in range(width):
                for d in range(2):    
                    solver.Add(auxVars[iAux,iFrame,i,j,d] ==   0 ) #otherwise get many solutions
                        
       
    for i in range(height):
        for j in range(width):
            for d in range(2):    
                solver.Add(lemmingsMaps[iFrame,i,j,d] ==   int(game.lemmingsMapsInit[i,j,d] ) )
               
    for i in range(height):
        for j in range(width):
            solver.Add(obstaclesMap[i,j] > int(game.obstaclesMap[i,j])-1)
    
            
    for iFrame in range(1,game.nbFrames):
        for i in range(height):
            for j in range(width):
                for d in range(2):
                        s=d*2-1                      
                        solver.Add(auxVars[0,iFrame,i,j,d] == lemmingsMaps[iFrame-1,(i-1)%height,j,d] * (1-obstaclesMap[i,j]) )
                        solver.Add(auxVars[1,iFrame,i,j,d] == lemmingsMaps[iFrame-1,i,(j-s)%width,d] *(1-obstaclesMap[i,j])* obstaclesMap[(i+1)%height,(j-s)%width] )
                        solver.Add(auxVars[2,iFrame,i,j,d] == lemmingsMaps[iFrame-1,i,j,1-d] * obstaclesMap[i,(j-s)%width]*obstaclesMap[(i+1)%height,j] )
                        solver.Add(lemmingsMaps[iFrame,i,j,d] == auxVars[0,iFrame,i,j,d]+auxVars[1,iFrame,i,j,d]+auxVars[2,iFrame,i,j,d])
                        
    
    solver.Add(solver.Sum([lemmingsMaps[game.nbFrames-1,i,j,d]*game.targetsMap[i,j] for i in range(height) for j in range(width) for d in range(2)] )==game.nbLemmings)  
    
    solver.Add(solver.Sum([obstaclesMap[i,j] for i in range(height) for j in range(width)] )<game.nbMaxBlocks+np.sum(game.obstaclesMap)+1)                 
    
    db = solver.Phase(allVars, solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE)
    solver.Solve(db)

    count = 0
    solutions=[]
    timestart=time.time()
    solutions=[]
    while solver.NextSolution():
        count += 1
        print("Solution" , count, '\n')
        
        
        lemmingsMapsNp=np.zeros((game.nbFrames,height,width,2))
        obstaclesMapNp=np.zeros((height,width))
        auxVarsNp=np.zeros((3,game.nbFrames,height,width,2))
        
        for iFrame in range(game.nbFrames):
            for i in range(height):
                for j in range(width):
                    for d in range(2):
                        lemmingsMapsNp[iFrame,i, j,d]=lemmingsMaps[iFrame,i, j,d].Value()
                        

        for i in range(height):
            for j in range(width):
                    obstaclesMapNp[i, j]=obstaclesMap[i, j].Value()   
                    
        for iAux in range(3):
            for iFrame in range(game.nbFrames):
                for i in range(height):
                    for j in range(width):
                        for d in range(2):    
                            auxVarsNp[iAux,iFrame,i, j,d]=auxVars[iAux,iFrame,i, j,d].Value() 
                    
        checkSolution(obstaclesMapNp,game.targetsMap,lemmingsMapsNp,auxVarsNp)
        
        np.sum(np.sum(lemmingsMapsNp[-1],axis=2)*game.targetsMap)
        
        print (hash(lemmingsMapsNp.tostring()))
        print (hash(auxVarsNp.tostring()))
        print (hash(obstaclesMapNp.tostring()))
        
        solutions.append( {'obstaclesMap':obstaclesMapNp,'lemmingsMap':lemmingsMapsNp,'auxVars':auxVarsNp})
        print ('finding %d solutions took %d seconds'%(len(solutions),time.time()-timestart))
    print ('finding all solutions took %d seconds'%(time.time()-timestart))
    return solutions

if __name__ == "__main__":
    
    #game=randomGame(height=10, width=10,seed=15,nbMaxBlocks=4,startLeftCorner=True) 
    game=readGameFile('level.txt') 
    
    solutions=solveGame(game)
    count=0
    for  solution in solutions:
        count+=1
    
        gifName='images/SolutionSAT%d.gif'%count               
        displayLemmingsAnimation(solution['lemmingsMap'],solution['obstaclesMap'],game.targetsMap,gifName=gifName,gifZoom=30)    
    if count==0:
        print ('could not find any solution')
