import pygame
import imageio
pygame.init()

pygame.mixer.init()


lemmingLeft = pygame.transform.rotozoom(pygame.image.load("resources/lemmingLeft.png"),0,0.5)
lemmingRight =  pygame.transform.rotozoom(pygame.image.load("resources/lemmingRight.png"),0,0.5)
brick =  pygame.transform.rotozoom(pygame.image.load("resources/brick.png"),0,0.5)
target =  pygame.transform.rotozoom(pygame.image.load("resources/target.png"),0,0.5)

elementSize=25

def displayLemmingsAnimationPyGame(lemmingsMaps,obstaclesMap,targetsMap,gifName=None):
    height,width=obstaclesMap.shape
    screen=pygame.display.set_mode((width*elementSize, height*elementSize+int(elementSize*0.45)))
    frames=[]
    for iFrame in range(len(lemmingsMaps)):
        screen.fill((255,255,255))
        
        for i in reversed(range(height)):
            for j in range(width):
                if lemmingsMaps[iFrame,i,j,0]>0:
                    lemmingLeft.set_alpha(100*lemmingsMaps[iFrame,i,j,0])
                    screen.blit(lemmingLeft,(j*elementSize,i*elementSize))
                if lemmingsMaps[iFrame,i,j,1]>0:
                    lemmingRight.set_alpha(255*lemmingsMaps[iFrame,i,j,1])
                    screen.blit(lemmingRight,(j*elementSize,i*elementSize))    
                if obstaclesMap[i,j]>0:    
                    brick.set_alpha(255*obstaclesMap[i,j])
                    screen.blit(brick,(j*elementSize,i*elementSize))  
                if targetsMap[i,j]>0: 
                    target.set_alpha(255*targetsMap[i,j])
                    screen.blit(target,(j*elementSize,i*elementSize))              
            
        pygame.display.flip()
        pygame.event.get()# needed to display
        if not gifName is None:
            frames.append(pygame.surfarray.array3d(screen).transpose([1,0,2]))
                          
        pygame.time.wait(200)
    pygame.quit()
    if not gifName is None:
        imageio.mimsave(gifName, frames,duration= 0.2) 
    
