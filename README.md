## Goal

The goal of this project is to devise an artificial intelligence that solves a small puzzle game that bares some similarities with the game lemmings, but that is much simpler.
The goal is to get the lemming to reach a the target point after N time steps in a 2D vertical map by adding bricks in the map.
It looks like an interesting problem as the AI needs to plan things. Random strategies are very unlikely to provide a solution.
We also want to be able for the AI to learn the rules governing the lemmings behaviour/physics from few examples.

## Dependencies

* numpy
* matplotlib
* scipy
* pygame (use to display better graphics with sprites)
* imageio (used to save animated gifs)
* scikit-learn
* [PysparseLP](https://github.com/martinResearch/PySparseLP)
* google's ortools for python. https://developers.google.com/optimization/
* Coin-or CBC and CLP executables. download binaries [clp.exe](https://www.coin-or.org/download/binary/Clp/) and [cbc](https://www.coin-or.org/download/binary/Cbc/) and copy them in the solvers\windows subfolder


## game rule

We have a 2D map of obstacles in a 2D vertical world with a starting point for the lemming, a starting orientation of the lemming (left or right), a target point. The original obstacles cannot be removed
and the player is only allowed to add obstacles.
Once the player has defines a set of new obstacles, the lemming in released. The goal is to get the lemming to the target point after N time steps.
The physics of the lemmings can be described as follow: 
If there is no obstacle below then lemming goes down one cell, otherwise it moves in the direction (left or right)that it has been previously given if no obstacle is encountered
If there is an obstacle in that direction the lemming changes direction and doesn't move that frame


An example of a problem with a maximum of 3 bricks and 20 time steps

![animation](./images/firstAnimation.gif)

Solution found by adding bricks using Coin-Or CBC Mixed Integer Linear Program (MILP) solver

![animation](./images/solutionCBC.gif)

## simulation 

simple implementation:

```
#!python
def lemmingsSimulation(obstaclesMap,targetsMap,lemmingsPositionsInit,lemmingsDirectionsInit,nbFrames):
  # this function simulate lemmings, in order to keep the code as simple as possible we handle the boundary using the modulo operator.
  # therefore obsacle must be added on the boundary to avoid the lemmings reintering the map from the opposite side
  nbLemmings=len(lemmingsPositionsInit)  
  height,width=obstaclesMap.shape
  lemmingsPositions=np.zeros((nbFrames,nbLemmings,2),dtype=np.int32)
  lemmingsDirections=np.zeros((nbFrames,nbLemmings),dtype=np.int32)
  lemmingsPositions[0]=lemmingsPositionsInit.copy()
  lemmingsDirections[0]=lemmingsDirectionsInit.copy()
  
  
  for i in range(1,nbFrames):
    lemmingsDirections[i]=lemmingsDirections[i-1]
    lemmingsPositions[i]=lemmingsPositions[i-1]    
    for i in range(nbLemmings):
      if targetsMap[lemmingsPositions[i-1,i,0],lemmingsPositions[i-1,i,1]]==0:
        if obstaclesMap[(lemmingsPositions[i-1,i,0]+1)%height,lemmingsPositions[i-1,i,1]]==0 :
          lemmingsPositions[i,i,0]=(lemmingsPositions[i-1,i,0]+1) % height
        elif obstaclesMap[lemmingsPositions[i-1,i,0],(lemmingsPositions[i-1,i,1]+lemmingsDirections[i-1,i]) % width]==0:
          lemmingsPositions[i,i,1]=(lemmingsPositions[i-1,i,1]+lemmingsDirections[i,i]) % width          
        else:
          lemmingsDirections[i,i]=-lemmingsDirections[i-1,i]
          
  return lemmingsPositions,lemmingsDirections
	
```


## First method: relax the problem from integer variables to continuous variables

Instead of representing the state of the lemmings at each frame with three integer variables we encode its state using a binary 3D occupancy  map array
full of zeros but one value with a one. 
Using this representation we can rewrite the simulation as follow:

```
#!python

def lemmingsSimulationStateSpace(lemmingsMaps,obstaclesMap,targetsMap,nbFrames):
  for i in range(1,nbFrames):
    for d in range(2):
      s=d*2-1
      lemmingsMaps[i,:,:,d]+=np.roll(lemmingsMaps[i-1,:,:,d]*(1-targetsMap),1,axis=0)*(1-obstaclesMap)
      lemmingsMaps[i,:,:,d]+=np.roll(lemmingsMaps[i-1,:,:,d]*(1-targetsMap),s,axis=1)*(1-obstaclesMap)*np.roll(np.roll(obstaclesMap,s,axis=1),-1,axis=0)
      lemmingsMaps[i,:,:,d]+=lemmingsMaps[i-1,:,:,1-d]*(1-targetsMap)*np.roll(obstaclesMap,s,axis=1)*np.roll(obstaclesMap,-1,axis=0)
      lemmingsMaps[i,:,:,d]+=lemmingsMaps[i-1,:,:,d]*targetsMap 		
```

We did not use logical or and and but addition and multiplication on purpose, this allows us to relax the binary constraint and to be able to simulate with continuous obstacle values on the interval between 0 and 1 that can be interpreted as a kind of density.
We can maximise the score by updating gradually the continuous valued obstacles map using gradient ascent. For the the gradient computation to be efficient we use an hand coded reverse accumulation of the gradient(a.k.a adjoint or back-propagation in the neural network context)
The corresponding code is in [lemmingsGradientDescent.py](./lemmingsGradientDescent.py)
The simulation with non zero (0.3) initial values
 
 
![animation](./images/relaxedSimulation.gif)

We are maximising a non linear function and thus may fall in a local minimum.
We need a careful initialisation with non zero obstacles values in order to get a fraction of the lemmings to reach the target in the initialisation and get on non-zero initial gradient for the method to start changing the obstacles map.
The obstacle evolution through gradient descent
  
![animation](./images/obstaclesOptimization.gif)

We added a small penalty on the obstacle map to encourage using as little new bricks as possible.
However note that not all the cells are binary at convergence, maybe we need more iterations, break some symmetry by using a slightly different penalty for each cell.
Finally we simulate the lemmings with the found non binary map

![animation](./images/solutionSimulation1.gif)

## Second method: using a 0-1 integer programming formulation

we can first interpret the problem as solving a non linear program through the introduction of a set of auxiliary variables that represent the number or density of lemmings coming from the top , the right ,the left and staying on the same cell.
we want to maximise  
```
#!python
score=np.sum(np.sum(lemmingsMaps[-1],axis=2)*targetsMap)
```
with respect to obstaclesMap lemmingsMaps and auxiliaryVariables
under constraints isValid(obstaclesMap, lemmingsMaps, auxVars) is true

```
#!python 

def isValid(obstaclesMap, lemmingsMaps, auxVars):
	v=True
  for i in range(1,nbFrames):
    for d in range(2):
      s=d*2-1
      v=v & np.all(auxVars[0,i,:,:,d] == np.roll(lemmingsMaps[i-1,:,:,d]*(1-targetsMap),1,axis=0)*(1-obstaclesMap)))
      v=v & np.all(auxVars[1,i,:,:,d] == np.roll(lemmingsMaps[i-1,:,:,d]*(1-targetsMap),s,axis=1)*(1-obstaclesMap)*np.roll(np.roll(obstaclesMap,s,axis=1),-1,axis=0))
      v=v & np.all(auxVars[2,i,:,:,d] == lemmingsMaps[i-1,:,:,1-d]*(1-targetsMap)*(np.roll(obstaclesMap,s,axis=1))*np.roll(maxDensity*obstaclesMap,-1,axis=0))
      v=v & np.all(auxVars[3,i,:,:,d] == lemmingsMaps[i-1,:,:,d]*targetsMap)  
      v=v & np.all(lemmingsMaps[i,:,:,d] == auxVars[0,i,:,:,d]+auxVars[1,i,:,:,d]+auxVars[2,i,:,:,d]+auxVars[3,i,:,:,d])
	return v
```

Instead of using non linear constraints that involve product between variables, we relax the problem by replacing them by a set of linear inequalities.
We replace a=b*c with binary values by (a<=b)&(a<=c)&(a>=b+c-1).
this allows us to reformulate the problem as a 0-1 integer linear program.
There are several modelling tools that are available to model linear programs from python ([Pulp](https://pythonhosted.org/PuLP),[CyPL](https://github.com/coin-or/CyLP),[Pyomo](http://www.pyomo.org/),[Cvxpy](http://www.cvxpy.org/en/latest/)[Cvxpy](http://www.cvxpy.org/en/latest/))
But none of these allows to handle array of variables with numpy arrays operations like slicing or the roll function that we are using, and we would have to write loops over array elements.
We use our own tool [PySparseLP](https://github.com/martinResearch/PySparseLP) which allows to define easily a set of constraints using several numpy arrays containing variable indices and coefficients.
The python code to formulate the MILP is in [lemmingsMILP.py](./lemmingsMILP.py).
We export the integer linear program into the MPS file format and solve the relaxed linear program (without the integer constraint) using the Coin-Or [CLP](https://www.coin-or.org/download/binary/Clp) executable. We get:

![animation](./images/solutionCLP.gif) 

Using a brick with density 2/3, the lemming splits into a lemmings with density 1/3 that can walk of bricks with density 1/3 given our set of lienar constraints.
Solving the problem with CBC takes about 3 seconds.

We solve the integer linear program (i.e. enforcing the variables to be integers) using Coin-Or [CBC](https://www.coin-or.org/download/binary/cbc) executable. We get

![animation](./images/solutionCBC.gif)

Solving the problem with CBC takes about 10 seconds.
 

## Third method: using a constraints satisfaction solver
 
There are quite a few python tools to formulate problems as SAT problems and to solve them ([or-tools](https://developers.google.com/optimization/), [numberjack](http://numberjack.ucc.ie/ ), [constraint](https://www.logilab.org/project/logilab-constraint)).
 A more exhaustive list can be found [here](./SATPythonTools.md).
I used google's or-tools to formulate and solve the problem. The formulation is written in [lemmingsSAT.py](./lemmingsSAT.py). Unfortunately This tool doesn't allow to define a set of constraint at once using numpy arrays notations, so we have to write each constraint at a time using loops over  array elements.
I find 6 solutions using the constraint that the number of added bricks should less or equal to 3.

![animation](./images/SolutionSAT1.gif) 
![animation](./images/SolutionSAT2.gif) 
![animation](./images/SolutionSAT3.gif) 
![animation](./images/SolutionSAT4.gif) 
![animation](./images/SolutionSAT5.gif) 
![animation](./images/SolutionSAT6.gif) 
 
findind all 6 solutions takes about 10 seconds and making sure there are no other solutions takes about 10 more seconds.
 
## learning the game's rules


We first generate random maps and simulate the lemming for each of these maps
From this simulated data we learn to predict a cell value from neighbouring  cells in the previous frames
we learn from fully observable state (direction given) as it is much more difficult to learn the rules having access only to the lemmings position and not heir orientation as we would then have to either 
learn rules based on an hidden direction state or use a large context with several previous frames, but we would have to get a very large context to go far back in time to guess the 
left/right direction of a falling lemmings.
We use a simple logistic regression but we augment the feature vectors by using product between pairs of variable in the local context.
We then simulate the lemmings using the the trained model with rounding and without rounding the predicted class probability.
The idea behind not rounding is to get a score that is differentiable with respect to the obstacle map and use a simple gradient descent method to solve the problem of estimating the obstacle that 
lead the lemming to the target, as we did previously with the hand coded rules.

simulation with rounding:

![animation](./images/learnedSimulationLogisticRegressionWithRounding.gif)

simulation without rounding:

![animation](./images/learnedSimulationLogisticRegressionWithoutRounding.gif)

Without rounding the predicted probabilities the lemmings loses a bit o contrast. We use normalisation to keep the sum to one, the lemmings spreads on some other areas
this is visible if we multiply by 30 the lemmings density to get better contrasted images

![animation](./images/learnedSimulationLogisticRegressionWithoutRoundingHigherContrast.gif)


We tried a Two layers neural network and a decision tree classifier. We get the best result with the decision tree classifier. Even by multiplying the lemmings density by 30 to augment the contrast we don't observe any wrong location of the lemmings.

![animation](./images/learnedSimulationDecisionTreeWithoutRoundingHigherContrast.gif)

with enough training data the decision seems to learn the right set of rules
The rules to predict lemmings[iFrame,i,j,0] i.e the lemmings values at location i,j pointing toward the left is 

	The binary tree structure to predict lemmings[iFrame,i,j,0] has 17 nodes and has the following tree structure:
	node=0 test node: go to node 1 if lemmings[iFrame-1,i,j,1] <= 0.5 else to node 12.
		node=1 test node: go to node 2 if lemmings[iFrame-1,i,j+1,0] <= 0.5 else to node 7.
			node=2 test node: go to node 3 if lemmings[iFrame-1,i-1,j,0] <= 0.5 else to node 4.
				node=3 leaf node with value 0.
				node=4 test node: go to node 5 if obstacle[iFrame-1,i,j] <= 0.5 else to node 6.
					node=5 leaf node with value 1.
					node=6 leaf node with value 0.
			node=7 test node: go to node 8 if obstacle[iFrame-1,i,j] <= 0.5 else to node 11.
				node=8 test node: go to node 9 if obstacle[iFrame-1,i+1,j+1] <= 0.5 else to node 10.
					node=9 leaf node with value 0.
					node=10 leaf node with value 1.
				node=11 leaf node with value 0.
		node=12 test node: go to node 13 if obstacle[iFrame-1,i,j+1] <= 0.5 else to node 14.
			node=13 leaf node with value 0.
			node=14 test node: go to node 15 if obstacle[iFrame-1,i+1,j] <= 0.5 else to node 16.
				node=15 leaf node with value 0.
				node=16 leaf node with value 1.
					
The rules to predict lemmings[iFrame,i,j,1] i.e. the lemmings values at location i,j pointing toward the right is 					

	The binary tree structure to predict lemmings[iFrame,i,j,1] has 17 nodes and has the following tree structure:
	node=0 test node: go to node 1 if lemmings[iFrame-1,i,j,0] <= 0.5 else to node 12.
		node=1 test node: go to node 2 if lemmings[iFrame-1,i,j-1,1] <= 0.5 else to node 7.
			node=2 test node: go to node 3 if lemmings[iFrame-1,i-1,j,1] <= 0.5 else to node 4.
				node=3 leaf node with value 0.
				node=4 test node: go to node 5 if obstacle[iFrame-1,i,j] <= 0.5 else to node 6.
					node=5 leaf node with value 1.
					node=6 leaf node with value 0.
			node=7 test node: go to node 8 if obstacle[iFrame-1,i,j] <= 0.5 else to node 11.
				node=8 test node: go to node 9 if obstacle[iFrame-1,i+1,j-1] <= 0.5 else to node 10.
					node=9 leaf node with value 0.
					node=10 leaf node with value 1.
				node=11 leaf node with value 0.
		node=12 test node: go to node 13 if obstacle[iFrame-1,i,j-1] <= 0.5 else to node 14.
			node=13 leaf node with value 0.
			node=14 test node: go to node 15 if obstacle[iFrame-1,i+1,j] <= 0.5 else to node 16.
				node=15 leaf node with value 0.
				node=16 leaf node with value 1.



In order to measure how many simulated games need to be used to train the right set of rules we can generate a huge amount of simulated games and measure the percentage of games that violates the rules as we use more and more of theses games to train the rules.
We could also check that the set of learnt rules is equivalent to the set of hand coded rules using a brute force method by enumerating all possibilities.

Instead of training a decision tree we could use a table for each possible configuration in a small neighbourhood and we can either

* be conservative and assume that any configuration not seen in the training set if forbidden
* be optimistic and assume that any configuration not seen in the training set is allowed
* penalise each configuration depending on how frequent they are in the training dataset

this is similar to naive estimation of potentials in a Markov field.
however we do not want to learn that some obstacle configuration are no permitted regardless of the lemmings position just because they where not existing on the training set.
we do not want to learn the distribution of the obstacle but the distribution of the lemming conditioned to the obstacles.

Instead of learning the rules from random games, we may want the learning system to be *active* and generate query games on which the simulation of the lemmings is run. We aim then at learning the right set of rules with a minimum number of query games.


we can learn a DNF using a decisionlist (from artificial intelligence a modern approach)
https://github.com/aimacode/aima-python/blob/master/learning.py

we could try to learn a disjonctive normal form by gradient descent:
c_i= prod_j((1-x_j)^aij *x_j^bij)
y=1-prod(1-c_i)
aij>0
bij>0
aij+bij<1
minimize w.r.t aij and bij
we start near 0
penalize a bit aij and bij to get sparse clauses ?

maybe we could learn rules using inductive logic programming?(see *artificial inelligence a modern approach*)


## solving problem with learnt rules
### Using google's constraint solver with learnt rules
Using a tree classifier we encode the tree decision rules as logic expression for each patch and use these to define a Conjunctive Normal Forms (CNF) on the entire set of patches and solve the problem using google's constraint satisfaction solver.

The code is in lemmingsLearnedRulesSAT.py
We convert the decision trees into Conjunctive Normal Forms (CNF) using the function decisionTreeToCNF.
We obtain

    (lemmings[iFrame-1,i,j,1] or lemmings[iFrame-1,i,j+1,0] or lemmings[iFrame-1,i-1,j,0] or not(lemmings[iFrame,i,j,0])) and
    (lemmings[iFrame-1,i,j,1] or lemmings[iFrame-1,i,j+1,0] or not(lemmings[iFrame-1,i-1,j,0] or not(obstacle[iFrame-1,i,j] or not(lemmings[iFrame,i,j,0])) and
    (lemmings[iFrame-1,i,j,1] or not(lemmings[iFrame-1,i,j+1,0] or not(obstacle[iFrame-1,i,j] or not(lemmings[iFrame,i,j,0])) and
    (not(lemmings[iFrame-1,i,j,1] or obstacle[iFrame-1,i,j+1] or not(lemmings[iFrame,i,j,0])) and
    (not(lemmings[iFrame-1,i,j,1] or not(obstacle[iFrame-1,i,j+1] or obstacle[iFrame-1,i+1,j] or not(lemmings[iFrame,i,j,0])) and
    (lemmings[iFrame-1,i,j,1] or lemmings[iFrame-1,i,j+1,0] or not(lemmings[iFrame-1,i-1,j,0] or obstacle[iFrame-1,i,j] or lemmings[iFrame,i,j,0]) and
    (lemmings[iFrame-1,i,j,1] or not(lemmings[iFrame-1,i,j+1,0] or obstacle[iFrame-1,i,j] or lemmings[iFrame,i,j,0]) and
    (not(lemmings[iFrame-1,i,j,1] or not(obstacle[iFrame-1,i,j+1] or not(obstacle[iFrame-1,i+1,j] or lemmings[iFrame,i,j,0])



    (lemmings[iFrame-1,i,j,0] or lemmings[iFrame-1,i-1,j,1] or lemmings[iFrame-1,i,j-1,1] or not(lemmings[iFrame,i,j,1])) and
    (lemmings[iFrame-1,i,j,0] or lemmings[iFrame-1,i-1,j,1] or not(lemmings[iFrame-1,i,j-1,1] or obstacle[iFrame-1,i,j] or obstacle[iFrame-1,i+1,j-1] or not(lemmings[iFrame,i,j,1])) and
    (lemmings[iFrame-1,i,j,0] or lemmings[iFrame-1,i-1,j,1] or not(lemmings[iFrame-1,i,j-1,1] or not(obstacle[iFrame-1,i,j] or not(lemmings[iFrame,i,j,1])) and
    (lemmings[iFrame-1,i,j,0] or not(lemmings[iFrame-1,i-1,j,1] or not(obstacle[iFrame-1,i,j] or not(lemmings[iFrame,i,j,1])) and
    (not(lemmings[iFrame-1,i,j,0] or obstacle[iFrame-1,i,j-1] or not(lemmings[iFrame,i,j,1])) and
    (not(lemmings[iFrame-1,i,j,0] or not(obstacle[iFrame-1,i,j-1] or obstacle[iFrame-1,i+1,j] or not(lemmings[iFrame,i,j,1])) and
    (lemmings[iFrame-1,i,j,0] or lemmings[iFrame-1,i-1,j,1] or not(lemmings[iFrame-1,i,j-1,1] or obstacle[iFrame-1,i,j] or not(obstacle[iFrame-1,i+1,j-1] or lemmings[iFrame,i,j,1]) and
    (lemmings[iFrame-1,i,j,0] or not(lemmings[iFrame-1,i-1,j,1] or obstacle[iFrame-1,i,j] or lemmings[iFrame,i,j,1]) and
    (not(lemmings[iFrame-1,i,j,0] or not(obstacle[iFrame-1,i,j-1] or not(obstacle[iFrame-1,i+1,j] or lemmings[iFrame,i,j,1])

we add the extra rules that the number of lemmings on the target should sum to one on the last frame and the constraint that we use a limited number of bricks. This last constraint is a cardinality constraint that might be tricky to convert to Conjunctive Normal Forms [6,7] if we want to use a SAT solver that need the problem to be formulated a a conjunctive normal form like minisat (not that some code based on minisat like [minsatp](https://github.com/niklasso/minisatp) or [minicard](https://github.com/liffiton/minicard) handles cardinality constraints).
 
Fortunately cardinality constraints are handled by google's constraint programming tool.
The solver find the 6 solutions. We note that finding all 6 solutions using the learnt rules is about 10 times slower than with the hand coded rules ( 110 seconds vs 10 seconds) and making sure all solution have been found much longer (205 seconds vs 22 seconds).

![animation](./images/SolutionLearnedRulesSAT1.gif)
![animation](./images/SolutionLearnedRulesSAT2.gif)
![animation](./images/SolutionLearnedRulesSAT3.gif)
![animation](./images/SolutionLearnedRulesSAT4.gif)
![animation](./images/SolutionLearnedRulesSAT5.gif)
![animation](./images/SolutionLearnedRulesSAT6.gif)



Maybe we could remove to variables in some of the clauses which could make the problem easier for SAT solvers to solve (maybe see Quinlan 1987).
We may actually want to get more rules than the strict minimum and have some redundancy and larger contexts in order to make it easier for tree search based SAT solver to prune nodes.
Is is a bit similar to clause learning in SAT solvers? we do not want too many clauses but we want clauses that cane prune bad solution quickly.

### Using Integer programming with learnt rules

we can convert each clauses in the CNF formula to a linear inequality and solve the problem using an integer programming solver like coin-or CBC or a pseudo Boolean optimisation solver like [minicard](https://github.com/liffiton/minicard).

## Generating difficult solvable games

we can generate random game and then try to find the solution with a minimum number of bricks. We can try to solve with one brick, then two etc
or use a integer programming solver with a penalization on the number of added bricks.
in order to be able to generate quickly feasible games we want to be able to discard quickly unfeasible games.


## Using other methods like convex input neural network or OptNet

I could have a better look at some structured-learning methods like

* OptNet [paper](https://arxiv.org/abs/1703.00443), [code](https://github.com/locuslab/optnet)
* Unifying Local Consistency and MAX SAT Relaxations for Scalable Inference with Rounding Guarantees. [paper](https://en.wikipedia.org/wiki/Markov_logic_network)
* Hinge-Loss Markov Random Fields and Probabilistic Soft Logic. [paper](https://arxiv.org/pdf/1505.04406.pdf),[code](http://psl.linqs.org/)
* Input-Convex Neural Network [paper](https://arxiv.org/abs/1609.07152), [code](https://github.com/locuslab/icnn)



## Other add-hoc or heuristic methods

We generate a set of solved problem examples obtained either by generating random game and solving them with one of the method above or using a method that generate solution and then removed bricks to generate a problem.
From this set of examples we can learn some heuristic function that predict for example a lower bound on the the number of bricks that are needed to solve the problem.
We then add one brick at a time and perform an A* search using this heuristic.
Note that at each time we add a brick there is no point trying to add a brick that is not on the lemmings trajectory, can this be learnt by the machine ?
 

maybe use approach like

* Learning Generalized Reactive Policies Edward Groshev using Deep Neural Networks https://arxiv.org/pdf/1708.07280.pdf
* Imagination-Augmented Agents for Deep Reinforcement Learning https://arxiv.org/pdf/1707.06203.pdf
 
## Other games
 
some games where we could apply the same approach to learn the rules from examples and then solve new instances with the learned rules


games with gravity

* Fire N' Ice: https://www.youtube.com/watch?v=1t782B0zK3Y
* iso ball 2D https://play.google.com/store/apps/details?id=net.asort.isoball2d&hl=en
* http://www.gamesgames.com/game/isoball-3 , similar to our gamme in 3D with a ball that can break if falls from too high
* gravnix https://www.youtube.com/watch?v=k3iZqXxbnWc

Sokoban and variants

* some sokoban variants: http://sokoban-jd.blogspot.fr/2013/04/sokoban-variants.html
* polar slide (a sokoban variant with slidings blocks): we could learn rules from random generated moves of the player  http://game-game.com/28523/
* Solomon's Key https://www.youtube.com/watch?v=ADB15SFW6hQ
* unger http://puzzlesea.com/unger
* Atomix. https://en.wikipedia.org/wiki/Atomix_(video_game). Open source clone here http://atomiks.sourceforge.net/
* Oxyd https://en.wikipedia.org/wiki/Oxyd
* Xor. Clone here http://jwm-art.net/?p=XorCurses

others

* peg solitaire: we learn rules from examples of valid moves

could have a look at https://en.wikipedia.org/wiki/List_of_puzzle_video_games


## Some references

* [1] *Partial Formalizations and the Lemmings Game*. John McCarthy. 1994. [download](http://jmc.stanford.edu/articles/lemmings/lemmings.pdf)
* [2] *Scripting the Game of Lemmings with a Genetic Algorithm*. Graham Kendall and Kristian Spoerer. Congress on Evolutionary Computation, 2004. ECEC2004. 
[download](http://www.cs.nott.ac.uk/~pszgxk/papers/cec2004kts.pdf)
* [3] *The Lemmings Puzzle: Computational Complexity of an Approach and Identification of Difficult Instances*. Kristian Spoerer. PhD Thesis. University of Nottingham 2007
[download](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.109.7654&rep=rep1&type=pdf)
* [4] *Micro and macro lemmings simulations based on ants colonies*. 	Antonio González-Pardo, Fernando Palero, David Camacho. EvoApplications 2014
* [5] *Lemmings is PSPACE-complete*. Giovanni Viglietta. Theoretical Computer Science. 2012. [download](https://arxiv.org/pdf/1202.6581.pdf)
* [6] *Towards Robust CNF Encodings of Cardinality Constraints*. Joao Marques-Silva and Ines Lynce[download](http://www.inesc-id.pt/ficheiros/publicacoes/4125.pdf)
* [7] *Game Engine Learning from Video*. Matthew Guzdial, Boyang Li, Mark O. Riedl [download](https://www.cc.gatech.edu/~riedl/pubs/ijcai17.pdf)
* [8] *Stochastic Variational Video Prediction*.  [download](https://arxiv.org/pdf/1710.11252.pdf)
* [9] *New Approaches to Constraint Acquisition*.ICON book chapter. C. Bessiere, R. Coletta,A. Daoudi, E. Hebrard, G. Katsirelos, N. Lazaar, Y. Mechqrane, N. Narodytska, C.G. Quimper, and T. Walsh. http://www.lirmm.fr/ConstraintAcquisition/
* [10] *Learning DNF Formulas*. Shie Mannor and Shai Shalev-Shwartz.  [download](https://www.cs.huji.ac.il/~shais/DNF.pdf)
* [11] *Learning DNF by Decision Trees*. Giulia Pagallo.  [download](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=65FF15062DDE3449803BBC66EBA87877?doi=10.1.1.104.573&rep=rep1&type=pdf) by Giulia Pagallo 
* [12] *A  Scheme  for  Feature  Construction  and  a  Comparison  of Empirical  Methods*. Der-Shung Yang, Larry Rendell, Gunnar Blix.  [download](https://www.ijcai.org/Proceedings/91-2/Papers/014.pdf)
* [13] *Generating production rules from decision trees*.     J.R.  Quinlan In IJCAI 1987.  
* [14] *Graph Neural Networks and Boolean Satisfiability*. Benedikt Bünz, Matthew Lamm. 2017.  [download](https://arxiv.org/pdf/1702.03592.pdf)
* [15] *Learning pseudo-Boolean k-DNF and Submodular Functions*. Sofya Raskhodnikova, Grigory Yaroslavtsev. [ download](https://arxiv.org/pdf/1208.2294.pdf)
* [16] *An O(n log(log(n))) Learning Algorithm For DNF under the Uniform Distribution*. Yishay Mansour.  [Download](http://www.cs.columbia.edu/~rocco/Teaching/S12/Readings/Mansour.pdf)
* [17] *Learning DNF from Random Walks*. Nader Bshouty.  [download](https://www.cs.cmu.edu/~odonnell/papers/random-walks.pdf)
* [18] *Learning Complex Boolean Functions:Algorithms and Applications* Arlindo L. Oliveira and Alberto Sangiovanni-Vincentelli.  [download](https://papers.nips.cc/paper/857-learning-complex-boolean-functions-algorithms-and-applications.pdf)
* [19] *On Construction of Binary Higher Order Neural Networks and Realization of Boolean Functions*. Poorva Agrawal and Aruna Tiwari.  [download](https://www.ijcait.com/IJCAIT/12/1213.pdf)
* [20] *Binary Higher Order Neural Networks for Realizing Boolean Functions*. Chao Zhang. [download ](https://pdfs.semanticscholar.org/45ac/ac17a71e43f2a70c820ce7f4b6b6299b3175.pdf)
* [21] *Realization of Boolean Functions Using Binary Pi-sigma Networks*. Yoan Shin and Joydeep Ghosh. [download](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.52.9813&rep=rep1&type=pdf)
* [22] *Towards Deep Symbolic Reinforcement Learning* Marta Garnelo, Kai Arulkumaran and Murray Shanahan. [download ](https://arxiv.org/pdf/1609.05518.pdf)
* [23] *OptNet: Differentiable Optimization as a Layer in Neural Networks* Brandon Amos, J. Zico Kolter. [paper](https://arxiv.org/abs/1703.00443), [code](https://github.com/locuslab/optnet)
* [24] *Unifying Local Consistency and MAX SAT Relaxations for Scalable Inference with Rounding Guarantees*.Stephen H. Bach, Bert Huang,Lise Getoor. [paper](http://proceedings.mlr.press/v38/bach15.pdf)
* [25] *Hinge-Loss Markov Random Fields and Probabilistic Soft Logic*. Stephen H. Bach, Matthias Broecheler, Bert Huang, Lise Getoor. [paper](https://arxiv.org/pdf/1505.04406.pdf),[code](http://psl.linqs.org/)
* [26] *Input-Convex Neural Network* [paper](https://arxiv.org/abs/1609.07152). Brandon Amos, Lei Xu, J. Zico Kolter [code](https://github.com/locuslab/icnn)
* [27] *Learning Generalized Reactive Policies Edward Groshev using Deep Neural Networks*. Edward Groshev, Aviv Tamar. [paper](https://arxiv.org/pdf/1708.07280.pdf)
* [28] *Imagination-Augmented Agents for Deep Reinforcement Learning*. Théophane Weber Sébastien Racanière David P. Reichert Lars Buesing Arthur Guez Danilo Rezende Adria Puigdomènech Badia Oriol Vinyals Nicolas Heess Yujia Li Razvan Pascanu Peter Battaglia David Silver Daan Wierstra [paper](https://arxiv.org/pdf/1707.06203.pdf)
 