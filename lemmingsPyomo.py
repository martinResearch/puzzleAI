from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.core import *
from pyomo import *
from pyomo.opt import *
from pyomo.core.base import *
from pyomo.environ import *



model = AbstractModel()
model.nbFrames = Param(within=NonNegativeIntegers)
model.width = Param(within=NonNegativeIntegers)
model.height = Param(within=NonNegativeIntegers)

model.Frame = RangeSet(1, model.nbFrames)
model.I = RangeSet(1, model.width)
model.J = RangeSet(1, model.height)

model.lemmingsMaps = Var(model.Frame,model.I, model.J,[0,1],domain=Boolean)
model.auxVars = Var([0,1,2,3,4],model.Frame,model.I, model.J,[0,1],domain=Boolean)
model.targetsMap = Param(model.I, model.J,domain=Boolean)
model.obstaclesMap= Var(model.I, model.J,domain=Boolean)
model.targetMap= Var(model.Frame,model.I, model.J,[0,1],domain=Boolean)


#def obj_expression(model):
#return summation(model.c, model.x)
#model.OBJ = Objective(rule=obj_expression)


#def objective():
    #return -sum(lemmingsMaps[nbFrames-1,i,j,d]*targetMap[i,j])

def constraint_rule1(model, iFrame,i,j):
    return model.auxVars[0,iFrame,i,j,d] == model.lemmingsMaps[iFrame-1,i,(j-1)%heigth,d]*(1-targetsMap[i,(j-1)%height])*(1-model.obstaclesMap[i,j])
def constraint_rule2(model, iFrame,i,j):
    return model.auxVars[1,iFrame,i,j,d] == model.lemmingsMaps[iFrame-1,(i-s)%width,j,d]*(1-targetsMap[(i-s)%width,j])*(1-model.obstaclesMap[i,j])*obstaclesMap[(i-s)%width,(j+1)%height]
def constraint_rule3(model, iFrame,i,j):
    return  model.auxVars[2,iFrame,i,j,d] == model.lemmingsMaps[iFrame-1,i,j,1-d]*(1-targetsMap[i,j])*obstaclesMap[(i-s)%width,j]*model.obstaclesMap[i,(j+1)%height]
def constraint_rule4(model, iFrame,i,j):
    model.auxVars[3,iFrame,i,j,d] == model.lemmingsMaps[iFrame-1,i,j,d]*targetsMap[i,j]
def constraint_rule5(model, iFrame,i,j):
    model.lemmingsMaps[iFrame,i,j,d]==model.auxVars[0,iFrame,i,j,d]+model.auxVars[1,iFrame,i,j,d]+model.auxVars[2,iFrame,i,j,d]+model.auxVars[3,iFrame,i,j,d]
    

model.constrains1 = Constraint(model.Frame,model.I,model.J, rule=constraint_rule1)
model.constrains2 = Constraint(model.Frame,model.I,model.J, rule=constraint_rule1)
model.constrains3 = Constraint(model.Frame,model.I,model.J, rule=constraint_rule1)
model.constrains4 = Constraint(model.Frame,model.I,model.J, rule=constraint_rule1)

def Obj_rule(model):
    return sum(lemmingsMaps[nbFrames-1,i,j,d]*targetMap[i,j] for (i,j,d) in model.I * model.J * [0,1])
model.Obj = Objective(rule=Obj_rule, sense=minimize)


instance = model.create()
opt = SolverFactory("cbc")
results = opt.solve(instance)
instance.load(results)
print results