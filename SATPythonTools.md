
 tools to forulated a SAT problem from python:
 
 * [or-tools](https://developers.google.com/optimization/ ). Can be install with pip install ortools (does not seem available fot python 32bits)
 * [satispy]( https://pypi.python.org/pypi/satispy) It supposes that you have minisat installed and accesible in the path and that you are under linux. the code is very small so it might be able to modify it
 * [pycosat]() is python binding to picosat. Unfortunaletly the command *pip install pycosat* fails on my winpython distribution (cannot open include file 'basetsd.h')
 * [numberjack](http://numberjack.ucc.ie/ ). Numberjack is a modelling package written in Python for constraint programming. There are a number of available back-ends, but you will have to install them sepeartely. There are three mixed integer programming solvers  Gurobi and CPLEX (commercial solvers) and SCIP (which is  only free for academic research), satisfiability solvers (MiniSat, Walksat, and many others), a constraint programming solver (Mistral), and a weighted constraint satisfaction solver (Toulbar2).
    I could not install it using pip install Numberjack (could not find xml2-config). On the webpage it says "At the moment Numberjack is not supported on windows".
 * python [python-constraint](https://labix.org/python-constraint). Installation using pip install python-constraint works on windows
 * python[constraint](https://www.logilab.org/project/logilab-constraint). Can be installed using  *pip install logilab-constraint* (from [here](https://pypi.python.org/pypi/logilab-constraint/0.6.0)). documenation [here](https://www.logilab.org/3441)
 * [simpleai](http://simpleai.readthedocs.io/en/latest/index.html).  implements many of the artificial intelligence algorithms described on the book “Artificial Intelligence, a Modern Approach”, from Stuart Russel and Peter Norvig. It contains Constraint Satisfaction Problems algorithms.
 * [pycryptosat](https://www.msoos.org/cryptominisat4/) installation using pip install pycryptosat.   I get the error "cl : Command line error D8021 : invalid numeric argument '/Wno-unused-variable'"
 * [geocode_python](https://launchpad.net/gecode-python). installation: you need to install gecode first, add an environnement variable called GECODEDIR and then run pip install gecode-python
 * pure python implementation of [cassowary](https://github.com/pybee/cassowary). installation using pip install cassowary. documenation [here](http://cassowary.readthedocs.io/en/latest/) Not maintained anymore
 * [casuarius](https://github.com/enthought/casuarius). last commit in 2011.
 * [z3](https://github.com/Z3Prover/z3) theorem prover from microsoft. Not sure how to use it to solve SAT problem . installation: pip install z3
 * [PyEDA](https://github.com/cjdrake/pyeda)supports Multi-Dimensional Bit Vectors
 
I would like to be able to define my problem using numpy array to avoid writting loops oover pixels in python. For example i would like to be able to write the constraint a=b or c with a , b and c arrays of binary variables. I alo woul like to be able to use array broadcasting like in numpy.
Similarly to what has been done in https://github.com/martinResearch/PySparseLP to define LP problems. 
We could store a Conjunctive Normal From (CNF) into a scipy sparse matrix :

  * M(i,j)=0 : variable j does not appear in clause j 
  * M(i,j)=1 : variable j is positive in clause j 
  * M(i,j)=-1 : variable j is negated in clause j

then we could export it in a text file using the DIMACS CNF SAT format (described [here](https://www.dwheeler.com/essays/minisat-user-guide.html) or [here](http://www.domagoj-babic.com/uploads/ResearchProjects/Spear/dimacs-cnf.pdf)))and solve it with an exeternal solver that can read that format (see a list below) 
we would still need to convert ou logic expressions into CNF. I would like it to support arbitrary nested expression combining AND, OR and Logical biconditional (often abreviated iff).
maybe we can reuse a tool to convert a logical expression to CNF from python (which might not need to know that we are dealing with arrays )
 
 * http://tt.brianwel.ch/en/latest/
 * https://github.com/asgordon/DPLL , quite simple , would all to go from string expression to CNF
 * https://github.com/tyler-utah/PBL it can be tested online here http://formal.cs.utah.edu:8080/pbl/PBL.php
 * https://github.com/omkarkarande/CNF-Converter
 * https://github.com/samidakhani/cnf_converter
 * python CNF converter [here](http://aima.cs.berkeley.edu/python/logic.html) from the book *Artificial Intelligence: A Modern Approach*  by Stuart Russell and Peter Norvig			
 * sympy. Example of conversions to CNF [here](http://docs.sympy.org/latest/modules/logic.html)				
 
executable that can solve DIMACS CNF SAT format 

  * [minisat](http://minisat.se/MiniSat.html) windows executable requires cygwin, which seems a bit heavy to install, it would be good to have a self-contained windows executable
  * [minion]( https://constraintmodelling.org/minion/ ). there is a windows executable.
  * [rsat](http://reasoning.cs.ucla.edu/rsat/download_new.php ). RSat is a complete Boolean satisfiability solver. RSat uses the phase selection heuristic that is oriented toward reducing work repetition and a frequent restart policy. Installation : you need to enter you email in a form to get the windows executable. 
  * [ubcsat](http://ubcsat.dtompkins.com/). UBCSAT is a Stochastic Local Search SAT Solver framework from the University of British Columbia. There is a windows executable
 
 more tool on the wikipedia page [Boolean_satisfiability_problem](see https://en.wikipedia.org/wiki/Boolean_satisfiability_problem)
 
