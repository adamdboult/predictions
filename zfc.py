#########
# To do #
#########
# get arbitrary function to work with *something*, even if just equals
# set up initial sets, including of functions
# merge func array and setarray, include flag for being an array of functions for use, and funcuse, update func to be paths to this
# initial sets should have
## functions
## input data
## subsets (eg 4 in list of numbers)
## dictionaries
## empty set

# axioms, add:
## copy set
## add to set
## remove from set
## update set
## select thing from set?

# define functions in reference to others, low complexity


# logic goal:
## world generates data at t1
## view some function of t1
## some function on t1 creates t2
## prediction on t2 made on view on t1
## view on t2 taken
## accuracy compared

## these parellals also work for layers of abstraction, maybe some function does similar to much more complex function (eg first guess for addition not optimal, another could be compared to previous)

##########
# Import #
##########
import copy
from axioms import *
from dictionaries import *
from functions import *

#########
# Logic #
#########
def arbFunc(funcUse, updateFunc,funcArray,setArray,funcInd,setInd):
    # get arguments
    args = []
    for f in funcInd:
        args.append(funcArray[f])
    for s in setInd:
        args.append(setArray[s])
    #call function
    funcUse(*args)
    # update state based on result
    updateFunc(funcUse, updateFunc, funcArray, setArray, funcInd, setInd)
    # check if another function should be run
    if (funcUse):
        arbFunc(funcUse, updateFunc, funcArray, setArray, funcInd, setInd)
    #return
    return

#########
# Input #
#########
mySet=[\
    [\
        ["v",[]],\
        ["t","n"]\
    ],\
    [\
        ["v",[[]]],\
        ["t","n"]\
    ]\
]

#######
# Run #
#######
print (mySet)
answer = loopList(mySet,matchByKey,["v", seq(empty)])
print (answer)

dictExpand(answer, numeralsL, "v")
print (answer)
