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
